import os
import sys
from typing import List

import gc
import fire
import torch
import transformers
from datasets import load_dataset
from velora.peft_zoo.helper import set_layers

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)
from transformers import LlamaForCausalLM, LlamaTokenizer
from utils.prompter import Prompter
from transformers.optimization import get_scheduler

def train(
    # model/data params
    base_model: str = "",  # the only required argument
    data_path: str = "yahma/alpaca-cleaned",
    output_dir: str = "./lora-alpaca",
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    cutoff_len: int = 256,
    val_set_size: int = 2000,
    # lora hyperparams
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = [
        "q_proj",
        "v_proj",
    ],
    # llm hyperparams
    train_on_inputs: bool = True,  # if False, masks out inputs in loss
    add_eos_token: bool = False,
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_watch: str = "",  # options: false | gradients | all
    wandb_log_model: str = "",  # options: false | true
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    prompt_template_name: str = "alpaca",  # The prompt template to use, will default to alpaca.
    # velora hyperparams
    velora_r: int = 1,
    velora_layers: str = 'vd',
    num_groups: int = 32,
    init_type: str = "batch_average_once",
    velora_scale: float = 0.1
):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Training Alpaca-LoRA model with params:\n"
            f"base_model: {base_model}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"val_set_size: {val_set_size}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"add_eos_token: {add_eos_token}\n"
            f"group_by_length: {group_by_length}\n"
            f"wandb_project: {wandb_project}\n"
            f"wandb_run_name: {wandb_run_name}\n"
            f"wandb_watch: {wandb_watch}\n"
            f"wandb_log_model: {wandb_log_model}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"prompt template: {prompt_template_name}\n"
        )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"
    gradient_accumulation_steps = batch_size // micro_batch_size

    prompter = Prompter(prompt_template_name)

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )

    use_wandb = False
    os.environ["WANDB_DISABLED"] = "true"
    
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    model = LlamaForCausalLM.from_pretrained(
        base_model,
        # load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map=device_map
    )

    tokenizer = LlamaTokenizer.from_pretrained(base_model)

    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference

    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"],
        )
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = prompter.generate_prompt(
                data_point["instruction"], data_point["input"]
            )
            tokenized_user_prompt = tokenize(
                user_prompt, add_eos_token=add_eos_token
            )
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            if add_eos_token:
                user_prompt_len -= 1

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # could be sped up, probably
        return tokenized_full_prompt

    model = prepare_model_for_kbit_training(model)
    
    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    # 
    # VeLoRA
    # =======================
    
    config = {
        'init_type': init_type,
        'rank': velora_r, 
        'num_groups': num_groups,
        'velora_scale': velora_scale,
        'layers': velora_layers
    }

    # modified in-place
    method = "velora+lora"
    set_layers(model, method, config)

    # =======================
    # -- End VeLoRA
    # 

    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=data_path)
    else:
        data = load_dataset(data_path)

    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = (
                False  # So the trainer won't try loading its state
            )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")
            exit()

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    if val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=42
        )
        train_data = (
            train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        )
        val_data = (
            train_val["test"].shuffle().map(generate_and_tokenize_prompt)
        )
    else:
        train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = None

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    class LayerWiseTrainer(transformers.Trainer):
        """Layer-wise updates for VeLoRA layers to reduce memory for weight updates.
        This is needed to bridge the theoretical and practical reduction in memory.
        The gradients reconstruction and weight updates for each layer will be fused.
        this is in contrast to reconstructing all the layer gradients first and then performing a global step update.
        However, since this is not needed for the VeLoRA logic itself we have left it out.
        """
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.layer_wise = False
            
        def create_optimizer(self):
            opt_model = self.model
    
            if self.optimizer is None:
                optimizer_cls, optimizer_kwargs = transformers.Trainer.get_optimizer_cls_and_kwargs(self.args)
    
                if self.layer_wise:
                    # fuse weight updates with backwards pass.
                    # this is needed to realise the memory reduction in practise.
                    self.optimizer_dict = {}
                    
                    # define an optimizer for each parameter p, and store them in optimizer_dict
                    for p in opt_model.parameters():
                        if p.requires_grad:
                            self.optimizer_dict[p] = optimizer_cls([p], **optimizer_kwargs)
    
                    # define a hook function to update the parameter p during the backward pass
                    def optimizer_hook(p):
                        if p.grad is None: 
                            return

                        self.optimizer_dict[p].step()
                        self.optimizer_dict[p].zero_grad()
                        # see self.create_scheduler(...)
                        self.scheduler_dict[p].step()
    
                    # register the hook onto every parameter
                    for p in opt_model.parameters():
                        if p.requires_grad:
                            p.register_post_accumulate_grad_hook(optimizer_hook)
    
                    # for compatibility
                    class DummyOptimizer(torch.optim.Optimizer):
                        def __init__(self, *args, **kwargs):
                            dummy_tensor = torch.randn(1, 1)
                            super().__init__([dummy_tensor], {"lr": 1e-3})
                    
                        def zero_grad(self, set_to_none = True):
                            pass
                    
                        def step(self, closure = None):
                            pass

                    self.optimizer = DummyOptimizer()
                else:
                    raise NotImplementedError
            else:
                self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
                
        def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
            if self.lr_scheduler is None:
                if self.layer_wise:
                    self.scheduler_dict = {}
                    for p in self.model.parameters():
                        if p.requires_grad:
                            self.scheduler_dict[p] = get_scheduler(
                                self.args.lr_scheduler_type,
                                optimizer=self.optimizer if optimizer is None else optimizer,
                                num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                                num_training_steps=num_training_steps
                            )
    
                    # for compatibility
                    class DummyScheduler:
                        def __init__(*args, **kwargs):
                            pass
                            
                        def get_last_lr(*args):
                            return [1e-5]

                        def step(*args):
                            pass

                        def state_dict(self):
                            dummy_tensor = torch.randn(1, 1)
                            return dummy_tensor

                    self.lr_scheduler = DummyScheduler(self.optimizer)
                else:
                    raise NotImplementedError
    
                self._created_lr_scheduler = True
                
            return self.lr_scheduler
        
        def training_step(self, model, inputs):
            """ overridden to add custom losses/behaviour """
            model.train()
            inputs = self._prepare_inputs(inputs)

            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)

            if self.args.n_gpu > 1:
                loss = loss.mean()
            
            # Perform backward pass
            # loss.backward()

            if self.do_grad_scaling:
                self.scaler.scale(loss).backward()
            elif self.use_apex:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                self.accelerator.backward(loss)

            # gc.collect()
            # torch.cuda.empty_cache()

            return loss.detach()

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            # gradient_accumulation_steps=1,
            warmup_steps=100,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=10,
            optim="adamw_torch",
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=1000 if val_set_size > 0 else None,
            save_steps=1000,
            output_dir=output_dir,
            save_total_limit=3,
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to="wandb" if use_wandb else None,
            run_name=wandb_run_name if use_wandb else None,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    model.config.use_cache = False

    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(
            self, old_state_dict()
        )
    ).__get__(model, type(model))

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    model.save_pretrained(output_dir)

    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )


    # ============================================
    # Evaluate MMLU after training for convenience
    # ============================================

    import numpy as np
    import pandas as pd
    import random
    import string
    from mmlu.categories import categories, subcategories

    choices = ["A", "B", "C", "D"]

    def format_subject(subject):
        l = subject.split("_")
        s = ""
        for entry in l:
            s += " " + entry
        return s

    def format_example(df, idx, include_answer=True):
        prompt = df.iloc[idx, 0]
        k = df.shape[1] - 2
        for j in range(k):
            prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
        prompt += "\nAnswer:"
        if include_answer:
            prompt += " {}\n\n".format(df.iloc[idx, k + 1])
        return prompt

    def gen_prompt(train_df, subject, k=-1):
        prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
            format_subject(subject)
        )
        if k == -1:
            k = train_df.shape[0]
        for i in range(k):
            prompt += format_example(train_df, i)
        return prompt

    @torch.no_grad()
    def eval_model(ntrain, subject, model, tokenizer, dev_df, test_df):
        cors = []
        all_probs = []
        answers = choices[: test_df.shape[1] - 2]

        for i in range(test_df.shape[0]):
            # get prompt and make sure it fits
            k = ntrain
            prompt_end = format_example(test_df, i, include_answer=False)
            train_prompt = gen_prompt(dev_df, subject, k)
            prompt = train_prompt + prompt_end

            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

            while input_ids.shape[-1] > 2048:
                k -= 1
                train_prompt = gen_prompt(dev_df, subject, k)
                prompt = train_prompt + prompt_end
                input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(
                    model.device
                )

            label = test_df.iloc[i, test_df.shape[1] - 1]

            logits = model(input_ids=input_ids).logits[0, -1]

            probs = (
                torch.nn.functional.softmax(
                    torch.tensor(
                        [
                            logits[tokenizer("A").input_ids[-1]],
                            logits[tokenizer("B").input_ids[-1]],
                            logits[tokenizer("C").input_ids[-1]],
                            logits[tokenizer("D").input_ids[-1]],
                        ]
                    ).float(),
                    dim=0,
                )
                .detach()
                .cpu()
                .numpy()
            )
            pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(probs)]

            cor = pred == label
            cors.append(cor)
            all_probs.append(probs)

        acc = np.mean(cors)
        cors = np.array(cors)

        all_probs = np.array(all_probs)
        print("Average accuracy {:.3f} - {}".format(acc, subject))

        return cors, acc, all_probs

    def clean_exp_name(x):
        x = x.replace("-", "_")
        x = x.replace("/", "_")
        return x

    unique_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))
    data_dir = "mmlu/data"
    save_dir = f"llama/mmlu/results/" + clean_exp_name(unique_id)
    ntrain = 5
    model_name = "llama"

    model.eval()
    subjects = sorted(
        [
            f.split("_test.csv")[0]
            for f in os.listdir(os.path.join(data_dir, "test"))
            if "_test.csv" in f
        ]
    )

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    all_cors = []
    subcat_cors = {
        subcat: [] for subcat_lists in subcategories.values() for subcat in subcat_lists
    }
    cat_cors = {cat: [] for cat in categories}

    for subject in subjects:
        dev_df = pd.read_csv(
            os.path.join(data_dir, "dev", subject + "_dev.csv"), header=None
        )[: ntrain]
        test_df = pd.read_csv(
            os.path.join(data_dir, "test", subject + "_test.csv"), header=None
        )

        cors, acc, probs = eval_model(ntrain, subject, model, tokenizer, dev_df, test_df)
        subcats = subcategories[subject]
        for subcat in subcats:
            subcat_cors[subcat].append(cors)
            for key in categories.keys():
                if subcat in categories[key]:
                    cat_cors[key].append(cors)
        all_cors.append(cors)

        test_df["{}_correct".format(base_model)] = cors
        for j in range(probs.shape[1]):
            choice = choices[j]
            test_df["{}_choice{}_probs".format(base_model, choice)] = probs[:, j]

        # save results
        with open(os.path.join(save_dir, "results.txt"), "a") as f:
            f.write("Average accuracy {:.3f} - {} \n".format(acc, subject))

    results = {"subcategories": {}, "categories": {}}
    for subcat in subcat_cors:
        subcat_acc = np.mean(np.concatenate(subcat_cors[subcat]))
        results["subcategories"][subcat] = subcat_acc
        print("Average accuracy {:.3f} - {}".format(subcat_acc, subcat))

    for cat in cat_cors:
        cat_acc = np.mean(np.concatenate(cat_cors[cat]))
        results["categories"][cat] = cat_acc
        print("Average accuracy {:.3f} - {}".format(cat_acc, cat))

    weighted_acc = np.mean(np.concatenate(all_cors))
    results["weighted_accuracy"] = weighted_acc

    print("Average accuracy: {:.3f}".format(weighted_acc))
    with open(os.path.join(save_dir, "results.txt"), "a") as f:
        f.write("Average accuracy: {:.3f} \n".format(weighted_acc))

if __name__ == "__main__":
    fire.Fire(train)
