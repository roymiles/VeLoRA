<div align="center">
<h1>VeLoRA : Memory Efficient Training using Rank-1 Sub-Token Projections</h1>

[Roy Miles](https://roymiles.github.io/)<sup>1</sup>, 
[Pradyumna Reddy](https://preddy5.github.io/)<sup>1</sup>,
[Ismail Elezi](https://therevanchist.github.io/)<sup>1</sup>,
[Jiankang Deng](https://jiankangdeng.github.io/)<sup>1</sup>

<sup>1</sup> Huawei Noah's Ark Lab

ArXiv Preprint ([arXiv 2405.17991](https://arxiv.org/abs/2405.17991))

 <img src="figure.png" alt="VeLoRA overview." style="float: left; margin-right: 10px;"/>

</div>

#

This is the official implementation for reproducing the results in the VeLoRA NeurIPS 2024 paper. 

- 🤗 **Planning to integrate into the HuggingFace PEFT library soon**

## Structure

We provide the training code and weights for the two main sets of experiments - pre-training LLaMA on C4 and fine-tuning LLaMA on Alpaca.

```
.
├── configs/             # LLaMA architecture configs for the c4 experiments.
├── results/             # By default, all training runs will save results here.
├── data/mmlu/           
├── data/alpaca/         
├── scripts/             # Example scripts for reproducing results.
├── velora/              # All VeLoRA specific code. velora.py ...
├── velora/peft_zoo/     # Implementing other various PEFT methods with VeLoRA.
```

## Installation

Create using conda:
```Shell
conda create --name velora python=3.10
conda activate velora
pip install -r requirements.txt
```

## Simple example code
Here we give a simple example of using VeLoRA on the value and down projection layers. See [set_layers()](...) for more details. We have `32` groups for both the value and down projection and initialise `v` using an average from the first batch during training. The `method = velora+full` parameter indicates that no other PEFT method is used with VeLoRA. Finally, the `rank` parameter will always 1. Using a `rank` other than 1 would require some other initialisation strategy.

```python
from velora import set_layers

model = LlamaForCausalLM.from_pretrained("/output/path")
method = "velora+full"

velora_config = {
    'rank': 1,
    'num_groups': 32,
    'init_type': 'batch_average_once',
    'layers': 'vd'
}

set_layers(model, method, velora_config)
```

## Pre-training LLaMA on C4
Tuning the `num_groups` parameter provides a memory/performance trade-off.

| Model | Validation Perplexity (↓) | Weights / Log |
| --- | --- |  --- |
| 60M | 33.28 | [link](https://drive.google.com/drive/folders/1TU3r6QUUhsVtuJht1KzQDD26_XqJX1LY?usp=drive_link) |
| 130M | Coming soon. | Coming soon. |

## Fine-tuning LLaMA on Alpaca

| Model | Mean 5-shot MMLU Test Accuracy (↑) | Weights / Log |
| --- | --- |  --- |
| 7B [3 epochs] | 38.6 | [link](https://drive.google.com/drive/folders/1IIP3E5z6gbI9VggnpOshweAVhdCYuYq2?usp=drive_link) |
| 13B | Coming soon. | Coming soon. |

### Training and Evaluation
We provide some example scripts in `scripts/` for both the training and evaluation stages.

<details>
<summary>Run this command to fine-tune the <code>LLaMA</code> model on the <code>Alpaca</code> dataset :</summary>
<pre><code>CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python finetune_llama.py \
    --base_model 'huggyllama/llama-7b' \
    --data_path 'tatsu-lab/alpaca' \
    --output_dir './results/lora-alpaca-7b' \
    --batch_size 16 \
    --micro_batch_size 16 \
    --num_epochs 4 \
    --learning_rate 4e-4 \
    --cutoff_len 512 \
    --val_set_size 1024 \
    --lora_r 32 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --lora_target_modules '[down_proj,up_proj,gate_proj,q_proj,k_proj,v_proj,o_proj]' \
    --train_on_inputs \
    --group_by_length \
    --velora_r 1 \
    --velora_layers 'vd' \
    --num_groups 32 \
    --init_type 'batch_average_once' \
    --velora_scale 0.1</code></pre>
</details>

<details>
<summary>Run this command to pre-train the small <code>LLaMA</code> models on the <code>C4</code> dataset :</summary>
<pre><code>torchrun --standalone --nproc_per_node 4 pretrain_llama.py \
    --model_config configs/llama_60m.json \
    --lr 0.01 \
    --velora_r 1 \
    --velora_layers 'vd' \
    --num_groups 64,86 \
    --init_type batch_average_once \
    --peft_type velora+full \
    --batch_size 128 \
    --total_batch_size 512 \
    --num_training_steps 10000 \
    --warmup_steps 1000 \
    --weight_decay 0 \
    --dtype float32 \
    --eval_every 1000 \
    --optimizer velora \
    --velora_scale 1.</code></pre>
</details>

## ✏ Citation
If you find our paper and code useful in your research, please consider giving a star ⭐ and citation 📝.
```BibTeX
@inproceedings{miles2024velora,
    title={VeLoRA: Memory Efficient Training using Rank-1 Sub-Token Projections}, 
    author={Roy Miles and Pradyumna Reddy and Ismail Elezi and Jiankang Deng},
    year={2024},
    journal={NeurIPS}
}

```

## ❤️ Acknowledgements
Our codebase is built upon the [GaLore](https://github.com/jiaweizzhao/GaLore) and [Alpaca-LoRA](https://github.com/tloen/alpaca-lora/tree/main) projects. Great work!