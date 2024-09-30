from functools import partial
import torch
from torch import nn
import timm
import torch.nn.functional as F
import pdb

from velora import Linear, VeLoRAWrapper, VeLoRA, VeLoRAScaleAndShift

import transformers
import transformers.models.llama as llama
import transformers.models.roberta as roberta
from pretraining.modeling_llama import LlamaDecoderLayer

def forward_llama_attn(
        self, 
        hidden_states,
        attention_mask = None,
        position_ids = None,
        past_key_value = None,
        output_attentions = False,
        use_cache = False,
        enable_velora = False,
        velora_scale = 1.0,
        velora_layers = 'vd',
        **kwargs
    ):
    assert transformers.__version__ == '4.31.0', "Only tested using transformer version 4.31.0"
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
    
    bsz, q_len, _ = hidden_states.size()

    if enable_velora:
        velora_kwargs = { "training": self.training, "scale_and_shift": False, "velora_scale": velora_scale }
        query_states = functional_velora_linear_fn(hidden_states, self.q_proj, self.velora_wrapper_q, **velora_kwargs) if 'q' in velora_layers else self.q_proj(hidden_states)
        value_states = functional_velora_linear_fn(hidden_states, self.v_proj, self.velora_wrapper_v, **velora_kwargs) if 'v' in velora_layers else self.v_proj(hidden_states)
        key_states = functional_velora_linear_fn(hidden_states, self.k_proj, self.velora_wrapper_k, **velora_kwargs) if 'k' in velora_layers else self.k_proj(hidden_states)
    else:
        # using either has the same memory overhead in practise and theory
        # query_states = self.q_proj(hidden_states)
        # value_states = self.v_proj(hidden_states)
        # key_states = self.k_proj(hidden_states)
        query_states = Linear.apply(hidden_states, self.q_proj.weight, self.q_proj.bias)
        value_states = Linear.apply(hidden_states, self.v_proj.weight, self.v_proj.bias)
        key_states = Linear.apply(hidden_states, self.k_proj.weight, self.k_proj.bias)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    if past_key_value is not None:
        # reuse k, v, self_attention
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_key_value = (key_states, value_states) if use_cache else None

    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
            )

    # WARNING: padding mask is ignored, causal is always applied
    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states, key_states, value_states, dropout_p=0.0, is_causal=True,
    )

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2)
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value

def forward_llama_mlp(self, x, enable_velora=False, velora_scale=1.0, velora_layers='vd'):
    z = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
    if enable_velora and 'd' in velora_layers:
        y = functional_velora_linear_fn(z, self.down_proj, self.velora_wrapper, self.training, scale_and_shift=False, velora_scale=velora_scale)
    else:
        y = Linear.apply(z, self.down_proj.weight, self.down_proj.bias)
        # y = self.down_proj(z)

    return y

def forward_vit_attn(self, x, enable_velora=False, velora_scale=1.0, velora_layers='vd'):
    B, N, C = x.shape

    if enable_velora and 'qkv' in velora_layers:
        # single merged linear layer for query, value, and key projection
        # so we must VeLoRA all or nothing
        qkv = functional_velora_linear_fn(x, self.qkv, self.velora_wrapper, self.training, scale_and_shift=False, velora_scale=velora_scale)
    else:
        qkv = Linear.apply(x, self.qkv.weight, self.qkv.bias)
        # qkv = self.qkv(x)
    
    qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    
    q, k, v = qkv.unbind(0)
    q, k = self.q_norm(q), self.k_norm(k)

    if self.fused_attn:
        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop.p if self.training else 0.
        )
    else:
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

    x = x.transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)

    return x

def forward_vit_mlp(self, x, enable_velora=False, velora_scale=1.0, velora_layers='vd'):
    x = self.fc1(x)
    x = self.act(x)
    x = self.drop1(x)
    x = self.norm(x)

    if enable_velora and 'd' in velora_layers:
        x = functional_velora_linear_fn(x, self.fc2, self.velora_wrapper, self.training, scale_and_shift=False, velora_scale=velora_scale)
    else:
        qkv = Linear.apply(x, self.fc2.weight, self.fc2.bias)
        # x = self.fc2(x)

    x = self.drop2(x)

    return x

def functional_velora_linear_fn(x, linear_layer, velora_wrapper, training, scale_and_shift=False, velora_scale=1.0):
    """
        implement VeLoRA for a single linear layer only
    """
    B, N, C = x.shape
    num_groups = velora_wrapper.num_groups

    if training:
        with torch.no_grad():
            # B x N x C -> (B x N) x C

            assert C % num_groups == 0, f"Feature depth C={C} must be divisible by the number of groups G={num_groups}"
            flattened = x.view(B * N * num_groups, C // num_groups)

            zp = velora_wrapper(flattened)

        _velora = VeLoRA if not scale_and_shift else VeLoRAScaleAndShift
        result = _velora.apply(x, linear_layer.weight, linear_layer.bias, zp, velora_wrapper.embed, velora_scale)

    else:
        # result = Linear.apply(x, linear_layer.fc2.weight, linear_layer.bias)
        result = linear_layer(x)

    return result

def functional_velora_w_lora_linear_fn(
    x, 
    # original layer
    # weight, 
    # bias,
    base_layer,
    # --lora parameters
    lora_dropout, 
    lora_A,
    lora_B,
    scaling,
    fan_in_fan_out,
    # --velora parameters
    velora_wrapper, 
    training, 
    scale_and_shift=False,
    velora_scale=1.0,
    *args, **kwargs
):
    """
        implement a LoRA layer with VeLoRA input feature projections to save more memory
        see: https://github.com/microsoft/LoRA/loralib/layers.py#L90
    """
    B, N, C = x.shape
    num_groups = velora_wrapper.num_groups

    def T(w):
        return w.transpose(0, 1) if fan_in_fan_out else w  

    # base layer
    # result = F.linear(x, T(weight), bias=bias)  
    result = base_layer(x)

    if training:
        lora_result = lora_dropout(x)
        # lora_result = x

        # reshape and project
        with torch.no_grad():
            # B x N x C -> (B x N) x C
            flattened = lora_result.view(B * N * num_groups, C // num_groups)

            zp = velora_wrapper(flattened)

        # modified backwards
        _velora = VeLoRA if not scale_and_shift else VeLoRAScaleAndShift
        lora_result = _velora.apply(lora_result, lora_A, None, zp, velora_wrapper.embed, velora_scale)

        # final up projection and scaling
        result += (lora_result @ lora_B.transpose(0, 1)) * scaling

    else:
        # VeLoRA only reduces memory for training
        # x = lora_dropout(x)
        result += (x @ lora_A.transpose(0, 1) @ lora_B.transpose(0, 1)) * scaling

    return result

def velora_w_lora_linear_fn(self, x, velora_scale=1.0):
    # VeLoRA w/ LoRA
    return functional_velora_w_lora_linear_fn(
        x,
        self.base_layer,
        self.lora_dropout.default, # or None?
        self.lora_A.default.weight,
        self.lora_B.default.weight,
        self.scaling['default'],
        self.fan_in_fan_out,
        self.velora_wrapper, 
        self.training,
        velora_scale=velora_scale
    )
    
def velora_linear_fn(self, x, velora_scale=1.0):
    # VeLoRA only
    return functional_velora_linear_fn(
        x,
        self,
        self.velora_wrapper, 
        self.training,
        velora_scale=velora_scale
    )

def set_layers(model, method, configs, device='cpu'):
    """
        PEFT surgery. Used for Alpaca/C4/Roberta/VTAB-1K experiments.
        Modify the ViT architecture to add in the appropriate adapters
        and update the forward pass functions accordingly

        model: ViT model 
        device: Device just for the embed `v`, which is small in size.
        configs: Dictionary with all the configs
    """
    from velora.peft_zoo.Hydra import HydraLayer, forward_vit_hydra_attn, forward_vit_hydra_ffn
    from velora.peft_zoo.LoRA import LoRALayer, forward_roberta_lora_attn, forward_roberta_lora_ffn
    from velora.peft_zoo.SSF import SSFLayer, forward_vit_ssf_attn, forward_vit_ssf_ffn

    if method not in ['full', 'velora+full', 'hydra', 'velora+hydra', 'lora', 'velora+lora', 'ssf', 'velora+ssf']:
        raise NotImplementedError(method)

    layers = configs['layers']
    enable_velora = True if method.startswith('velora+') else False
    velora_scale = configs['velora_scale']

    # num groups is either a single int or a list of ints
    if isinstance(configs['num_groups'], list):
        assert len(configs['num_groups']) == len(layers), "Must specify the number of groups for each layer type"
        num_groups = [int(g) for g in configs['num_groups']]
    else:
        num_groups = len(layers) * [int(configs['num_groups'])]

    wrapper_kwargs = {}
    if enable_velora: 
        wrapper_kwargs = { 
            "init_type": configs['init_type'],
            "rank": configs['rank'],
            "layers": configs['layers'],
            "device": device
        }

    for x in model.children():
        bound_method_attn = None
        bound_method_ffn = None
        bound_kwargs = { "velora_scale": configs['velora_scale'] }
        if type(x) == llama.modeling_llama.LlamaAttention:
            """ Used for LLaMA alpaca fine-tuning experiments """
            assert method in ["velora+lora"], "Only implemented with VeLoRA+LoRA."

            if 'v' in layers:
                dims = x.v_proj.in_features
                x.v_proj.velora_wrapper = VeLoRAWrapper(dims, **wrapper_kwargs)
                bound_method = partial(velora_w_lora_linear_fn.__get__(x.v_proj, x.v_proj.__class__), **bound_kwargs)
                setattr(x.v_proj, 'forward', bound_method)

            if 'k' in layers:
                dims = x.k_proj.in_features
                x.k_proj.velora_wrapper = VeLoRAWrapper(dims, **wrapper_kwargs)
                bound_method = partial(velora_w_lora_linear_fn.__get__(x.k_proj, x.k_proj.__class__), **bound_kwargs)
                setattr(x.k_proj, 'forward', bound_method)

            if 'q' in layers:
                dims = x.q_proj.in_features
                x.q_proj.velora_wrapper = VeLoRAWrapper(dims, **wrapper_kwargs)
                bound_method = partial(velora_w_lora_linear_fn.__get__(x.q_proj, x.q_proj.__class__), **bound_kwargs)
                setattr(x.q_proj, 'forward', bound_method)

        elif type(x) == llama.modeling_llama.LlamaMLP:
            """ Used for LLaMA alpaca fine-tuning experiments """
            assert method in ["velora+lora"], "Only implemented with VeLoRA+LoRA."
            
            if 'd' in layers:
                dims = x.down_proj.in_features
                x.down_proj.velora_wrapper = VeLoRAWrapper(dims, **wrapper_kwargs)
                bound_method = partial(velora_w_lora_linear_fn.__get__(x.down_proj, x.down_proj.__class__), **bound_kwargs)
                setattr(x.down_proj, 'forward', bound_method)
                
        if type(x) == LlamaDecoderLayer:
            """ Used for C4 experiments """
            assert method in ["full", "velora+full"], "Not yet implemented for other PEFT methods."

            # note on 60m model
            # (self_attn): LlamaAttention(
            #   (q_proj): Linear(in_features=512, out_features=512, bias=False)
            #   (k_proj): Linear(in_features=512, out_features=512, bias=False)
            #   (v_proj): Linear(in_features=512, out_features=512, bias=False)
            #   (o_proj): Linear(in_features=512, out_features=512, bias=False)
            #   (rotary_emb): LlamaRotaryEmbedding()
            # )
            # (mlp): LlamaMLP(
            #   (gate_proj): Linear(in_features=512, out_features=1376, bias=False)
            #   (down_proj): Linear(in_features=1376, out_features=512, bias=False)
            #   (up_proj): Linear(in_features=512, out_features=1376, bias=False)
            #   (act_fn): SiLUActivation()
            # )

            dims = x.self_attn.q_proj.in_features

            if 'q' in layers:
                x.self_attn.velora_wrapper_q = VeLoRAWrapper(dims, **wrapper_kwargs, num_groups=num_groups[layers.find('q')])

            if 'k' in layers:
                x.self_attn.velora_wrapper_k = VeLoRAWrapper(dims, **wrapper_kwargs, num_groups=num_groups[layers.find('k')])

            if 'v' in layers:
                x.self_attn.velora_wrapper_v = VeLoRAWrapper(dims, **wrapper_kwargs, num_groups=num_groups[layers.find('v')])

            kwargs = { "enable_velora": enable_velora, "velora_scale": velora_scale, "velora_layers": layers }
            bound_method_attn = partial(forward_llama_attn.__get__(x.self_attn, x.self_attn.__class__), **kwargs)

            if 'd' in layers:
                dims = x.mlp.down_proj.in_features
                x.mlp.velora_wrapper = VeLoRAWrapper(dims, **wrapper_kwargs, num_groups=num_groups[layers.find('d')])
                bound_method_ffn = partial(forward_llama_mlp.__get__(x.mlp, x.mlp.__class__), **kwargs)

            # FFN: https://github.com/huggingface/transformers/models/llama/modeling_llama.py#L239
            # ATN: https://github.com/huggingface/transformers/models/llama/modeling_llama.py#L285

            # update the forward functions
            if bound_method_ffn is not None:
                setattr(x.mlp, 'forward', bound_method_ffn)

            if bound_method_attn is not None:
                setattr(x.self_attn, 'forward', bound_method_attn)

        elif type(x) == timm.models.vision_transformer.Block:
            """ Used for VTAB-1K experiments """
            dims = x.attn.qkv.in_features
            if method in ['full', 'velora+full']:

                if method == 'velora+full':
                    x.attn.velora_wrapper = VeLoRAWrapper(dims, **wrapper_kwargs)
                    x.mlp.velora_wrapper = VeLoRAWrapper(x.mlp.fc2.in_features, **wrapper_kwargs)

                bound_method_attn = partial(forward_vit_attn.__get__(x.attn, x.attn.__class__), enable_velora=enable_velora, velora_scale=velora_scale)
                bound_method_ffn = partial(forward_vit_mlp.__get__(x.mlp, x.mlp.__class__), enable_velora=enable_velora, velora_scale=velora_scale)

            elif method in ['hydra', 'velora+hydra']:
                # see: https://github.com/extremebird/Hydra/blob/main/vtab1k/hydra.py
                x.mlp.hydra_mlp_par = HydraLayer(in_features=4 * dims, hidden_dim=configs['rank'], scale=configs['scale'], do=configs['dropout'], out_features=dims)
                x.mlp.hydra_mlp_seq = HydraLayer(in_features=dims, hidden_dim=configs['rank'], scale=configs['scale'], do=configs['dropout'], out_features=dims)
                x.attn.hydra_proj_par = HydraLayer(in_features=dims, hidden_dim=configs['rank'], scale=configs['scale'], do=configs['dropout'], out_features=dims)
                x.attn.hydra_proj_seq = HydraLayer(in_features=dims, hidden_dim=configs['rank'], scale=configs['scale'], do=configs['dropout'], out_features=dims)

                if method == 'velora+hydra':
                    x.mlp.hydra_mlp_par.velora_wrapper = VeLoRAWrapper(4 * dims, **wrapper_kwargs)
                    x.attn.hydra_proj_par.velora_wrapper = VeLoRAWrapper(dims, **wrapper_kwargs)

                bound_method_attn = partial(forward_vit_hydra_attn.__get__(x.attn, x.attn.__class__), enable_velora=enable_velora, velora_scale=velora_scale)
                bound_method_ffn = partial(forward_vit_hydra_ffn.__get__(x.mlp, x.mlp.__class__), enable_velora=enable_velora, velora_scale=velora_scale)

            elif method in ['lora', 'velora+lora']:
                # see: https://github.com/JamesQFreeman/LoRA-ViT
                x.attn.lora = LoRALayer(dims=dims, rank=configs['rank'], alpha=configs['alpha'])

                if method == 'velora+lora':
                    x.attn.lora.velora_wrapper_q = VeLoRAWrapper(dims, **wrapper_kwargs)
                    x.attn.lora.velora_wrapper_v = VeLoRAWrapper(dims, **wrapper_kwargs)

                bound_method_attn = partial(forward_vit_lora_attn.__get__(x.attn, x.attn.__class__), enable_velora=enable_velora, velora_scale=velora_scale)

            elif method in ['ssf', 'velora+ssf']:
                # see: https://github.com/dongzelian/SSF/blob/main/models/vision_transformer.py
                x.mlp.ssf = SSFLayer(hidden_features=4*dims, out_features=dims)
                x.attn.ssf = SSFLayer(hidden_features=3*dims, out_features=dims)
                
                if method == 'velora+ssf':
                    # only apply to the first scale and shift
                    x.mlp.ssf.velora_wrapper = VeLoRAWrapper(4*dims, configs['rank'],  configs['num_groups'], configs['init_type'])
                    x.attn.ssf.velora_wrapper = VeLoRAWrapper(3*dims, configs['rank'], 8*configs['num_groups'], configs['init_type'])

                bound_method_attn = partial(forward_vit_ssf_attn.__get__(x.attn, x.attn.__class__), enable_velora=enable_velora, velora_scale=velora_scale)
                bound_method_ffn = partial(forward_vit_ssf_ffn.__get__(x.mlp, x.mlp.__class__), enable_velora=enable_velora, velora_scale=velora_scale)
            
            else:
                raise Exception(f"PEFT method: '{method}' is currently not implemented.")
            
            # update the forward functions
            if bound_method_ffn is not None:
                setattr(x.mlp, 'forward', bound_method_ffn)

            if bound_method_attn is not None:
                setattr(x.attn, 'forward', bound_method_attn)

        elif type(x) == roberta.modeling_roberta.RobertaLayer:
            """ Used for RoBERTa GLUE experiments """
            # ATTN: self.attention
            # FFN:  self.intermediate
            dims = x.attention.self.query.in_features
            if method in ['lora', 'velora+lora']:
                x.attention.self.lora = LoRALayer(dims=dims, rank=configs['rank'], alpha=configs['alpha'])

                if method == 'velora+lora':

                    # lora layers must be appropriately enabled with 
                    # target_modules_list param in finetune_roberta.py 
                    if 'q' in layers:
                        x.attention.self.lora.velora_wrapper_q = VeLoRAWrapper(dims, **wrapper_kwargs, num_groups=num_groups[layers.find('q')])

                    if 'v' in layers:
                        x.attention.self.lora.velora_wrapper_v = VeLoRAWrapper(dims, **wrapper_kwargs, num_groups=num_groups[layers.find('v')])

                    if 'k' in layers:
                        x.attention.self.lora.velora_wrapper_k = VeLoRAWrapper(dims, **wrapper_kwargs, num_groups=num_groups[layers.find('k')])

                    # x.output is the real down projection
                    # if 'd' in layers:
                    #     768 -> 3072
                    #     x.intermediate.self.lora.velora_wrapper = VeLoRAWrapper(dims, **wrapper_kwargs)

                    if 'd' in layers:
                        # e.g. 3072 -> 768
                        dims = x.output.dense.in_features
                        x.output.dense.velora_wrapper = VeLoRAWrapper(dims, **wrapper_kwargs, num_groups=num_groups[layers.find('d')])

                kwargs = { "enable_velora": enable_velora, "velora_scale": velora_scale, "velora_layers": layers }
                bound_method_attn = partial(forward_roberta_lora_attn.__get__(x.attention.self, x.attention.self.__class__), **kwargs)
                bound_method_ffn = partial(forward_roberta_lora_ffn.__get__(x.output, x.output.__class__), **kwargs)

            else:
                raise Exception(f"PEFT method: '{method}' is currently not implemented.")

            # update the forward functions
            if bound_method_ffn is not None:
                setattr(x.output, 'forward', bound_method_ffn)

            if bound_method_attn is not None:
                setattr(x.attention.self, 'forward', bound_method_attn)

        elif len(list(x.children())) != 0:
            # recursively call set_layers on children
            set_layers(x, method, configs, device)