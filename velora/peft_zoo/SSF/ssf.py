"""
    Code: https://github.com/dongzelian/SSF/blob/main/models/vision_transformer.py
"""

import random
import torch
from torch import nn
import timm
from collections import namedtuple
import torch.nn.functional as F

def init_ssf_scale_shift(dim):
    scale = nn.Parameter(torch.ones(dim))
    shift = nn.Parameter(torch.zeros(dim))

    nn.init.normal_(scale, mean=1, std=.02)
    nn.init.normal_(shift, std=.02)

    return scale, shift

def ssf_ada(x, scale, shift):
    assert scale.shape == shift.shape
    if x.shape[-1] == scale.shape[0]:
        return x * scale + shift
    elif x.shape[1] == scale.shape[0]:
        return x * scale.view(1, -1, 1, 1) + shift.view(1, -1, 1, 1)
    else:
        raise ValueError('the input tensor shape does not match the shape of the scale factor.')
  
class SSFLayer(nn.Module):
    def __init__(self, hidden_features, out_features):
        super().__init__()
        self.ssf_scale_1, self.ssf_shift_1 = init_ssf_scale_shift(hidden_features)
        self.ssf_scale_2, self.ssf_shift_2 = init_ssf_scale_shift(out_features)

class ILinear:
    def __init__(self, weight, bias):
        super().__init__()
        self.weight = weight
        self.bias = bias

    def __call__(self, x):
        return (x * self.weight) + self.bias

def forward_vit_ssf_ffn(self, x, enable_velora=False):
    from velora.peft_zoo.helper import functional_velora_linear_fn
    x = self.fc1(x)

    if enable_velora:
        # emulate linear layer dynamically to preserve the same interface
        linear_layer = ILinear(weight=self.ssf.ssf_scale_1, bias=self.ssf.ssf_shift_1)
        velora_kwargs = { "training": self.training, "scale_and_shift": True, "independent_subtokens": independent_subtokens }
        x = functional_velora_linear_fn(x, linear_layer, self.velora_wrapper, **velora_kwargs)
    else:
        x = ssf_ada(x, self.ssf.ssf_scale_1, self.ssf.ssf_shift_1)

    x = self.act(x)
    x = self.drop1(x)
    x = self.fc2(x) 

    x = ssf_ada(x, self.ssf.ssf_scale_2, self.ssf.ssf_shift_2)
    
    x = self.drop2(x)
    
    return x

def forward_vit_ssf_attn(self, x, enable_velora=False):
    B, N, C = x.shape
    qkv = self.qkv(x)

    if enable_velora:
        linear_layer = ILinear(weight=self.ssf.ssf_scale_1, bias=self.ssf.ssf_shift_1)
        velora_kwargs = { "training": self.training, "scale_and_shift": True, "independent_subtokens": independent_subtokens }
        qkv = velora_linear_fn(qkv, linear_layer, self.velora_wrapper, **velora_kwargs)
    else:
        qkv = ssf_ada(qkv, self.ssf.ssf_scale_1, self.ssf.ssf_shift_1)

    # qkv: 512, 197, 2304
    # proj: 768 -> 768
    qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)

    attn = (q @ k.transpose(-2, -1)) * self.scale
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)

    x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)

    x = ssf_ada(x, self.ssf.ssf_scale_2, self.ssf.ssf_shift_2)

    x = self.proj_drop(x)
    return x