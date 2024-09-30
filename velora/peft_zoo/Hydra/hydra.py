import random
import torch
from torch import nn
import timm
import math
import transformers

class HydraLayer(nn.Module):
    def __init__(
        self, in_features=768, hidden_dim=8, out_features=768, scale=1, do=0.0
    ):
        super().__init__()
        self.down_proj = nn.Linear(in_features, hidden_dim, bias=True)
        self.up_proj = nn.Linear(hidden_dim, out_features, bias=True)
        self.dropout = nn.Dropout(do)
        self.scale=scale

        nn.init.xavier_uniform_(self.down_proj.weight)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)
    
    def forward(self, x, enable_velora=False):
        from velora.peft_zoo.helper import functional_velora_linear_fn
        
        if enable_velora:
            x = functional_velora_linear_fn(x, self.down_proj, self.velora_wrapper, self.training)
            x = self.up_proj(self.dropout(x)) * self.scale
        else:
            x = self.up_proj(self.dropout(self.down_proj(x))) * self.scale

        return x

def forward_vit_hydra_ffn(self, x, enable_velora=False, independent_subtokens=False):
    x = self.fc1(x)
    x = self.act(x)
    x = self.drop1(x)

    par = self.hydra_mlp_par(x, enable_velora=enable_velora, independent_subtokens=independent_subtokens)
    x = self.fc2(x)
    seq = self.hydra_mlp_seq(x, enable_velora=False)

    x = self.drop2(x + par + seq)
    return x

def forward_vit_hydra_attn(self, x, enable_velora=False, independent_subtokens=False):
    B, N, C = x.shape
    qkv = self.qkv(x)
        
    qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    q, k = self.q_norm(q), self.k_norm(k)

    if self.fused_attn:
        x = nn.functional.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop.p,
        )
    else:
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

    x = x.transpose(1, 2).reshape(B, N, C)
    
    par = self.hydra_proj_par(x, enable_velora=enable_velora, independent_subtokens=independent_subtokens)
    x = self.proj(x)
    seq = self.hydra_proj_seq(x, enable_velora=False)

    x = self.proj_drop(x + par + seq)
    return x