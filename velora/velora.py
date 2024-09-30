"""
    This file contains the generic VeLoRA utility classes and functions.
"""

from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import peft
import gc
import transformers.models.llama as llama


def orthogonal_init(*shape):
    t = torch.empty(shape)
    nn.init.orthogonal_(t)
    return t

class VeLoRAWrapper(nn.Module):
    """
    Inputs:
    - rank : number of distinct projections
    - dims : input feature dimension size
    - num_groups : sub dividing the input token dim
    - target_dim : (optional) fixed projection to a different size
    - independent_subtokens : (not used) this is False in the original submission. Use independant vectors for each sub-token.
                                         will probably require a larger batch size for better initialisation.
    """

    def __init__(
            self,
            dims : int,
            rank : int = 1,
            num_groups : int = 32,
            init_type = "batch_average_once",
            target_dim = None,
            fp16 = True,
            independent_subtokens = False,
            device = 'cpu',
            **kwargs
        ):
        super(VeLoRAWrapper, self).__init__()

        assert init_type in ["uniform", "batch_average", "batch_average_once", "none"] or init_type[0:3] == "svd" or init_type[0:3] == "ema"
        self.init_type = init_type
        self.rank = rank
        self.dims = dims
        self.num_groups = num_groups
        self.dtype = torch.float16 if fp16 else torch.float32
        self.independent_subtokens = independent_subtokens

        # (optional) fixed projection to a different size
        project_dim = target_dim if target_dim is not None else dims // num_groups
        requires_projection = project_dim != dims // num_groups

        if requires_projection:
            self.project_in = nn.Linear(dims // num_groups, target_dim, bias=False)
            self.project_out = nn.Linear(target_dim, dims // num_groups, bias=False)
            torch.nn.init.orthogonal_(self.project_in.weight)
            self.project_out.weight = torch.linalg.pinv(self.project_in.weight)
            # torch.nn.init.orthogonal_(self.project_out.weight)

        self.has_projections = requires_projection

        # re-initialised in init_embed
        if independent_subtokens:
            embed = F.normalize(orthogonal_init(rank, num_groups, dims // num_groups), p=2, dim=2)
        else:
            embed = F.normalize(torch.rand(rank, dims // num_groups), p=2, dim=1)

        embed = embed.to(dtype=self.dtype, device=device)

        self.register_buffer('is_first_batch', torch.Tensor([True]))
        self.register_buffer('initted', torch.Tensor([self.init_type == "none"]))
        self.register_buffer('embed', embed)

    @torch.jit.ignore
    def init_embed_(self, data):
        if self.initted:
            return
        
        if self.init_type[0:3] == "svd":
            dims = data.shape[2]

            # SVD is not implemented for fp16
            # and we must also sample a subset to avoid OOM
            if data[0].shape[1] > 2048:
                sample_size = 2048
                indices = torch.randint(0, data[0].shape[0], (sample_size,))
                sampled_data = data[0][indices].float()
            else:
                # no sampling
                sampled_data = data[0].float()

            # do decomposition
            U, S, V = torch.svd(sampled_data)

            svd_init_variant = self.init_type.split("_")[1]
            if svd_init_variant == "U[0:C]":
                embed = U[0:self.codebook_size].view(1, self.codebook_size, dims).to(data[0].dtype)

            elif svd_init_variant == "U[:, 0:C]":
                embed = U[:, 0:self.codebook_size].view(1, self.codebook_size, dims).to(data[0].dtype)

            elif svd_init_variant == "V.t()[0:C]":
                embed = V.t()[0:self.codebook_size].view(1, self.codebook_size, dims).to(data[0].dtype)

            elif svd_init_variant == "V[:, 0:C]":
                embed = V[0:self.codebook_size].view(1, self.codebook_size, dims).to(data[0].dtype)

            else:
                raise Exception(f"Unknow svd init type: {svd_init_variant}")

        elif self.init_type in ["batch_average_once", "batch_average"]:
            assert self.rank == 1, "Only implemented for rank-1."
            embed = data.mean(0) 

        elif self.init_type[0:3] == "ema":
            assert self.rank == 1, "Only implemented for rank-1."
            decay = float(self.init_type.split("_")[1])

            if self.is_first_batch:
                embed = data.mean(0)
                self.is_first_batch.data.copy_(torch.Tensor([False]))
            else:
                decay = float(self.init_type.split("_")[1])
                embed = decay*self.embed + (1-decay)*data.mean(0)
            
        else:
            # default to init.orthogonal_ initialisation
            return

        # make sure normalised
        embed = F.normalize(embed, p=2, dim=-1)
        # embed = F.normalize(embed, p=2, dim=0)

        # gc.collect()
        torch.cuda.empty_cache()
        self.embed.data.copy_(embed)
        torch.cuda.empty_cache()

        if self.init_type != "batch_average" and self.init_type[0:3] != "ema":
            self.initted.data.copy_(torch.Tensor([True]))

    @torch.no_grad
    def forward(self, x):
        x = x.to(self.dtype)

        self.init_embed_(x)

        embed = self.embed.to(x.device)
        zp = torch.einsum('h s, r s -> h', x, embed)
        return zp

class Linear(torch.autograd.Function):
    """
        Baseline comparison
    """
    @staticmethod
    def forward(ctx, input, weight, bias):
        # input, weight = input.detach(), weight.detach()
        ctx.use_bias = False
        if bias is not None:
            ctx.use_bias = True
            bias = bias.detach()

        out = F.linear(input, weight, bias)

        ctx.input_shape = input.shape
        weight = weight.to(input.dtype)
        ctx.save_for_backward(input, weight)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
            grad_output: B x L x M
        """
        input, weight = ctx.saved_tensors

        # B x L x M @ M x D -> B x L x D
        weight = weight.to(grad_output.dtype)
        grad_input = torch.matmul(grad_output, weight)

        # B x M x L @ B x L x D -> B x M x D
        grad_weight = torch.matmul(torch.transpose(grad_output.to(input.dtype), 1, 2), input)

        # B x L x M -> B x M
        if ctx.use_bias:
            grad_bias = grad_output.sum(1)
        else:
            grad_bias = None

        # gc.collect()
        torch.cuda.empty_cache()
        return grad_input, grad_weight, grad_bias
    
class VeLoRA(torch.autograd.Function):
    """
        Can be used to overload the forward/backwards of any linear layer.
        See: https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html
    """
    @staticmethod
    def forward(ctx, input, weight, bias, zp, v, scale):
        """
            zp: projected inputs
            v: fixed vector direction

            input: B x L x D
            weight: M x D
            bias: M
            out: B x L x M
        """
        
        # normal forward pass
        ctx.use_bias = False
        if bias is not None:
            ctx.use_bias = True

        out = F.linear(input, weight, bias)
        # v = v.to(input.device)

        ctx.scale = scale
        ctx.input_shape = input.shape
        ctx.save_for_backward(zp, weight, v)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
            grad_output: B x L x M
        """
        zp, weight, v = ctx.saved_tensors

        # reconstruct the input
        v = v.to(zp.device)
        input_hat = torch.einsum('h, r s -> h s', zp, v).view(ctx.input_shape)

        # B x L x M @ M x D -> B x L x D
        weight = weight.to(grad_output.dtype)
        grad_input = torch.matmul(grad_output, weight)

        # B x M x L @ B x L x D -> B x M x D
        grad_weight = torch.matmul(torch.transpose(grad_output.to(input_hat.dtype), 1, 2), input_hat)

        # B x L x M -> B x M
        if ctx.use_bias:
            grad_bias = grad_output.sum(1)
        else:
            grad_bias = None
        
        # gc.collect()
        torch.cuda.empty_cache()
        return grad_input, ctx.scale*grad_weight, grad_bias, None, None, None

class VeLoRAScaleAndShift(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, zp, v, scale):
        """
            Layer (w . x) + b where . is the element-wise product.
            e.g. used in SSF PEFT.
        """
        
        # normal forward pass
        input, weight = input.detach(), weight.detach()
        ctx.use_bias = False
        if bias is not None:
            ctx.use_bias = True
            bias = bias.detach()

        # scale and shift, not matmul
        out = (input * weight) + bias

        ctx.input_shape = input.shape
        weight = weight.to(input.dtype)

        ctx.scale = scale
        ctx.save_for_backward(zp, weight, v)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
            grad_output: B x L x M
        """
        zp, weight, v = ctx.saved_tensors

        # reconstruct the input
        v = v.to(zp.device)
        input_hat = torch.einsum('h, s -> h s', zp, v).view(ctx.input_shape)

        # B x L x M @ M x D -> B x L x D
        weight = weight.to(grad_output.dtype)
        grad_input = grad_output * weight

        # B x M x L @ B x L x D -> B x M x D
        grad_weight = grad_output.to(input_hat.dtype) * input_hat

        # B x L x M -> B x M
        if ctx.use_bias:
            grad_bias = grad_output.sum(1)
        else:
            grad_bias = None

        gc.collect()
        torch.cuda.empty_cache()
        return grad_input, ctx.scale*grad_weight, grad_bias, None, None, None

def functional_velora_linear_fn(
    x, 
    # original layer
    weight, 
    bias,
    # lora parameters
    lora_dropout, 
    lora_A,
    lora_B,
    scaling,
    fan_in_fan_out,
    # velora parameters
    velora_wrapper, 
    training, 
    enable_velora=False,
    scale_and_shift=False,
    velora_scale=1.0,
    *args, **kwargs
):
    """
        Implements a LoRA layer with VeLoRA input feature projections to save more memory
        See: https://github.com/microsoft/LoRA/.../loralib/layers.py#L90
    """
    B, N, C = x.shape
    num_groups = velora_wrapper.num_groups

    def T(w):
        return w.transpose(0, 1) if fan_in_fan_out else w  

    # base layer
    result = F.linear(x, T(weight), bias=bias)  
    
    if training and enable_velora:
        lora_result = lora_dropout(x)

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
        # lora_result = lora_dropout(x)
        # lora_result = Linear.apply(lora_result, lora_A, None)
        # result += (lora_result @ lora_B.transpose(0, 1)) * scaling

        # original
        result += (lora_dropout(x) @ lora_A.transpose(0, 1) @ lora_B.transpose(0, 1)) * scaling

    return result

def velora_linear_fn(self, x, enable_velora=False, velora_scale=1.0):
    return functional_velora_linear_fn(
        x,
        self.weight, 
        self.bias,
        self.lora_dropout.default, 
        self.lora_A.default.weight,
        self.lora_B.default.weight,
        self.scaling['default'],
        self.fan_in_fan_out,
        self.velora_wrapper, 
        self.training,
        velora_scale=velora_scale
    )