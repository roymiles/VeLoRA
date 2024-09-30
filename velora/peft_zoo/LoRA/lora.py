import torch
import math
from torch import nn
import timm
import transformers


class LoRALayer(nn.Module):
    def __init__(self, dims=768, rank=8, alpha=1):
        super().__init__()
        self.dims = dims
        self.rank = rank
        self.alpha = alpha

        self.w_a_linear_q = nn.Linear(dims, rank, bias=False)
        self.w_b_linear_q = nn.Linear(rank, dims, bias=False)

        self.w_a_linear_v = nn.Linear(dims, rank, bias=False)
        self.w_b_linear_v = nn.Linear(rank, dims, bias=False)
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight.data)

def forward_vit_lora_attn(self, x, enable_velora=False, independent_subtokens=False):
    B, N, C = x.shape

    """
        In timm it is implemented as
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        but we wish to add only to the q and v components.
    """
    if enable_velora:
        qkv = self.qkv(x)

        # Output 0 of SliceBackward0 is a view and is being modified inplace. This view was created inside a custom Function ...
        # tldr; we cannot use inplace += on qkv here
        if enable_velora:
            velora_kwargs = { "training": self.training, "scale_and_shift": False, "independent_subtokens": independent_subtokens }
            new_q = functional_velora_linear_fn(x, self.lora.w_a_linear_q, self.lora.velora_wrapper_q, **velora_kwargs)
            new_v = functional_velora_linear_fn(x, self.lora.w_a_linear_v, self.lora.velora_wrapper_v, **velora_kwargs)
        else:
            new_q = self.lora.w_a_linear_q(x)
            new_v = self.lora.w_a_linear_v(x)

        new_q = self.lora.w_b_linear_q(new_q)
        new_v = self.lora.w_b_linear_v(new_v)

        q, k, v = torch.split(qkv, self.lora.dims, dim=2)
        q = q + self.lora.alpha*new_q
        v = v + self.lora.alpha*new_v
        qkv = torch.cat([q, k, v], dim=2)

    else:
        qkv = self.qkv(x)

        new_q = self.lora.w_b_linear_q(self.lora.w_a_linear_q(x))
        new_v = self.lora.w_b_linear_v(self.lora.w_a_linear_v(x))
        qkv[:, :, : self.lora.dims]  += self.lora.alpha*new_q
        qkv[:, :, -self.lora.dims :] += self.lora.alpha*new_v
        
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
    x = self.proj(x)
    x = self.proj_drop(x)
    return x

def forward_roberta_lora_attn(
        self,
        hidden_states,
        attention_mask = None,
        head_mask = None,
        encoder_hidden_states = None,
        encoder_attention_mas = None,
        past_key_value = None,
        output_attentions = False,
        output_key_value = False,
        enable_velora = False,
        independent_subtokens = False,
        velora_layers = 'vd'
    ):
        from velora.peft_zoo.helper import functional_velora_linear_fn
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        # =========== (Ve)LoRA code ============ #
        if enable_velora:
            velora_kwargs = { "training": self.training, "scale_and_shift": False, "independent_subtokens": independent_subtokens }
            new_q = functional_velora_linear_fn(hidden_states, self.lora.w_a_linear_q, self.lora.velora_wrapper_q, **velora_kwargs) if 'q' in velora_layers else self.lora.w_a_linear_q(hidden_states)
            new_v = functional_velora_linear_fn(hidden_states, self.lora.w_a_linear_v, self.lora.velora_wrapper_v, **velora_kwargs) if 'v' in velora_layers else self.lora.w_a_linear_v(hidden_states)
        else:
            new_q = self.lora.w_a_linear_q(hidden_states)
            new_v = self.lora.w_a_linear_v(hidden_states)

        # project back up and add head-dim
        new_q = self.transpose_for_scores(self.lora.w_b_linear_q(new_q))
        new_v = self.transpose_for_scores(self.lora.w_b_linear_v(new_v))

        query_layer = query_layer + self.lora.alpha*new_q
        value_layer = value_layer + self.lora.alpha*new_v
        # ================================== #

        use_cache = past_key_value is not None
        if self.is_decoder or output_key_value:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]
            if use_cache:
                position_ids_l = torch.tensor(key_length - 1, dtype=torch.long, device=hidden_states.device).view(
                    -1, 1
                )
            else:
                position_ids_l = torch.arange(query_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(key_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r

            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in RobertaModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder or output_key_value:
            outputs = outputs + (past_key_value,)

        return outputs

# ** RobertaIntermediate **
# def forward_roberta_lora_ffn(self, hidden_states, enable_velora = False, independent_subtokens = False, velora_layers = 'vd'):
#     from velora.peft_zoo.helper import functional_velora_linear_fn
#     hidden_states = self.dense(hidden_states)
#     hidden_states = self.intermediate_act_fn(hidden_states)
#     return hidden_states

# ** RobertaOutput **
def forward_roberta_lora_ffn(self, hidden_states, input_tensor, enable_velora = False, independent_subtokens = False, velora_layers = 'vd'):
    from velora.peft_zoo.helper import functional_velora_linear_fn
    if enable_velora and 'd' in velora_layers:
        velora_kwargs = { "training": self.training, "scale_and_shift": False, "independent_subtokens": independent_subtokens }
        hidden_states = functional_velora_linear_fn(hidden_states, self.dense, self.dense.velora_wrapper, **velora_kwargs)
    else:
        hidden_states = self.dense(hidden_states)
        
    hidden_states = self.dropout(hidden_states)
    hidden_states = self.LayerNorm(hidden_states + input_tensor)
    return hidden_states