# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch LLaMA model."""
import math
import copy
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import (
    AttentionMaskConverter,
    _prepare_4d_attention_mask,
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS, is_torch_greater_or_equal_than_1_13
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from transformers.utils.import_utils import is_torch_fx_available
from transformers.models.llama.configuration_llama import LlamaConfig

if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa


# This makes `_prepare_4d_causal_attention_mask` a leaf function in the FX graph.
# It means that the function will not be traced through and simply appear as a node in the graph.
if is_torch_fx_available():
    if not is_torch_greater_or_equal_than_1_13:
        import torch.fx

    _prepare_4d_causal_attention_mask = torch.fx.wrap(_prepare_4d_causal_attention_mask)

from .visual_processing import FocalLoss
from ..mm_utils import get_singleturn_query_text_hs,get_multiturn_query_text_hs_and_expanded_visuals,aggregate_text2visual_attention_per_turn, \
create_pseudo_labels_torch, get_batched_sub_images, interplot_img_feat, insert_sub_feat
from llava.ana_utils import get_bbox4src, get_bbox_from_noisy_map
from llava.model.multimodal_projector.builder import build_vision_projector

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LlamaConfig"


def get_visual_token_masks_where(hidden_states, image_token_positions):
    batch_size, seq_len, hidden_size = hidden_states.shape
    device = hidden_states.device

    # Create position indices tensor once
    position_indices = torch.arange(seq_len, device=device).expand(batch_size, -1)
    visual_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=device)

    # Create masks for each batch item
    for batch_idx, positions in enumerate(image_token_positions):
        start, end = positions[0], positions[-1] + 1
        # Create a mask where positions are within the range
        visual_mask[batch_idx] = (position_indices[batch_idx] >= start) & (position_indices[batch_idx] < end)

    return visual_mask

def get_visual_tokens(hidden_states, image_token_positions, dim=1024):
    '''
    Extract visual tokens and non-visual tokens from hidden states.

    Args:
        hidden_states: torch.Tensor with shape (batch_size, seq_len, hidden_size)
        image_token_positions: List[torch.LongTensor] with shape (batch_size, num_visual_tokens)
        dim: int, the dimension to trancated the visual tokens (default: 1024)

    Returns:
        visual_tokens: torch.Tensor with shape (batch_size, num_visual_tokens, dim)
                      or None if no visual tokens
        non_visual_tokens: torch.Tensor with shape (batch_size, num_non_visual_tokens, hidden_size)
        visual_mask: torch.BoolTensor with shape (batch_size, seq_len) indicating visual token positions
    '''
    batch_size, seq_len, hidden_size = hidden_states.shape
    device = hidden_states.device
    image_tokens = []
    non_image_tokens = []
    visual_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=device)
    if len(image_token_positions) == 0:
        return None, hidden_states, visual_mask
    for batch_idx, image_token_position in enumerate(image_token_positions):
        img_token_start, img_token_end = image_token_position[0], image_token_position[-1] + 1
        image_tokens.append(hidden_states[batch_idx, img_token_start:img_token_end])
        non_image_tokens.append(torch.cat([hidden_states[batch_idx, :img_token_start], hidden_states[batch_idx, img_token_end:]], dim=0))
        visual_mask[batch_idx, img_token_start:img_token_end] = True
    visual_tokens = torch.stack(image_tokens, dim=0)[:, :, :dim]
    non_visual_tokens = torch.stack(non_image_tokens, dim=0)
    return visual_tokens, non_visual_tokens, visual_mask


def map_visual_token_back(hidden_states, visual_tokens, visual_masks):
    '''
    Map visual tokens back to hidden states.

    Args:
        hidden_states: torch.Tensor with shape (batch_size, seq_len-num_visual_tokens, hidden_size)
        visual_tokens: torch.Tensor with shape (batch_size, num_visual_tokens, dim)
        visual_masks: torch.BoolTensor with shape (batch_size, seq_len)

    Returns:
        merged_hidden_states: torch.Tensor with shape (batch_size, seq_len, hidden_size)
    '''
    if visual_tokens==None: return hidden_states
    batch_size, non_visual_len, hidden_size = hidden_states.shape
    _, num_visual_tokens, visual_dim = visual_tokens.shape
    device = hidden_states.device

    # Get the original sequence length
    seq_len = visual_masks.shape[1]

    # Initialize the output tensor
    merged_hidden_states = torch.zeros((batch_size, seq_len, hidden_size), dtype=hidden_states.dtype, device=device)

    # For each batch item
    for batch_idx in range(batch_size):
        # Find the start and end of visual tokens (assuming they're contiguous)
        if visual_masks[batch_idx].any():
            visual_positions = torch.where(visual_masks[batch_idx])[0]
            img_token_start = visual_positions[0].item()
            img_token_end = visual_positions[-1].item() + 1

            # Handle dimension mismatch if visual_dim != hidden_size
            if visual_dim != hidden_size:
                # Pad or project the visual tokens to match hidden_size
                if visual_dim < hidden_size:
                    # Pad with zeros
                    padded_visual_tokens = torch.zeros(
                        (num_visual_tokens, hidden_size),
                        dtype=visual_tokens.dtype,
                        device=device
                    )
                    padded_visual_tokens[:, :visual_dim] = visual_tokens[batch_idx]
                    merged_hidden_states[batch_idx, img_token_start:img_token_end] = padded_visual_tokens
                else:
                    # Truncate
                    merged_hidden_states[batch_idx, img_token_start:img_token_end] = visual_tokens[batch_idx, :, :hidden_size]
            else:
                # Dimensions match, direct assignment
                merged_hidden_states[batch_idx, img_token_start:img_token_end] = visual_tokens[batch_idx]

            # Map non-visual tokens before and after the visual tokens
            merged_hidden_states[batch_idx, :img_token_start] = hidden_states[batch_idx, :img_token_start]
            merged_hidden_states[batch_idx, img_token_end:] = hidden_states[batch_idx, img_token_start:non_visual_len]
        else:
            # No visual tokens, just copy the hidden states
            merged_hidden_states[batch_idx] = hidden_states[batch_idx]

    return merged_hidden_states

def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    warnings.warn(
        "Calling `transformers.models.llama.modeling_llama._prepare_4d_attention_mask` is deprecated and will be removed in v4.37. Use `transformers.modeling_attn_mask_utils._prepare_4d_attention_mask"
    )
    return _prepare_4d_attention_mask(mask=mask, dtype=dtype, tgt_len=tgt_len)


def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    warnings.warn(
        "Calling `transformers.models.llama.modeling_llama._make_causal_mask` is deprecated and will be removed in v4.37. Use `transformers.models.llama.modeling_llama.AttentionMaskConverter._make_causal_mask"
    )
    return AttentionMaskConverter._make_causal_mask(
        input_ids_shape=input_ids_shape, dtype=dtype, device=device, past_key_values_length=past_key_values_length
    )


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))

        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        if torch.isnan(self.weight).any():
            print(f"  Fixing NaNs in RMSNorm weight")
            return hidden_states.to(input_dtype)
        return self.weight * hidden_states.to(input_dtype)


ALL_LAYERNORM_LAYERS.append(LlamaRMSNorm)


class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


class LlamaLinearScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        t = t / self.scaling_factor

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


class LlamaDynamicNTKScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids,  unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        self.multi_tower_mode = False

        self._init_rope()

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        image_token_ids: Optional[List[torch.LongTensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        bsz, q_len, _ = hidden_states.size()
        if self.config.pretraining_tp > 1:
            non_visual_tokens = hidden_states

            # Process non-visual tokens with tensor parallelism
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            # Apply tensor parallelism to non-visual tokens
            non_visual_query_states = [F.linear(non_visual_tokens, query_slices[i]) for i in range(self.config.pretraining_tp)]
            non_visual_query_states = torch.cat(non_visual_query_states, dim=-1)

            non_visual_key_states = [F.linear(non_visual_tokens, key_slices[i]) for i in range(self.config.pretraining_tp)]
            non_visual_key_states = torch.cat(non_visual_key_states, dim=-1)

            non_visual_value_states = [F.linear(non_visual_tokens, value_slices[i]) for i in range(self.config.pretraining_tp)]
            non_visual_value_states = torch.cat(non_visual_value_states, dim=-1)

            # If no visual tokens, use non-visual states directly
            query_states = non_visual_query_states
            key_states = non_visual_key_states
            value_states = non_visual_value_states
        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:

            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
            if kwargs.get('pass_kv_cache', None):
                past_key_value.key_cache[self.layer_idx] = past_key_value.key_cache[self.layer_idx][:,:,:-1]
                past_key_value.value_cache[self.layer_idx] = past_key_value.value_cache[self.layer_idx][:,:,:-1]

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            attention_mask = attention_mask[:, :, :q_len, :kv_seq_len]  # add this line
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states) #batch, num_heads, q_len, head_dim

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous() #

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum(
                [F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class LlamaFlashAttention2(LlamaAttention):
    """
    Llama flash attention module. This module inherits from `LlamaAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        image_token_ids: Optional[List[torch.LongTensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # LlamaFlashAttention2 attention does not support output_attentions
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

            # overwrite attention_mask with padding_mask
            attention_mask = kwargs.pop("padding_mask")

        output_attentions = False

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
        # to be able to avoid many of these transpose/reshape/view.
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        dropout_rate = self.attention_dropout if self.training else 0.0

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (LlamaRMSNorm handles it correctly)

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output = self._flash_attention_forward(
            query_states, key_states, value_states, attention_mask, q_len, dropout=dropout_rate
        )

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def _flash_attention_forward(
        self, query_states, key_states, value_states, attention_mask, query_length, dropout=0.0, softmax_scale=None
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`int`, *optional*):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        """
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1. For details, please see the comment in LlamaFlashAttention2 __init__.
            causal = self.is_causal and query_length != 1

        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=causal,
            )

            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            attn_output = flash_attn_func(
                query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=causal
            )

        return attn_output

    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )


class LlamaSdpaAttention(LlamaAttention):
    """
    Llama attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `LlamaAttention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    # Adapted from LlamaAttention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        image_token_ids: Optional[List[torch.LongTensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "LlamaModel is using LlamaSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                image_token_ids=image_token_ids,
                **kwargs
            )

        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)


        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        if query_states.shape[3] != 1 and kv_seq_len != position_ids[0, -1] +1:
            kv_seq_len += position_ids[0, -1] + 1 - kv_seq_len
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        if attention_mask is not None:
            attention_mask = attention_mask[:, :, :q_len, :kv_seq_len]  # add this line
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
            is_causal=self.is_causal and attention_mask is None and q_len > 1,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


LLAMA_ATTENTION_CLASSES = {
    "eager": LlamaAttention,
    "flash_attention_2": LlamaFlashAttention2,
    "sdpa": LlamaSdpaAttention,
}


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LLAMA_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        image_token_ids: Optional[List[torch.LongTensor]] = None,
        visual_masks: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)


        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            image_token_ids=image_token_ids,
            **kwargs,
        )

        hidden_states = residual + hidden_states


        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


LLAMA_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`LlamaConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class LlamaPreTrainedModel(PreTrainedModel):
    config_class = LlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


LLAMA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            Two formats are allowed:
            - a [`~cache_utils.Cache`] instance;
            - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
            shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
            cache format.

            The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
            legacy cache format will be returned.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class LlamaModel(LlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._use_sdpa = config._attn_implementation == "sdpa"
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing

        self.enable_twig = False
        self.post_init()

        if getattr(config, "enable_twig", False):
            self.enable_twig = True
            self.twig_K = config.twig_K  # K-th layer is the branch point (0-indexed)
            self.twig_T = config.twig_T
            self.sys_token_num = 35
            self.roi_conf_thresh = 0.15
            self.roi_loss = config.roi_loss if hasattr(config, "roi_loss") else 'bce'
            if self.roi_loss == 'mse': self.roi_conf_thresh = -0.15
            # Initialize twig layers from base model layers K+1 to K+T
            self.twig_layers = nn.ModuleList()
            for i in range(self.twig_T):
                # Assuming K is the index of the last frozen layer that provides input to twig
                # Original layers to copy from would be K+1, K+2, ..., K+T
                new_twig_layer = LlamaDecoderLayer(config, layer_idx=self.twig_K + i)
                self.twig_layers.append(new_twig_layer)
            self.twig_norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.roi_branch = config.roi_branch if hasattr(config, "roi_branch") else False
            self.roi_source = config.roi_source if hasattr(config, "roi_source") else "qk" #or 'qk'
            self.roi_super_type = config.roi_super_type if hasattr(config, "roi_super_type") else "v1" #or 'v1'
            self.roi_enable2stage = config.roi_enable2stage if hasattr(config, "roi_enable2stage") and not self.training else False
            self.inplace_subimage_pos = False
            self.is_debug = False
            ##mask for roi_map-->binanized roi map+resized original image-->clip encoder, need to set token budget for this setting
            self.upscale_method = config.up_scale_method if hasattr(config,"upscale_method") else 'bbox'  # bbox for roi_map-->connect region analysis-->bbox crop-->resize
            self.mask_upscale_budget = config.mask_upsample_budget if hasattr(config,"mask_upscale_budget") else 576
            self.roi2stage_max_ratio = config.roi2stage_max_ratio if hasattr(config, "roi2stage_max_ratio") else -1.0

            if self.roi_enable2stage: #note enable2 stage is a failed attempt to make 2-stage training work, we just need two_stage_vanilla
                self.two_stage_vanilla = True if not self.training else False
                for i in range(self.twig_K, len(self.layers)):
                    self.layers[i].self_attn.layer_idx = i + self.twig_T-1 #FIXME, the last twig layer does not involve in kv cache.
            self.roi_multi_head = config.roi_multi_head if hasattr(config, "roi_multi_head") else False
            if self.roi_super_type == 'lazy' and not self.roi_enable2stage:
                #assert self.roi_enable2stage is not True, "Lazy supervision is not compatible with 2-stage supervision."
                self.sdpa_layers = [2, 14]
                self.roi_lazy_layer = 14
                self.sink_lazy_layer = 2
                self.layers[self.sink_lazy_layer].self_attn = LLAMA_ATTENTION_CLASSES['sdpa'](config=config, layer_idx=self.sink_lazy_layer)
                self.layers[self.roi_lazy_layer].self_attn = LLAMA_ATTENTION_CLASSES['sdpa'](config=config, layer_idx=self.roi_lazy_layer)
    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        image_token_ids: Optional[List[torch.LongTensor]] = None,
        src_images: Optional[List] = None,
        encode_image_fn=None,
        image_processor=None,
        labels = None,
        default_projector = None
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        past_key_values_length = 0
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if self._use_flash_attention_2:
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self._use_sdpa and not output_attentions:
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )

        # embed positions
        hidden_states = inputs_embeds
        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None
        visual_masks = get_visual_token_masks_where(hidden_states, image_token_ids)
        visual_token_num = visual_masks[0].sum().item()
        labels_updated = labels.clone() if labels is not None else None

        if seq_length>1: self.visual_masks = visual_masks
        if self.enable_twig and 576>=visual_token_num>0:
            #before the twig layers, we need to run the first K layers
            hidden_states_init = hidden_states.clone()

            if self.roi_super_type == 'lazy' and self.training and not self.roi_enable2stage:
                if output_attentions is not False: print('warning: output_attentions is not None, but lazy supervision utilize output_attentions=False. Check the logit here.')
                all_self_attns = ()
                eager_attn_mask = _prepare_4d_causal_attention_mask(
                    attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
                )

            for decoder_layer in self.layers[:self.twig_K]:
                if output_hidden_states:
                    all_hidden_states += (hidden_states,)
                cur_layer_idx = decoder_layer.self_attn.layer_idx
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        decoder_layer.__call__,
                        hidden_states,
                        attention_mask if self.roi_super_type != 'lazy' or self.roi_enable2stage or cur_layer_idx not in self.sdpa_layers else eager_attn_mask,
                        position_ids,
                        past_key_values,
                        output_attentions if self.roi_super_type != 'lazy' or self.roi_enable2stage or cur_layer_idx not in self.sdpa_layers else True,
                        use_cache,
                        image_token_ids,
                        visual_masks
                    )
                else:
                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_values,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        image_token_ids=image_token_ids,
                        visual_masks=visual_masks,
                    )

                hidden_states = layer_outputs[0]
                if self.training and self.roi_super_type == 'lazy' and not self.roi_enable2stage and cur_layer_idx in self.sdpa_layers and output_attentions is False:
                    all_self_attns += (layer_outputs[1],)
                if use_cache:
                    next_decoder_cache = layer_outputs[2 if output_attentions else 1]
                if output_attentions:
                    all_self_attns += (layer_outputs[1],)

            #lazy supervision layers, only for training
            if self.roi_super_type == 'lazy' and self.training and not self.roi_enable2stage and self.twig_K<self.roi_lazy_layer:
                #FIXME: get self_attn from the last backbone layer for lazy supervision, fix this logic
                lazy_output_attentions = output_attentions
                lazy_attn_mask = attention_mask
                lazy_hidden_states = hidden_states
                for decoder_layer in self.layers[self.twig_K:self.roi_lazy_layer+1]:
                    if decoder_layer.self_attn.layer_idx == self.roi_lazy_layer:
                        lazy_attn_mask = eager_attn_mask
                        lazy_output_attentions = True
                    if self.gradient_checkpointing and self.training:
                        lazy_layer_outputs = self._gradient_checkpointing_func(decoder_layer.__call__,
                            lazy_hidden_states, lazy_attn_mask, position_ids, past_key_values,
                            lazy_output_attentions, use_cache, image_token_ids, visual_masks)
                    else:
                        lazy_layer_outputs = decoder_layer(lazy_hidden_states, attention_mask=lazy_attn_mask,
                            position_ids=position_ids, past_key_value=past_key_values, output_attentions=lazy_output_attentions,
                            use_cache=use_cache, image_token_ids=image_token_ids, visual_masks=visual_masks)
                    lazy_hidden_states = lazy_layer_outputs[0]
                target_attn = lazy_layer_outputs[1]
                all_self_attns += (target_attn,)

            # Twig layers
            twig_layers = self.twig_layers if self.roi_source == "hidden_states" else self.twig_layers[:-1]
            twig_hidden_states = hidden_states
            for twig_layer in twig_layers:
                if output_hidden_states:
                    all_hidden_states += (twig_hidden_states,)
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        twig_layer.__call__, twig_hidden_states, attention_mask, position_ids,
                        past_key_values, output_attentions, use_cache, image_token_ids, visual_masks)
                else:
                    layer_outputs = twig_layer(
                        twig_hidden_states, attention_mask=attention_mask, position_ids=position_ids, past_key_value=past_key_values,
                        output_attentions=output_attentions, use_cache=use_cache,image_token_ids=image_token_ids,visual_masks=visual_masks)

                twig_hidden_states = layer_outputs[0]

                if use_cache:
                    next_decoder_cache = layer_outputs[2 if output_attentions else 1]
                if output_attentions:
                    all_self_attns += (layer_outputs[1],)
            if self.roi_super_type == 'v1':
                twig_hidden_states = self.twig_norm(twig_hidden_states)

            if self.roi_enable2stage and seq_length>1:
                for i in range(self.twig_K, len(self.layers)):
                    self.layers[i].self_attn.layer_idx = i + self.twig_T-1
                ## get 2-stage sub-image features from the roi-twig-branch
                #assume roi source is qk now
                if labels is not None and self.training: query_hidden_states = get_singleturn_query_text_hs(twig_hidden_states, labels)
                else: query_hidden_states = twig_hidden_states[:, -1:]
                visual_tokens = twig_hidden_states[:, self.sys_token_num:self.sys_token_num + visual_token_num]
                q, k = self.twig_layers[-1].input_layernorm(query_hidden_states), self.twig_layers[-1].input_layernorm(visual_tokens)
                q, k = self.twig_layers[-1].self_attn.q_proj(q), self.twig_layers[-1].self_attn.k_proj(k)
                if self.roi_multi_head == False:
                    pred_roi = q @ k.transpose(1, 2)
                else:
                    query_states, key_states = q, k
                    bsz, q_len, _ = query_states.size()
                    k_len = key_states.size(1)
                    num_heads, head_dim = self.twig_layers[-1].self_attn.num_heads, self.twig_layers[-1].self_attn.head_dim
                    query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
                    key_states = key_states.view(bsz, k_len, num_heads, head_dim).transpose(1, 2)
                    mask_score = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)  # batch, num_heads, q_len, k_len
                    if self.roi_loss == 'mse':
                        mask_score = nn.functional.softmax(mask_score, dim=-1, dtype=torch.float32).to(query_states.dtype)
                    pred_roi = mask_score.mean(dim=1)  # average over heads, batch, 1, k_len

                feat_h, feat_w = int(math.sqrt(visual_token_num)), int(math.sqrt(visual_token_num))
                pred_roi = pred_roi.reshape(-1, feat_h, feat_w)  # 24x24 grid
                img_pos_id = position_ids[visual_masks] if self.inplace_subimage_pos else None
                sub_img_feats, sub_img_nums, sub_img_bboxes, \
                sub_img_pos_id, vis_bbox, blurred_map = get_batched_sub_images(pred_roi, src_images, image_processor, encode_image_fn, self.roi_conf_thresh,upscale_method=self.upscale_method)
                self.vis_bboxes = vis_bbox if len(vis_bbox.shape)==2 else vis_bbox.unsqueeze(0)
                self.blurred_map = blurred_map
                if sub_img_feats is not None:
                    sub_img_feats = default_projector(sub_img_feats)
                    sub_img_feats, sub_padding_mask, sub_img_pos_id = interplot_img_feat(sub_img_feats, sub_img_bboxes, self.roi2stage_max_ratio,self.upscale_method, sub_img_pos_id)

                if sub_img_feats is not None:
                    if self.two_stage_vanilla: hidden_states = hidden_states_init
                    ###inject sub-image feature to hidden_states
                    padding_side = self.config.padding_side if hasattr(self.config, "padding_side") else "right"
                    hidden_states, labels_updated, attention_mask, position_ids = insert_sub_feat(
                        hidden_states, sub_img_feats, sub_padding_mask, sub_img_nums, visual_token_num, self.sys_token_num, #sys tokens
                        labels, attention_mask, position_ids, padding_side=padding_side, sub_img_pos_id=sub_img_pos_id,
                    ) #check the logic of updating new labels and pos id here, it may not accurate
                    seq_length = hidden_states.shape[1]  # update seq_length after sub-image feature injection

                    #update mask due to length change
                    if self._use_flash_attention_2: attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
                    elif self._use_sdpa and not output_attentions:
                        attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(attention_mask,(batch_size, seq_length), inputs_embeds,past_key_values_length)
                    else:
                        attention_mask = _prepare_4d_causal_attention_mask(
                            attention_mask, (batch_size, seq_length), hidden_states, past_key_values_length
                        )

                if self.two_stage_vanilla:
                    if sub_img_feats is None: hidden_states = hidden_states_init
                    # clear kv cache as this is a new sequence
                    if self.twig_T>1:
                        tmp_keys, tmp_values = [], []
                        for i in range(self.twig_K, self.twig_K + self.twig_T-1):
                            tmp_k, tmp_v = past_key_values[i][0], past_key_values[i][1]
                            tmp_keys.append(tmp_k)
                            tmp_values.append(tmp_v)
                    past_key_values = None
                    past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                    for decoder_layer in self.layers:
                        if output_hidden_states:
                            all_hidden_states += (hidden_states,)

                        if self.gradient_checkpointing and self.training:
                            layer_outputs = self._gradient_checkpointing_func(
                                decoder_layer.__call__, hidden_states, attention_mask, position_ids,
                                past_key_values, output_attentions, use_cache, image_token_ids, visual_masks)
                        else:
                            layer_outputs = decoder_layer(
                                hidden_states, attention_mask=attention_mask, position_ids=position_ids,
                                past_key_value=past_key_values, output_attentions=output_attentions,
                                use_cache=use_cache,
                                image_token_ids=image_token_ids, visual_masks=visual_masks)

                        if len(past_key_values) == self.twig_K and self.twig_T>1:
                            cos, sin = decoder_layer.self_attn.rotary_emb(tmp_values[0], seq_len=tmp_values[0].shape[2])
                            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models

                            for i in range(self.twig_K, self.twig_K + self.twig_T-1):
                                past_key_values.update(tmp_keys[i - self.twig_K], tmp_values[i - self.twig_K], i, cache_kwargs)
                            # past_key_values.update(key12, value12, 12, cache_kwargs)
                            # past_key_values.update(key13, value13, 13, cache_kwargs)
                        hidden_states = layer_outputs[0]
                        if use_cache:
                            next_decoder_cache = layer_outputs[2 if output_attentions else 1]
                        if output_attentions:
                            all_self_attns += (layer_outputs[1],)

            else:
                hidden_states = twig_hidden_states

            if self.roi_enable2stage: hidden_states = self.norm(hidden_states)

        else:
            for decoder_layer in self.layers:
                if output_hidden_states:
                    all_hidden_states += (hidden_states,)
                if self.enable_twig and self.roi_enable2stage:
                    if seq_length> 1 and (visual_token_num==0 or visual_token_num>=577):
                        for i in range(self.twig_K, len(self.layers)):
                            self.layers[i].self_attn.layer_idx = i #+ self.twig_T - 1
                    else:
                        cur_layer_id = decoder_layer.self_attn.layer_idx
                        kv_len = past_key_values[cur_layer_id][0].shape[2] if past_key_values is not None else 0
                        if kv_len and int(position_ids[0, 0]) != kv_len:
                            position_ids[0, 0] =  kv_len
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        decoder_layer.__call__,
                        hidden_states,
                        attention_mask,
                        position_ids,
                        past_key_values,
                        output_attentions,
                        use_cache,
                        image_token_ids,
                        visual_masks
                    )
                else:
                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_values,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        image_token_ids=image_token_ids,
                        visual_masks=visual_masks,
                    )

                hidden_states = layer_outputs[0]

                if use_cache:
                    next_decoder_cache = layer_outputs[2 if output_attentions else 1]

                if output_attentions:
                    all_self_attns += (layer_outputs[1],)

            hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache
        if not return_dict or getattr(self.config, "get_kl_div", False):
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None), labels_updated

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        ), labels_updated


class LlamaForCausalLM(LlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

        self.enable_twig = False
        if getattr(config, "enable_twig", False):
            self.enable_twig = True
            self.twig_head = copy.deepcopy(self.lm_head)
            self.roi_branch = config.roi_branch if hasattr(config, "roi_branch") else False
            self.roi_source = config.roi_source if hasattr(config, "roi_source") else "hidden_states" #or 'qk'
            self.roi_loss = config.roi_loss if hasattr(config, "roi_loss") else 'bce'
            self.roi_super_type = config.roi_super_type if hasattr(config, "roi_super_type") else 'v1'
            self.roi_lazy_heads = [13,24,26]
            self.roi_lazy_layer = 2
            self.roi_multi_head = config.roi_multi_head if hasattr(config, "roi_multi_head") else False
            self.roi_enable2stage = config.roi_enable2stage if hasattr(config, "roi_enable2stage") else False


    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        image_token_ids: Optional[List[torch.LongTensor]] = None,
        roi_target_map: Optional[torch.LongTensor] = None,
        src_images: Optional[List] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs, labels = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            image_token_ids=image_token_ids,

            encode_image_fn=self.get_model().get_vision_tower(),
            src_images=src_images,
            image_processor=self.get_model().get_vision_tower().image_processor,
            labels = labels,
            default_projector = self.get_model().mm_projector
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            if self.enable_twig and not self.roi_enable2stage and self.training:
                # Use twig head for training
                lm_head_slices = self.twig_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
                logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
                logits = torch.cat(logits, dim=-1)
            else:
                lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
                logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
                logits = torch.cat(logits, dim=-1)
        else:
            if self.training and self.enable_twig and not self.roi_enable2stage:
                # Use twig head for training
                logits = self.twig_head(hidden_states)
            else:
                logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        visual_masks = get_visual_token_masks_where(hidden_states, image_token_ids)
        visual_token_num = visual_masks[0].sum().item()

        if labels is not None:
            if (not self.enable_twig) or self.roi_enable2stage or visual_token_num == 0:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = CrossEntropyLoss()
                shift_logits = shift_logits.view(-1, self.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels)
                if self.enable_twig and not self.roi_enable2stage and visual_token_num==0:
                    zero_scalar_multiplier = torch.tensor(0.0, device=shift_logits.device, dtype=shift_logits.dtype)
                    loss = shift_logits.sum() * zero_scalar_multiplier
            else:
                visual_tokens = hidden_states[visual_masks].view(-1, visual_token_num, hidden_states.size(-1))  # batch, seq_len, hidden_size
                #for single round conversation
                if self.roi_super_type == 'v1':
                    assert roi_target_map is not None, "roi_target_map must be provided for roi branch"
                    query_hidden_states = get_singleturn_query_text_hs(hidden_states, labels)
                elif self.roi_super_type == 'lazy': #generate pseudo label in forward process
                    assert outputs[-1] is not None, "Lazy self attention must be provided for lazy supervision"
                    lazy_self_attn = outputs[-1][-1].detach()  # last self attention output
                    lazy_sink_attn = outputs[-1][0].detach()  # first self attention output, [B, head, seq_len, seq_len]
                    system_end_idx = visual_masks.nonzero(as_tuple=True)[1][0].item()
                    image_end_idx = system_end_idx + visual_masks[0].sum().item()
                    lazy_sink_o2i = lazy_sink_attn[:, :, image_end_idx:, system_end_idx:image_end_idx].mean(dim=1).mean(dim=1) # [B, visual_token_num]

                    expanded_query_hs, expanded_vis_token, expanded_sink_attn = get_multiturn_query_text_hs_and_expanded_visuals(hidden_states, labels, visual_tokens, lazy_sink_o2i)
                    output2image_attn = aggregate_text2visual_attention_per_turn(labels, lazy_self_attn, visual_masks)
                    query_hidden_states, visual_tokens = torch.cat(expanded_query_hs), torch.cat(expanded_vis_token)
                    expanded_sink_attn, output2image_attn = torch.cat(expanded_sink_attn), torch.cat(output2image_attn)
                    output2image_attn = (output2image_attn[:, 24] + output2image_attn[:, 13] + output2image_attn[:, 26])/3 #TODO: rewrite this line
                    pseudo_labels = create_pseudo_labels_torch(expanded_sink_attn, output2image_attn, sink_thresh=1e-3, binary_coff=0.2, bg_coff=0.1) #TODO: check hyper-parameter here
                    roi_target_map = pseudo_labels['labels']
                    pass
                if self.roi_source == "qk":
                    q, k = self.model.twig_layers[-1].input_layernorm(query_hidden_states), self.model.twig_layers[-1].input_layernorm(visual_tokens)
                    q, k = self.model.twig_layers[-1].self_attn.q_proj(q), self.model.twig_layers[-1].self_attn.k_proj(k)
                    if self.roi_super_type == 'lazy':
                        q = q.unsqueeze(1)  # [batch, 1, dim]
                    if self.roi_multi_head == False:
                        mask_score = q @ k.transpose(-1, -2)  # batch, 1, dim
                    else:
                        query_states, key_states = q, k
                        bsz, q_len, _ = query_states.size()
                        k_len = key_states.size(1)
                        num_heads, head_dim = self.model.twig_layers[-1].self_attn.num_heads, self.model.twig_layers[-1].self_attn.head_dim
                        query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
                        key_states = key_states.view(bsz, k_len, num_heads, head_dim).transpose(1, 2)
                        mask_score = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim) # batch, num_heads, q_len, k_len
                        if self.roi_loss == 'mse':
                            mask_score = nn.functional.softmax(mask_score, dim=-1, dtype=torch.float32).to(query_states.dtype)
                        mask_score = mask_score.mean(dim=1)  # average over heads, batch, 1, k_len

                else: raise NotImplementedError(f"ROI source {self.roi_source} not implemented")
                # Flatten the tokens
                flat_scores = mask_score.contiguous().view(-1)  # Shape: (batch_size * visual_token_num)
                flat_original_labels = roi_target_map.contiguous().view(-1)  # Shape: (batch_size * visual_token_num), dtype likely int/long
                valid_labels_mask = (flat_original_labels != -100)
                scores_for_loss = flat_scores[valid_labels_mask]
                labels_for_loss = flat_original_labels[valid_labels_mask].to(dtype=scores_for_loss.dtype, device=scores_for_loss.device)

                if scores_for_loss.numel() > 0:  # Check if there are any valid elements to compute loss on
                    if self.roi_loss == 'bce':
                        loss_fct = torch.nn.BCEWithLogitsLoss()  # Default reduction is 'mean'
                    elif self.roi_loss == 'focal':
                        loss_fct = FocalLoss(gamma=2.0, reduction='mean')
                    elif self.roi_loss == 'mse':
                        labels_for_loss = (roi_target_map/roi_target_map.sum(dim=-1,keepdim=True)).contiguous().view(-1).to(dtype=scores_for_loss.dtype, device=scores_for_loss.device)*100
                        scores_for_loss = flat_scores*100
                        loss_fct = torch.nn.MSELoss()  # Default reduction is 'mean'
                    else: raise NotImplementedError(f"ROI loss {self.roi_loss} not implemented")
                    loss = loss_fct(scores_for_loss, labels_for_loss)
                else:
                    # If all labels were -100 (no valid elements for loss), loss is 0.
                    zero_scalar_multiplier = torch.tensor(0.0, device=flat_scores.device, dtype=flat_scores.dtype)
                    loss = flat_scores.sum() * zero_scalar_multiplier

        if not return_dict or getattr(self.config, "get_kl_div", False):
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
                max_cache_length = past_key_values.get_max_length()
                cache_length = max(past_key_values[-1][0].shape[2], cache_length)
                past_length = max(past_key_values[-1][0].shape[2], past_length)
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                # past_length = max(past_key_values[-1][0].shape[2], past_length)
                # cache_length = max(past_key_values[-1][0].shape[2], cache_length)
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                if attention_mask.shape[1] - past_length>0 or (not self.enable_twig and not self.roi_enable2stage):
                    input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
                else:
                    padding_attention_mask = attention_mask.new_ones(
                        (input_ids.shape[0], past_length + 1 - attention_mask.shape[1]), dtype=attention_mask.dtype
                    )
                    attention_mask = torch.cat([padding_attention_mask, attention_mask], dim=1)
                    input_ids = input_ids[:, -(attention_mask.shape[1] - past_length):]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past


@add_start_docstrings(
    """
    The LLaMa Model transformer with a sequence classification head on top (linear layer).

    [`LlamaForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    LLAMA_START_DOCSTRING,
)
class LlamaForSequenceClassification(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = LlamaModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
                sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(logits.device)
            else:
                sequence_lengths = -1

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
