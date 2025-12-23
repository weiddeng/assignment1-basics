import torch
import torch.nn as nn
from jaxtyping import Float, Int
from typing import Callable
from torch import Tensor
from einops import einsum, rearrange
from .linear import Linear
from cs336_basics.utils.scaled_dot_product_attention import scaled_dot_product_attention


# Compare w/ DeepSeek V2 MLA, and LoRA
class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, device:torch.device | None = None, dtype:torch.dtype | None = None):
        super().__init__()

        self.device = device
        self.dtype = dtype

        # We do the version where proj matrices are square
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = self.d_model // self.num_heads

        self.q_proj = Linear(self.d_model, self.d_model, self.device, self.dtype)
        self.k_proj = Linear(self.d_model, self.d_model, self.device, self.dtype)
        self.v_proj = Linear(self.d_model, self.d_model, self.device, self.dtype)
        self.o_proj = Linear(self.d_model, self.d_model, self.device, self.dtype)

    def forward(self, x: Float[Tensor, "... seq_len d_model"], rotary_fn: Callable | None = None):
        seq_len = x.shape[-2]
        # NO shift at QKV. The label seq shifts.
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device), diagonal=0)

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = rearrange(q, "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", num_heads=self.num_heads)
        k = rearrange(k, "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", num_heads=self.num_heads)
        v = rearrange(v, "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", num_heads=self.num_heads)
        if rotary_fn is not None:
            q = rotary_fn(q)
            k = rotary_fn(k)

        # The scaled_dot_product_attention is where to swap to the flash attention
        o = scaled_dot_product_attention(q, k, v, mask)
        o = rearrange(o, "... num_heads seq_len d_k -> ... seq_len (num_heads d_k)", num_heads=self.num_heads)

        return self.o_proj(o)