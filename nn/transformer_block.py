import torch
from torch import nn
from torch import Tensor
from jaxtyping import Float, Int
from typing import Callable

from .rms_norm import RMSNorm
from .multihead_self_attention import MultiheadSelfAttention
from .swiglu import SwiGLUFeedForward


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, rotary_fn: Callable|None = None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rotary_fn = rotary_fn

        self.rms_norm_0 = RMSNorm(self.d_model)
        self.multihead_self_attention = MultiheadSelfAttention(self.d_model, self.num_heads)
        self.rms_norm_1 = RMSNorm(self.d_model)
        self.swiglu = SwiGLUFeedForward(self.d_model, self.d_ff)

    def forward(self, in_features: Float[Tensor, "... seq_len d_model"]):
        block_0_residual = self.multihead_self_attention(self.rms_norm_0(in_features), self.rotary_fn)
        block_0_out = in_features + block_0_residual
        block_1_residual = self.swiglu(self.rms_norm_1(block_0_out))
        return block_0_out + block_1_residual
