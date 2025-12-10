import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor
from .linear import Linear


class SwiGLUFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device:torch.device | None = None, dtype:torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = Linear(self.d_model, self.d_ff, device, dtype)
        self.w2 = Linear(self.d_ff, self.d_model, device, dtype)
        self.w3 = Linear(self.d_model, self.d_ff, device, dtype)

    def forward(self, x: Float[Tensor, "... seq_len d_model"]):
        # linear gate
        lift_gate = F.silu(self.w1(x))
        lift = lift_gate * self.w3(x)
        return self.w2(lift)