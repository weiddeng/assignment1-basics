import torch
import torch.nn as nn
import torch.nn.functional as F
from .linear import Linear


class SwiGLUFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device:torch.device | None = None, dtype:torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = Linear(self.d_model, self.d_ff, device, dtype)
        self.w2 = Linear(self.d_ff, self.d_model, device, dtype)
        self.w3 = Linear(self.d_model, self.d_ff, device, dtype)

    def forward(self, x: torch.Tensor):
        x_w1 = self.w1(x)
        # x_w1 * F.sigmoid(x_w1)
        x_w1_silu = F.silu(x_w1)
        x_w3 = self.w3(x)
        return self.w2(x_w1_silu * x_w3)
