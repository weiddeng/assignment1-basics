import torch
from torch import nn
from jaxtyping import Float, Int
from torch import Tensor
from .linear import Linear


class UnEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model:int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.linear = Linear(d_model, vocab_size, device, dtype)

    def forward(self, in_features: Float[Tensor, "... seq_len d_model"]):
        return self.linear(in_features)