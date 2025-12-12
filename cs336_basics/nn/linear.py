import torch
import torch.nn as nn
import math
from einops import einsum


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device:torch.device | None = None, dtype:torch.dtype | None = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # This needs no T and is the opposite of Torch implementation, per homework ask
        # torch.empty is the modern and preferred way to create an uninitialized tensor
        # The point of nn.Parameter is to automatically register a tensor as a learnable parameter of a model.
        self.weight = nn.Parameter(torch.empty(self.in_features, self.out_features, device=device, dtype=dtype))
        self.reset_parameters()

    # Convention: encapsulate weight reinitialization and called in __init__
    def reset_parameters(self) -> None:
        # Xavier/Glorot initialization, from "Understanding the difficulty of training deep feedforward neural networks (2010)"
        # - weights should be initialized so that forward activations and backward gradients stay in the same magnitude range across depth,
        # has to do with the singular values of each layer's Jacobian
        standard_deviation = math.sqrt(2/(self.in_features + self.out_features))
        torch.nn.init.trunc_normal_(self.weight, mean=0, std=standard_deviation, a=-3*standard_deviation, b=3*standard_deviation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Same as x @ self.weight
        return einsum(x, self.weight, "... d_in, d_in d_out -> ... d_out")
