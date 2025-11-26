import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float, Int
from einops import rearrange


class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None):
        super().__init__()

        assert d_k % 2 == 0

        seq_position = torch.arange(max_seq_len, device=device)
        seq_position = rearrange(seq_position, "... -> ... 1")

        dimension_indices = torch.arange(d_k // 2, device=device)
        # Not (2 * dimension_indices - 1) / d_k as mentioned in the homework
        pseudo_inverse_phase = theta ** (2 * dimension_indices / d_k)
        pseudo_inverse_phase = rearrange(pseudo_inverse_phase, "... -> 1 ...")

        theta_i_k = seq_position / pseudo_inverse_phase

        cos = torch.cos(theta_i_k)
        sin = torch.sin(theta_i_k)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)


    # NOT implemented as a matrix multiplication for efficiency!
    def forward(self, x: Float[Tensor, "... seq_len d_k"], token_positions: Int[Tensor, "... seq_len"]) -> torch.Tensor:

        # PyTorch replaces axis 0 of self.cos with the entire shape of token_positions.
        cos_slice = self.cos[token_positions, :]
        sin_slice = self.sin[token_positions, :]
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        rotated_even = cos_slice * x_even - sin_slice * x_odd
        rotated_odd = sin_slice * x_even + cos_slice * x_odd
        stacked = torch.stack([rotated_even, rotated_odd], dim=-1)
        rotated_x = rearrange(stacked, "... penultimate last -> ... (penultimate last)")
        return rotated_x
