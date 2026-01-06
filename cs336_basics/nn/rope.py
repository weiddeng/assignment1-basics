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


# NOTE: the torch.complex intention below was great, but got UserWarning: Torchinductor does not support code generation for complex operators. Performance may be worse than eager.


# import torch
# import torch.nn as nn
# from torch import Tensor
# from jaxtyping import Float, Int
# from einops import rearrange


# # Using the better implementation now
# class RoPE(nn.Module):
#     def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None):
#         """theta is angular wavelength"""
#         super().__init__()

#         assert d_k % 2 == 0

#         # distance
#         seq_positions = torch.arange(max_seq_len, device=device).unsqueeze(-1)

#         # theta-based log
#         # 0 -> angular_wavelength = 1
#         # 1 -> angular_wavelength = theta
#         log_angular_wavelength = torch.arange(0, d_k, 2, device=device) / d_k
#         angular_wavelength = (theta ** log_angular_wavelength).unsqueeze(0)
#         # 1/angular_wavelength is the angular_wavenumber, measuring the radians of phase (change) per 1 unit distance
#         # theta_i_k is the radians of phase (change) per seq_positions units distance
#         theta_i_k = seq_positions / angular_wavelength

#         phase_vectors = torch.polar(torch.ones_like(theta_i_k), theta_i_k)
#         self.register_buffer("phase_vectors", phase_vectors, persistent=False)


#     # This is a matrix multiplication, but we rarely construct the matrix out and then multiply, for efficiency
#     def forward(self, x: Float[Tensor, "... seq_len d_k"], token_positions: Int[Tensor, "... seq_len"]) -> Float[Tensor, "... seq_len d_k"]:
#         phase_vectors_slice = self.phase_vectors[token_positions, :]
#         # torch.complex() doesn't support bfloat16, so cast to float32 temporarily
#         original_dtype = x.dtype
#         x = x.float()
#         x = torch.complex(x[..., 0::2], x[..., 1::2])
#         x_rotated = phase_vectors_slice * x
#         x_rotated = torch.view_as_real(x_rotated)
#         x_rotated = rearrange(x_rotated, "... penultimate last -> ... (penultimate last)")
#         return x_rotated.to(original_dtype)