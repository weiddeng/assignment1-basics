import torch
import torch.nn as nn
from einops import einsum, rearrange


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(self.d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_dtype = x.dtype
        x = x.to(torch.float32)
        x_squared_sum = einsum(x, x, "... d_model, ... d_model -> ...")
        x_squared_sum = rearrange(x_squared_sum, "... -> ... 1")
        # Use torch.sqrt on a tensor while math.sqrt on a number
        x_rms = torch.sqrt(x_squared_sum / self.d_model + self.eps)
        x = x / x_rms
        x = x.to(x_dtype)
        return self.weight * x
