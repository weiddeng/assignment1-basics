from collections.abc import Iterable
import torch
import math


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    grads = [param.grad for param in parameters if param.grad is not None]
    if not grads:
        return
    # l2 norm is NOT RMS!
    l2_norm = math.sqrt(sum([torch.linalg.norm(g)**2 for g in grads]))

    if l2_norm > max_l2_norm:
        for g in grads:
            g.mul_(max_l2_norm / (l2_norm + 1e-6))