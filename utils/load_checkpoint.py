import os
from typing import IO, BinaryIO
import torch


def load_checkpoint(src: str | os.PathLike | BinaryIO | IO[bytes], model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    device = next(model.parameters()).device
    checkpoint = torch.load(src, map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["iteration"]