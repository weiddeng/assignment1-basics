import torch
import os
from typing import IO, BinaryIO


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out: str | os.PathLike | BinaryIO | IO[bytes]):
    checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "iteration": iteration}
    torch.save(checkpoint, out)