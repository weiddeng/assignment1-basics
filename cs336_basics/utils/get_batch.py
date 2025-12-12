import torch
from jaxtyping import Int
from torch import Tensor
from typing import Tuple
import numpy as np
import numpy.typing as npt


# A simple data feeder, stateless, sample "with replacement", but also "augmentation"
def get_batch(
        x: npt.NDArray[np.int_],
        batch_size: int,
        context_length: int,
        device: str
    ) -> Tuple[Int[Tensor, "batch_size context_length"], Int[Tensor, "batch_size context_length"]]:

    assert len(x) > context_length

    # Recall numpy ftw no for loop!
    ix = np.random.randint(len(x) - context_length, size=(batch_size, 1))
    # broadcasting
    ix_input = ix + np.arange(context_length)
    ix_label = ix + np.arange(1, context_length + 1)
    # fancy indexing
    input = x[ix_input]
    label = x[ix_label]

    input_tensor = torch.from_numpy(input.astype(np.int64))
    label_tensor = torch.from_numpy(label.astype(np.int64))

    if "cuda" in device:
        return input_tensor.pin_memory().to(device, non_blocking=True), label_tensor.pin_memory().to(device, non_blocking=True)
    return input_tensor.to(device), label_tensor.to(device)