import torch
import numpy as np
import numpy.typing as npt


def get_batch(x: npt.NDArray[np.int_], batch_size: int, context_length: int, device: str):
    ix = torch.randint(len(x) - context_length, (batch_size,))
    input_batch = torch.stack([
        torch.from_numpy(x[pos.item():pos.item()+context_length].astype(np.int64)) for pos in ix
    ])
    label_batch = torch.stack([
        torch.from_numpy(x[pos.item()+1:pos.item()+context_length+1].astype(np.int64)) for pos in ix
    ])

    if "cuda" in device:
        return input_batch.pin_memory().to(device, non_blocking=True), label_batch.pin_memory().to(device, non_blocking=True)
    return input_batch.to(device), label_batch.to(device)
