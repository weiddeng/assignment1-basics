import torch
from torch import Tensor
from jaxtyping import Float, Int


# dimension is the axis along which softmax is computed
def softmax(input: Float[Tensor, "..."], dimension: int):
    input_max, _ = input.max(dim=dimension, keepdim=True)
    input -= input_max
    input_exp = torch.exp(input)
    # .max() returns a pair while .sum() doesn't
    return input_exp / input_exp.sum(dim=dimension, keepdim=True)