import torch
from torch import Tensor
from jaxtyping import Float, Int


# dimension is the axis along which softmax is computed
def softmax(input: Float[Tensor, "..."], dimension: int):
    input_translate = input - input.max(dim=dimension, keepdim=True).values
    input_exp = torch.exp(input_translate)
    # .max() returns a pair while .sum() doesn't
    return input_exp / input_exp.sum(dim=dimension, keepdim=True)