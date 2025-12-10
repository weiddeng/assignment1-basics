import torch
from torch import Tensor
from jaxtyping import Float, Int
from einops import einsum, rearrange
import math
from .softmax import softmax


def scaled_dot_product_attention(
    Q: Float[Tensor, "... queries d_k"],
    K: Float[Tensor, "... keys d_k"],
    # Note values == keys
    V: Float[Tensor, "... values d_v"],
    mask: Float[Tensor, "... queries keys"] | None = None,
) -> Float[Tensor, "... queries d_v"]:
    d_k = Q.shape[-1]
    assert K.shape[-1] == d_k
    q_on_k = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys")
    scaled_q_on_k = q_on_k / math.sqrt(d_k)

    # It is better to scale first and then mask, due to -torch.inf may be implemented as -65504 which can be sqrt-ed down

    # Cannot do `if mask` as cannot bool on a tensor object with more than one value
    if mask is not None:
        scaled_q_on_k += torch.where(mask == 1, 0.0, -torch.inf)

    attention_prob = softmax(scaled_q_on_k, -1)

    assert V.shape[-2] == K.shape[-2]
    return einsum(attention_prob, V, "... queries keys, ... keys d_v -> ... queries d_v")