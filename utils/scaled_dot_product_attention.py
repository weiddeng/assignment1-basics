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
    q_on_k = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys")
    # Cannot do `if mask` as cannot bool on a tensor object with more than one value
    if mask is not None:
        q_on_k += torch.where(mask == 1, 0.0, -torch.inf)

    d_model = Q.shape[-1]
    scaled_q_on_k = q_on_k / math.sqrt(d_model)

    attention_prob = softmax(scaled_q_on_k, -1)

    return einsum(attention_prob, V, "... queries keys, ... keys d_v -> ... queries d_v")
