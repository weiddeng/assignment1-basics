import torch
from torch import Tensor
from jaxtyping import Float
from .softmax import softmax


def softmax_with_temperature(logits: Float[Tensor, "vocab_size"], temperature: float = 1.0) -> Float[Tensor, "vocab_size"]:
    assert temperature >= 0
    if temperature == 0:
        probabilities = torch.zeros_like(logits)
        probabilities[logits.argmax()] = 1.0
        return probabilities
    logits /= temperature
    return softmax(logits, -1)


def top_p_sampling(probs: Float[Tensor, "vocab_size"], p: float = 1.0) -> Float[Tensor, "vocab_size"]:
    """
    Apply nucleus (top-p) sampling to probability distribution.
    
    Args:
        probs: tensor of shape (vocab_size,) with probabilities
        p: float, cumulative probability threshold (0 < p <= 1)
    
    Returns:
        filtered_probs: tensor of shape (vocab_size,) with filtered and renormalized probabilities
    """
    sorted_probs, sorted_indices = probs.sort(dim=-1, descending=True)
    cumsum = sorted_probs.cumsum(dim=-1)
    mask_sorted = cumsum - sorted_probs >= p

    # Make sure mask_sorted is not masking everything
    mask_sorted[0] = False

    # Boolean indexing/filtering!
    indices_to_remove = sorted_indices[mask_sorted]

    prob_output = probs.clone()
    prob_output[indices_to_remove] = 0
    return prob_output / prob_output.sum()