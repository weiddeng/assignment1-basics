import torch
from torch import Tensor
from jaxtyping import Float
from .softmax import softmax


def softmax_with_temperature(logits: Float[Tensor, "vocab_size"], temperature: float = 1.0) -> Float[Tensor, "vocab_size"]:
    """
    Apply temperature scaling to logits and return softmax probabilities.
    
    Args:
        logits: tensor of shape (vocab_size,)
        temperature: float, temperature parameter for scaling
    
    Returns:
        probs: tensor of shape (vocab_size,) with softmax probabilities
    """
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
    sorted_probs, sorted_indices = probs.sort(descending=True)
    cumsum = sorted_probs.cumsum(dim=-1)
    mask_sorted = cumsum - sorted_probs >= p
    # Make sure mask_sorted not masking everything
    mask_sorted[0] = False
    sorted_probs[mask_sorted] = 0.0
    filtered_probs = torch.zeros_like(probs)
    filtered_probs.scatter_(dim=-1, index=sorted_indices, src=sorted_probs)
    return filtered_probs / filtered_probs.sum()