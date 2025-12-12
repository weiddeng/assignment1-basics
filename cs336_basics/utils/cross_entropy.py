from jaxtyping import Float, Int
from torch import Tensor
import torch


# torch.gather selects specific elements of a tensor along A dimension. The index tensor must have the same number of dimensions as the input tensor.
# Do NOT log a small probability.
def cross_entropy(target_token_id: Int[Tensor, "..."], logits: Float[Tensor, "... vocab_size"]) -> Float[Tensor, ""]:
    logits_max, _ = logits.max(dim=-1, keepdim=True)
    logits = logits - logits_max

    from_numerator = torch.gather(logits, -1, target_token_id.unsqueeze(-1))
    from_denominator = torch.log(torch.exp(logits).sum(dim=-1, keepdim=True))

    neg_log_likelihood = -from_numerator + from_denominator
    return neg_log_likelihood.mean()