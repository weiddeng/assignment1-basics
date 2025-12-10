import torch
from torch import nn
from jaxtyping import Float, Int
from torch import Tensor
from typing import Callable

from .embedding import Embedding
from .transformer_block import TransformerBlock
from .rms_norm import RMSNorm
from .unembedding import UnEmbedding


class TransformerLM(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, vocab_size: int, context_length: int, num_layers: int, rotary_fn: Callable|None = None):
        super().__init__()
        self.context_length = context_length
        self.num_layers = num_layers
        self.embedding = Embedding(vocab_size, d_model)

        self.transformer_blocks = nn.ModuleList([
            # rotary_fn is plug-and-play
            TransformerBlock(d_model, num_heads, d_ff, rotary_fn) for _ in range(self.num_layers)
        ])

        self.rms_norm_final = RMSNorm(d_model)
        self.unembedding = UnEmbedding(vocab_size=vocab_size, d_model=d_model)


    def forward(self, token_ids: Int[Tensor, "... seq_len"]):
        features = self.embedding(token_ids)
        for block in self.transformer_blocks:
            features = block(features)
        return self.unembedding(self.rms_norm_final(features))