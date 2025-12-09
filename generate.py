import argparse
import json
import torch
import numpy as np

from tokenizer import Tokenizer
from nn.transformer_lm import TransformerLM
from nn.rope import RoPE
from utils.load_checkpoint import load_checkpoint
from utils.sampling import softmax_with_temperature, top_p_sampling


@torch.no_grad()
def generate_text(model: TransformerLM, tokenizer: Tokenizer, prompt: str, context_length: int = 128, max_new_tokens: int = 100, temperature: float = 1.0, top_p: float = 1.0, device: str = "cpu") -> str:
    """
    Generate text from a trained language model.
    """
    model.eval()
    prompt_tokens_list = tokenizer.encode(prompt)
    prompt_tokens_tensor_batch = torch.tensor(prompt_tokens_list, device=device).unsqueeze(0)
    if prompt_tokens_tensor_batch.shape[-1] > context_length:
        prompt_tokens_tensor_batch = prompt_tokens_tensor_batch[..., -context_length:]
    endoftext_id = tokenizer.bytes_to_id[b"<|endoftext|>"]
    for i in range(max_new_tokens):
        logits = model(prompt_tokens_tensor_batch)[0][-1]
        probs = top_p_sampling(softmax_with_temperature(logits, temperature), top_p)
        next_token = torch.multinomial(probs, num_samples=1)
        if next_token.item() == endoftext_id:
            break
        prompt_tokens_tensor_batch = torch.cat([prompt_tokens_tensor_batch, next_token.unsqueeze(0)], dim=1)
        if prompt_tokens_tensor_batch.shape[-1] > context_length:
            prompt_tokens_tensor_batch = prompt_tokens_tensor_batch[..., -context_length:]
    return tokenizer.decode(prompt_tokens_tensor_batch[0].cpu().tolist())


def load_tokenizer(vocab_path: str, merges_path: str, special_tokens: list[str] | None = None) -> Tokenizer:
    special_tokens = special_tokens or ["<|endoftext|>"]
    return Tokenizer.from_files(vocab_path, merges_path, special_tokens)


def main():
    parser = argparse.ArgumentParser()
    
    # Paths
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, required=True, help="Path to config.json from training")
    parser.add_argument("--vocab", type=str, default="vocab.json", help="Path to vocabulary file")
    parser.add_argument("--merges", type=str, default="merges.txt", help="Path to merges file")
    
    # Generation parameters
    parser.add_argument("--prompt", type=str, default="The quick brown fox", help="Input prompt")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p nucleus sampling threshold")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        cfg = json.load(f)
    
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Initialize model architecture
    assert cfg["d_model"] % cfg["num_heads"] == 0
    d_k = cfg["d_model"] // cfg["num_heads"]
    rope = RoPE(cfg["rope_theta"], d_k, cfg["context_length"], device=device)
    position_ids = torch.arange(cfg["context_length"], device=device).unsqueeze(0)
    rotary_fn = lambda x: rope(x, position_ids[:, :x.shape[-2]])
    
    model = TransformerLM(
        d_model=cfg["d_model"],
        num_heads=cfg["num_heads"],
        d_ff=cfg["d_ff"],
        vocab_size=cfg["vocab_size"],
        context_length=cfg["context_length"],
        num_layers=cfg["num_layers"],
        rotary_fn=rotary_fn
    ).to(device)
    
    # Load checkpoint
    load_checkpoint(args.checkpoint, model, None)
    print(f"Loaded checkpoint from {args.checkpoint}")
    
    # Load tokenizer
    tokenizer = load_tokenizer(args.vocab, args.merges)
    
    # Generate text
    generated_text = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        context_length=cfg["context_length"],
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        device=device
    )
    
    print(f"Prompt: {args.prompt}")
    print(f"Generated: {generated_text}")


if __name__ == "__main__":
    main()