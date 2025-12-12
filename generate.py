import argparse
import json
import torch
import numpy as np

from cs336_basics.tokenizer import Tokenizer
from cs336_basics.nn.transformer_lm import TransformerLM
from cs336_basics.nn.rope import RoPE
from cs336_basics.utils.load_checkpoint import load_checkpoint
from cs336_basics.utils.sampling import softmax_with_temperature, top_p_sampling


@torch.no_grad()
def generate_text(
    model: TransformerLM,
    tokenizer: Tokenizer,
    prompt: str,
    context_length: int = 256,
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    top_p: float = 0.9,
    device: str | None = None
) -> str:
    model.eval()

    # 1. Device Setup
    if device is None:
        device = next(model.parameters()).device

    # 2. Prepare Input
    prompt_tokens_list = tokenizer.encode(prompt)
    x = torch.tensor(prompt_tokens_list, device=device).unsqueeze(0)

    endoftext_id = tokenizer.bytes_to_id[b"<|endoftext|>"]

    # 3. Autocast Setup
    # Only use bf16 if on CUDA and supported.
    device_type = "cuda" if "cuda" in str(device) else "cpu"
    enable_autocast = (device_type == "cuda" and torch.cuda.is_bf16_supported())

    for _ in range(max_new_tokens):
        # print("====================")
        # print("x:", x)
        # Ensure context window compliance
        if x.shape[-1] > context_length:
            x = x[..., -context_length:]
        # print("x:", x)

        # A. Model Forward Pass (in BFloat16 for speed)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=enable_autocast):
            # ⚠️🚨 inefficiency, no KV Cache
            logits = model(x)

        # B. Sampling Logic (in Float32 for precision)
        last_logits = logits[0, -1, :].float()

        probs = softmax_with_temperature(last_logits, temperature)
        probs = top_p_sampling(probs, top_p)

        next_token = torch.multinomial(probs, num_samples=1)
        # print("next_token:", next_token)

        # C. Stop Condition
        if next_token.item() == endoftext_id:
            break

        # D. Append
        x = torch.cat([x, next_token.unsqueeze(0)], dim=1)

    return tokenizer.decode(x[0].cpu().tolist())


def load_tokenizer(vocab_path: str, merges_path: str, special_tokens: list[str] | None = None) -> Tokenizer:
    special_tokens = special_tokens or ["<|endoftext|>"]
    return Tokenizer.from_files(vocab_path, merges_path, special_tokens)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, required=True, help="Path to config.json from training")
    parser.add_argument("--vocab", type=str, default="vocab.json", help="Path to vocabulary file")
    parser.add_argument("--merges", type=str, default="merges.txt", help="Path to merges file")

    parser.add_argument("--prompt", type=str, default="The quick brown fox", help="Input prompt")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p nucleus sampling threshold")
    parser.add_argument("--seed", type=int, default=2113, help="Random seed")

    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        cfg = json.load(f)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Initialize model architecture
    assert cfg["d_model"] % cfg["num_heads"] == 0
    d_k = cfg["d_model"] // cfg["num_heads"]
    rope = RoPE(cfg["rope_theta"], d_k, cfg["context_length"], device=device)
    token_positions = torch.arange(cfg["context_length"], device=device).unsqueeze(0)
    rotary_fn = lambda x: rope(x, token_positions[:, :x.shape[-2]])

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

    generated_text = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        context_length=cfg["context_length"],
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p
    )

    print(f"Prompt: {args.prompt}")
    print(f"Generated: {generated_text}")


if __name__ == "__main__":
    main()