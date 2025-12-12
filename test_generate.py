import torch
from cs336_basics.nn.transformer_lm import TransformerLM
from cs336_basics.nn.rope import RoPE
from generate import generate_text, load_tokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Testing on device: {device}")

cfg = {
    "d_model": 64,
    "num_heads": 2,
    "d_ff": 128,
    "num_layers": 2,
    "vocab_size": 50257,
    "context_length": 32,
    "rope_theta": 10000.0,
}

# Init model
d_k = cfg["d_model"] // cfg["num_heads"]
rope = RoPE(theta=cfg["rope_theta"], d_k=d_k, max_seq_len=cfg["context_length"], device=device)
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

# Load tokenizer
tokenizer = load_tokenizer("vocab.json", "merges.txt")

# Generate (will be gibberish since model is untrained)
prompt = "Once upon a time, "
print(f"Prompt: {prompt}")
print(f"Generating with untrained model...")

output = generate_text(
    model=model,
    tokenizer=tokenizer,
    prompt=prompt,
    context_length=cfg["context_length"],
    max_new_tokens=50,
    temperature=1.0,
    top_p=0.9,
    device=device
)

print(f"Output: {output}")
print("\nTest passed! (Output is gibberish because model is untrained)")