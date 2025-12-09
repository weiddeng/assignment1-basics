import torch
from tokenizer import Tokenizer
from nn.transformer_lm import TransformerLM
from nn.rope import RoPE
from generate import generate_text, load_tokenizer

device = "cpu"

# Small model config for quick testing
cfg = {
    "d_model": 64,
    "num_heads": 2,
    "d_ff": 128,
    "num_layers": 2,
    "vocab_size": 50257,
    "context_length": 32,
    "rope_theta": 10000.0,
}

# Build model
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

# Load tokenizer
tokenizer = load_tokenizer("vocab.json", "merges.txt")

# Generate (will be gibberish since model is untrained)
prompt = "Once upon a time"
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
