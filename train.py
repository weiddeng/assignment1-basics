import argparse
import os
from dataclasses import dataclass, asdict
import numpy as np
import torch
import wandb

from nn.transformer_lm import TransformerLM
from nn.rope import RoPE
from optim.adamw import AdamW
from utils.cross_entropy import cross_entropy
from utils.lr_cosine_schedule import lr_cosine_schedule
from utils.get_batch import get_batch
from utils.gradient_clipping import gradient_clipping
from utils.save_checkpoint import save_checkpoint
from utils.load_checkpoint import load_checkpoint


@dataclass
class TrainingConfig:
    d_model: int
    num_heads: int
    d_ff: int
    num_layers: int
    vocab_size: int
    context_length: int
    rope_theta: float

    batch_size: int
    lr_max: float
    lr_min: float
    weight_decay: float
    beta1: float
    beta2: float
    eps: float
    max_iters: int
    warmup_iters: int
    grad_clip: float

    train_path: str
    val_path: str
    out_dir: str
    resume: str
    log_interval: int
    save_interval: int


@torch.no_grad()
def evaluate_loss(model, data, config, device, eval_iters=50):
    model.eval()
    losses = []
    for _ in range(eval_iters):
        x, y = get_batch(data, config.batch_size, config.context_length, device)
        logits = model(x)
        loss = cross_entropy(y, logits)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses) if losses else 0.0


def train(cfg: TrainingConfig):
    wandb.init(project="cs336-assignment-1", config=asdict(cfg))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device}...")

    train_data = np.memmap(cfg.train_path, dtype=np.uint16, mode='r')
    val_data = np.memmap(cfg.val_path, dtype=np.uint16, mode='r')

    # Validate dataset token ranges (sample-based for performance)
    train_sample_size = min(1_000_000, len(train_data))
    val_sample_size = min(1_000_000, len(val_data))

    train_sample = train_data[:train_sample_size]
    val_sample = val_data[:val_sample_size]

    print(f"Max token in train sample ({train_sample_size} tokens): {train_sample.max()}")
    print(f"Min token in train sample ({train_sample_size} tokens): {train_sample.min()}")
    print(f"Max token in val sample ({val_sample_size} tokens): {val_sample.max()}")
    print(f"Min token in val sample ({val_sample_size} tokens): {val_sample.min()}")

    assert train_sample.max() < cfg.vocab_size, f"Train token {train_sample.max()} >= vocab_size {cfg.vocab_size}"
    assert val_sample.max() < cfg.vocab_size, f"Val token {val_sample.max()} >= vocab_size {cfg.vocab_size}"
    assert train_sample.min() >= 0, f"Train dataset has negative tokens: {train_sample.min()}"
    assert val_sample.min() >= 0, f"Val dataset has negative tokens: {val_sample.min()}"

    assert cfg.d_model % cfg.num_heads == 0
    d_k = cfg.d_model // cfg.num_heads
    rope = RoPE(cfg.rope_theta, d_k, cfg.context_length, device=device)

    # Precompute position indices to avoid recomputation on every forward pass
    position_ids = torch.arange(cfg.context_length, device=device).unsqueeze(0)
    rotary_fn = lambda x: rope(x, position_ids[:, :x.shape[-2]])

    model = TransformerLM(
        d_model=cfg.d_model,
        num_heads=cfg.num_heads,
        d_ff=cfg.d_ff,
        vocab_size=cfg.vocab_size,
        context_length=cfg.context_length,
        num_layers=cfg.num_layers,
        rotary_fn=rotary_fn
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=cfg.lr_max,
        betas=(cfg.beta1, cfg.beta2),
        eps=cfg.eps,
        weight_decay=cfg.weight_decay
    )

    iteration = 0
    if cfg.resume:
        iteration = load_checkpoint(cfg.resume, model, optimizer)
        iteration += 1
        print(f"Resumed, working on iteration {iteration} now")

    model.train()
    while iteration < cfg.max_iters:

        current_lr = lr_cosine_schedule(
            t=iteration,
            lr_max=cfg.lr_max,
            lr_min=cfg.lr_min,
            T_w=cfg.warmup_iters,
            T_c=cfg.max_iters
        )
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        x, y = get_batch(train_data, cfg.batch_size, cfg.context_length, device)

        optimizer.zero_grad()
        logits = model(x)
        loss = cross_entropy(y, logits)
        loss.backward()

        gradient_clipping(model.parameters(), cfg.grad_clip)

        optimizer.step()

        if iteration % cfg.log_interval == 0:
            val_loss = evaluate_loss(model, val_data, cfg, device)
            print(f"Iter {iteration} | Loss: {loss.item():.4f} | Val: {val_loss:.4f} | LR: {current_lr:.6f}")

            wandb.log({
                "train/loss": loss.item(),
                "val/loss": val_loss,
                "lr": current_lr,
                "iteration": iteration
            })

        if iteration > 0 and iteration % cfg.save_interval == 0:
            ckpt_path = os.path.join(cfg.out_dir, f"ckpt_{iteration}.pt")
            save_checkpoint(model, optimizer, iteration, ckpt_path)
            print(f"Saved -> {ckpt_path}")

        iteration += 1

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--vocab_size", type=int, default=50257)
    parser.add_argument("--context_length", type=int, default=128)
    parser.add_argument("--rope_theta", type=float, default=10000.0)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr_max", type=float, default=6e-4)
    parser.add_argument("--lr_min", type=float, default=6e-5)
    parser.add_argument("--max_iters", type=int, default=5000)
    parser.add_argument("--warmup_iters", type=int, default=100)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-8)

    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--val_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="./checkpoints")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=1000)

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    config = TrainingConfig(**vars(args))
    train(config)