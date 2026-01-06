import argparse
import os
import json
import time
from dataclasses import dataclass, asdict
import numpy as np
import numpy.typing as npt
import torch
import wandb

from cs336_basics.nn.transformer_lm import TransformerLM
from cs336_basics.nn.rope import RoPE
from cs336_basics.optim.adamw import AdamW
from cs336_basics.utils.cross_entropy import cross_entropy
from cs336_basics.utils.lr_cosine_schedule import lr_cosine_schedule
from cs336_basics.utils.get_batch import get_batch
from cs336_basics.utils.gradient_clipping import gradient_clipping
from cs336_basics.utils.save_checkpoint import save_checkpoint
from cs336_basics.utils.load_checkpoint import load_checkpoint


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
    overfit_single_batch: bool


@torch.no_grad()
def evaluate_loss(model: TransformerLM, data: npt.NDArray[np.int_], config: TrainingConfig, device: str, eval_iters: int=50) -> float:
    model.eval()
    losses = []
    # multiple rounds for statistical significance
    for _ in range(eval_iters):
        x, y = get_batch(data, config.batch_size, config.context_length, device)
        # mixed-precision computation
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits = model(x)
            loss = cross_entropy(y, logits)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses) if losses else 0.0


def train(cfg: TrainingConfig):
    torch.manual_seed(2113)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(2113)
    np.random.seed(2113)

    # ~3x faster matmul on Ampere+ GPUs 4090, A100, H100, etc. negligible precision loss
    torch.set_float32_matmul_precision('high')

    wandb.init(project="cs336-assignment-1", config=asdict(cfg))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device}...")

    # deceptively simply but very powerful
    train_data = np.memmap(cfg.train_path, dtype=np.uint16, mode='r')
    val_data = np.memmap(cfg.val_path, dtype=np.uint16, mode='r')

    assert cfg.d_model % cfg.num_heads == 0
    d_k = cfg.d_model // cfg.num_heads
    rope = RoPE(theta=cfg.rope_theta, d_k=d_k, max_seq_len=cfg.context_length, device=device)

    # batch size 1 but it will broadcast to x's batch size in rope's forward
    token_positions = torch.arange(cfg.context_length, device=device).unsqueeze(0)
    rotary_fn = lambda x: rope(x, token_positions[..., :x.shape[-2]])

    model = TransformerLM(
        d_model=cfg.d_model,
        num_heads=cfg.num_heads,
        d_ff=cfg.d_ff,
        vocab_size=cfg.vocab_size,
        context_length=cfg.context_length,
        num_layers=cfg.num_layers,
        rotary_fn=rotary_fn
    ).to(device)

    # PyTorch 2.0+: "Free" 30-50% speedup on H100. Compile to fused Triton kernels.
    # Note: The first step will lag for ~60s while it compiles.
    if torch.cuda.is_available():
        model = torch.compile(model)

    optimizer = AdamW(
        model.parameters(),
        lr=cfg.lr_max,
        betas=(cfg.beta1, cfg.beta2),
        eps=cfg.eps,
        weight_decay=cfg.weight_decay
    )

    try:
        iteration = 0
        if cfg.resume:
            iteration = load_checkpoint(cfg.resume, model, optimizer)
            iteration += 1
            print(f"Resumed, working on iteration {iteration} now")

        model.train()

        start_time = time.time()

        overfit_x, overfit_y = None, None
        if cfg.overfit_single_batch:
            overfit_x, overfit_y = get_batch(train_data, cfg.batch_size, cfg.context_length, device)
            print("Overfitting to a single batch...")

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

            if cfg.overfit_single_batch:
                x, y = overfit_x, overfit_y
            else:
                x, y = get_batch(train_data, cfg.batch_size, cfg.context_length, device)

            optimizer.zero_grad(set_to_none=True)
            # mixed-precision computation
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits = model(x)
                loss = cross_entropy(y, logits)
            loss.backward()

            # clip gradient, meanwhile keep grad_norm for line plot
            grad_norm = gradient_clipping(model.parameters(), cfg.grad_clip)

            optimizer.step()
            wandb.log({"grad_norm": grad_norm}, step=iteration)

            if iteration % cfg.log_interval == 0 or iteration == cfg.max_iters - 1:
                elapsed = time.time() - start_time
                iter_per_sec = (iteration + 1) / elapsed

                if iteration < 10:
                    speed_str = "(compiling...)"
                else:
                    speed_str = f"{iter_per_sec:.2f}"

                val_loss = evaluate_loss(model, val_data, cfg, device)
                print(f"Iter {iteration} | train/loss: {loss.item():.4f} | val/loss: {val_loss:.4f} | LR: {current_lr:.6f} | iter/sec: {speed_str}")

                wandb.log({
                    "train/loss": loss.item(),
                    "val/loss": val_loss,
                    "lr": current_lr,
                    "iter_per_sec": iter_per_sec,
                    "iteration": iteration
                })

            if iteration % cfg.save_interval == 0 or iteration == cfg.max_iters - 1:
                ckpt_path = os.path.join(cfg.out_dir, f"ckpt_{iteration}.pt")
                save_checkpoint(model, optimizer, iteration, ckpt_path)
                print(f"Saved -> {ckpt_path}")
                # TODO: consolidate this into save_checkpoint
                wandb.save(ckpt_path)

            iteration += 1
    except KeyboardInterrupt:
        print("Training interrupted manually.")
    finally:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=2048) # best to be a multiple of 64
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--vocab_size", type=int, default=50257)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--rope_theta", type=float, default=10000.0)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr_max", type=float, default=6e-4)
    parser.add_argument("--lr_min", type=float, default=6e-5)
    parser.add_argument("--max_iters", type=int, default=5000)
    parser.add_argument("--warmup_iters", type=int, default=100) # 2% of max_iters
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-6) # 1e-8 can break when fp16

    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--val_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="./checkpoints")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--overfit_single_batch", action="store_true", help="Train on a single batch to test memorization")

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    config = TrainingConfig(**vars(args))

    config_path = os.path.join(args.out_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(asdict(config), f, indent=2)
    print(f"Saved config to {config_path}")

    train(config)