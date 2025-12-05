import torch
from collections.abc import Callable
from typing import Optional
import math


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr, betas, eps, weight_decay):
        # hyperparams
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)


    # Disables gradient tracking globally for this function
    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr, betas, eps, weight_decay = group["lr"], group["betas"], group["eps"], group["weight_decay"]
            beta_1, beta_2 = betas

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    # mean, variance
                    state["m"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["v"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                state["step"] += 1
                t, m, v = state["step"], state["m"], state["v"]
                g = p.grad
                # Use in-place tensor updates, no new memory allocation for temp tensor!
                m.mul_(beta_1).add_(g, alpha=1-beta_1)
                v.mul_(beta_2).addcmul_(g, g, value=1-beta_2)
                step_size = lr * math.sqrt(1 - math.pow(beta_2, t)) / (1 - math.pow(beta_1, t))
                denom = v.sqrt().add_(eps)
                p.addcdiv_(m, denom, value=-step_size)
                p.mul_(1 - lr * weight_decay)
        # For Second-Order Optimizers (like L-BFGS or Conjugate Gradient) but dummy here
        return loss