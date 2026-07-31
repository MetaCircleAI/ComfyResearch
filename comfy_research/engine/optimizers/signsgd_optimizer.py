from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch


class SignSGD(torch.optim.Optimizer):
    """Minimal SignSGD optimizer: parameter -= lr * sign(gradient)."""

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        *,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        defaults = {"lr": float(lr), "weight_decay": float(weight_decay)}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Any | None = None) -> Any | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = float(group["lr"])
            weight_decay = float(group["weight_decay"])
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                if grad.is_sparse:
                    raise RuntimeError("SignSGD does not support sparse gradients.")
                if weight_decay != 0.0:
                    grad = grad.add(param, alpha=weight_decay)
                param.add_(grad.sign(), alpha=-lr)

        return loss
