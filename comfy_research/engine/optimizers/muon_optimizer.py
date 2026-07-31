"""Muon optimizer (single-device + aux Adam), adapted from Keller Jordan's reference implementation.

Source notebook pattern: Newton-Schulz orthogonalized momentum updates for 2D parameters,
with Adam-style updates for remaining parameters (biases, 1D tensors).

References:
- https://kellerjordan.github.io/posts/muon/
- https://github.com/KellerJordan/Muon/blob/master/muon.py
"""

from __future__ import annotations

import torch


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int) -> torch.Tensor:
    """Newton-Schulz iteration for orthogonalization (see Muon reference)."""
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    # bfloat16 matches the reference Muon recipe on GPU; CPU uses float32 for broad compatibility.
    work_dtype = torch.bfloat16 if G.is_cuda else torch.float32
    X = G.to(dtype=work_dtype)
    if G.size(-2) > G.size(-1):
        X = X.mT

    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X.to(dtype=G.dtype)


def muon_update(
    grad: torch.Tensor,
    momentum: torch.Tensor,
    *,
    beta: float = 0.95,
    ns_steps: int = 5,
    nesterov: bool = True,
) -> torch.Tensor:
    momentum.lerp_(grad, 1 - beta)
    if nesterov:
        # Match ``grad.lerp_(momentum, beta)`` without mutating ``grad`` (``p.grad``).
        update = torch.lerp(grad, momentum, beta)
    else:
        update = momentum
    if update.ndim == 4:
        update = update.view(len(update), -1)
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    update *= max(1, update.size(-2) / update.size(-1)) ** 0.5
    return update


def adam_update(
    grad: torch.Tensor,
    buf1: torch.Tensor,
    buf2: torch.Tensor,
    step: int,
    betas: tuple[float, float],
    eps: float,
) -> torch.Tensor:
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    buf1c = buf1 / (1 - betas[0] ** step)
    buf2c = buf2 / (1 - betas[1] ** step)
    return buf1c / (buf2c.sqrt() + eps)


class SingleDeviceMuonWithAuxAdam(torch.optim.Optimizer):
    """Non-distributed Muon + auxiliary Adam for 1D / non-matrix parameters."""

    def __init__(self, param_groups: list[dict]) -> None:
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0.0)
                group["ns_steps"] = int(group.get("ns_steps", 5))
                assert set(group.keys()) == {
                    "params",
                    "lr",
                    "momentum",
                    "weight_decay",
                    "use_muon",
                    "ns_steps",
                }
            else:
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0.0)
                assert set(group.keys()) == {
                    "params",
                    "lr",
                    "betas",
                    "eps",
                    "weight_decay",
                    "use_muon",
                }
        super().__init__(param_groups, {})

    @torch.no_grad()
    def step(self, closure=None):  # noqa: ARG002
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = muon_update(
                        p.grad,
                        state["momentum_buffer"],
                        beta=float(group["momentum"]),
                        ns_steps=int(group["ns_steps"]),
                        nesterov=True,
                    )
                    wd = float(group["weight_decay"])
                    if wd != 0:
                        p.mul_(1 - float(group["lr"]) * wd)
                    p.add_(update.reshape(p.shape), alpha=-float(group["lr"]))
            else:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] = int(state["step"]) + 1
                    betas = group["betas"]
                    assert isinstance(betas, tuple) and len(betas) == 2
                    update = adam_update(
                        p.grad,
                        state["exp_avg"],
                        state["exp_avg_sq"],
                        int(state["step"]),
                        (float(betas[0]), float(betas[1])),
                        float(group["eps"]),
                    )
                    wd = float(group["weight_decay"])
                    if wd != 0:
                        p.mul_(1 - float(group["lr"]) * wd)
                    p.add_(update, alpha=-float(group["lr"]))

        return loss


def build_muon_with_aux_adam(
    model: torch.nn.Module,
    *,
    lr: float,
    momentum: float,
    weight_decay: float = 0.0,
    ns_steps: int = 5,
    aux_betas: tuple[float, float] = (0.9, 0.95),
    aux_eps: float = 1e-10,
    aux_lr: float | None = None,
) -> SingleDeviceMuonWithAuxAdam:
    """Split parameters: ndim >= 2 -> Muon; ndim < 2 -> Adam (aux)."""
    muon_params: list[torch.nn.Parameter] = []
    adam_params: list[torch.nn.Parameter] = []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if p.ndim >= 2:
            muon_params.append(p)
        else:
            adam_params.append(p)

    groups: list[dict] = []
    if muon_params:
        groups.append(
            dict(
                params=muon_params,
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,
                use_muon=True,
                ns_steps=ns_steps,
            )
        )
    if adam_params:
        groups.append(
            dict(
                params=adam_params,
                lr=aux_lr if aux_lr is not None else lr,
                betas=aux_betas,
                eps=aux_eps,
                weight_decay=weight_decay,
                use_muon=False,
            )
        )
    if not groups:
        raise ValueError("Muon optimizer requires at least one trainable parameter.")
    return SingleDeviceMuonWithAuxAdam(groups)
