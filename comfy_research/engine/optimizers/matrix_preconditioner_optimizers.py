from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch


_EIGH_JITTER_SCALES = (1.0, 10.0, 100.0, 1000.0, 10000.0)


def _check_optimizer_hparams(
    *,
    lr: float,
    betas: tuple[float, float] | None = None,
    momentum: float | None = None,
    eps: float,
    weight_decay: float,
    precondition_frequency: int,
    max_preconditioner_dim: int,
) -> None:
    if lr < 0.0:
        raise ValueError(f"Invalid learning rate: {lr}")
    if betas is not None:
        beta1, beta2 = betas
        if not 0.0 <= beta1 < 1.0:
            raise ValueError(f"Invalid beta1 value: {beta1}")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"Invalid beta2 value: {beta2}")
    if momentum is not None and not 0.0 <= momentum < 1.0:
        raise ValueError(f"Invalid momentum value: {momentum}")
    if eps <= 0.0:
        raise ValueError(f"Invalid epsilon value: {eps}")
    if weight_decay < 0.0:
        raise ValueError(f"Invalid weight_decay value: {weight_decay}")
    if precondition_frequency < 1:
        raise ValueError(f"Invalid precondition_frequency value: {precondition_frequency}")
    if max_preconditioner_dim < 1:
        raise ValueError(f"Invalid max_preconditioner_dim value: {max_preconditioner_dim}")


def _unfold_along_dim(t: torch.Tensor, dim: int) -> torch.Tensor:
    return t.movedim(dim, 0).reshape(t.shape[dim], -1)


def _left_multiply_along_dim(t: torch.Tensor, mat: torch.Tensor, dim: int) -> torch.Tensor:
    out = torch.tensordot(mat, t, dims=([1], [dim]))
    return out.movedim(0, dim)


def _as_symmetric_float_matrix(mat: torch.Tensor) -> torch.Tensor:
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError(f"Expected a square matrix, got shape {tuple(mat.shape)}")
    work = mat.float()
    return (work + work.T) * 0.5


def _eigh_psd_with_fallback(mat: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    work = _as_symmetric_float_matrix(mat)
    jitter = max(float(eps), float(torch.finfo(work.dtype).eps))
    eye = torch.eye(work.shape[0], dtype=work.dtype, device=work.device)
    last_exc: RuntimeError | None = None
    for scale in _EIGH_JITTER_SCALES:
        try:
            return torch.linalg.eigh(work + (jitter * scale) * eye)
        except RuntimeError as exc:
            last_exc = exc

    if work.device.type != "cpu":
        # Low-rank Shampoo statistics can make CUDA eigh fail; retry in CPU double.
        cpu_work = work.detach().cpu().double()
        cpu_jitter = max(float(eps), float(torch.finfo(cpu_work.dtype).eps))
        cpu_eye = torch.eye(cpu_work.shape[0], dtype=cpu_work.dtype, device=cpu_work.device)
        for scale in _EIGH_JITTER_SCALES:
            try:
                evals, evecs = torch.linalg.eigh(cpu_work + (cpu_jitter * scale) * cpu_eye)
                return evals.to(device=work.device, dtype=work.dtype), evecs.to(
                    device=work.device, dtype=work.dtype
                )
            except RuntimeError as exc:
                last_exc = exc

    assert last_exc is not None
    raise last_exc


def _matrix_power_from_eigh(mat: torch.Tensor, power: float, eps: float) -> torch.Tensor:
    evals, evecs = _eigh_psd_with_fallback(mat, eps)
    evals = evals.clamp_min(float(eps)).pow(float(power))
    powered = (evecs * evals.unsqueeze(0)) @ evecs.T
    return powered.to(dtype=mat.dtype)


def _orthogonal_basis_from_eigh(mat: torch.Tensor, eps: float) -> torch.Tensor:
    evals, evecs = _eigh_psd_with_fallback(mat, eps)
    order = torch.argsort(evals, descending=True)
    return evecs.index_select(1, order).to(dtype=mat.dtype)


def _orthogonal_basis_from_power_iteration(
    mat: torch.Tensor,
    basis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    work = _as_symmetric_float_matrix(mat)
    old_basis = basis.float()
    estimated_evals = torch.diag(old_basis.T @ work @ old_basis)
    order = torch.argsort(estimated_evals, descending=True)
    sorted_basis = old_basis.index_select(1, order)
    power_iter = work @ sorted_basis
    refreshed, _ = torch.linalg.qr(power_iter)
    return refreshed.to(dtype=mat.dtype), order


def _preconditioner_roots(
    preconditioners: list[torch.Tensor | None],
    *,
    root_power: float,
    eps: float,
) -> list[torch.Tensor | None]:
    return [
        _matrix_power_from_eigh(preconditioner, root_power, eps)
        if preconditioner is not None
        else None
        for preconditioner in preconditioners
    ]


def _apply_dim_matrices(
    t: torch.Tensor,
    matrices: list[torch.Tensor | None],
    *,
    transpose: bool = False,
) -> torch.Tensor:
    out = t.float()
    for dim, mat in enumerate(matrices):
        if mat is None:
            continue
        op = mat.T if transpose else mat
        out = _left_multiply_along_dim(out, op.float(), dim)
    return out


def _projected_first_moment_in_new_basis(
    exp_avg: torch.Tensor,
    old_basis: list[torch.Tensor | None],
    new_basis: list[torch.Tensor | None],
) -> torch.Tensor:
    original_space = _apply_dim_matrices(exp_avg, old_basis)
    return _apply_dim_matrices(original_space, new_basis, transpose=True)


def _refresh_soap_basis_and_second_moment(
    preconditioners: list[torch.Tensor | None],
    old_basis: list[torch.Tensor | None],
    exp_avg_sq: torch.Tensor,
    *,
    eps: float,
    can_use_power_iteration: bool,
) -> tuple[list[torch.Tensor | None], torch.Tensor]:
    new_basis: list[torch.Tensor | None] = []
    reordered_exp_avg_sq = exp_avg_sq.float()
    for dim, preconditioner in enumerate(preconditioners):
        if preconditioner is None:
            new_basis.append(None)
            continue

        basis = old_basis[dim] if dim < len(old_basis) else None
        if can_use_power_iteration and basis is not None:
            refreshed, order = _orthogonal_basis_from_power_iteration(preconditioner, basis)
            reordered_exp_avg_sq = reordered_exp_avg_sq.index_select(dim, order)
            new_basis.append(refreshed)
        else:
            new_basis.append(_orthogonal_basis_from_eigh(preconditioner, eps))

    return new_basis, reordered_exp_avg_sq.to(dtype=exp_avg_sq.dtype)


def _init_preconditioners_for_param(
    grad: torch.Tensor,
    *,
    max_preconditioner_dim: int,
) -> list[torch.Tensor | None]:
    preconditioners: list[torch.Tensor | None] = []
    for dim_size in grad.shape:
        if int(dim_size) <= int(max_preconditioner_dim):
            preconditioners.append(
                torch.zeros(
                    (int(dim_size), int(dim_size)),
                    dtype=torch.float32,
                    device=grad.device,
                )
            )
        else:
            preconditioners.append(None)
    return preconditioners


def _accumulate_shampoo_statistics(preconditioners: list[torch.Tensor | None], grad: torch.Tensor) -> None:
    g = grad.detach().float()
    for dim, preconditioner in enumerate(preconditioners):
        if preconditioner is None:
            continue
        unfolded = _unfold_along_dim(g, dim)
        preconditioner.add_(unfolded @ unfolded.T)


def _preconditioned_dim_count(preconditioners: list[torch.Tensor | None]) -> int:
    return sum(1 for p in preconditioners if p is not None)


class Shampoo(torch.optim.Optimizer):
    """Single-device Shampoo optimizer with bounded per-dimension preconditioners.

    Each tensor dimension with size <= ``max_preconditioner_dim`` gets a full Shampoo
    matrix. Larger dimensions are skipped rather than approximated, which keeps this
    suitable for interactive experiments with embeddings and wide layers.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        *,
        lr: float = 1e-2,
        momentum: float = 0.0,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        precondition_frequency: int = 10,
        max_preconditioner_dim: int = 1024,
    ) -> None:
        _check_optimizer_hparams(
            lr=float(lr),
            momentum=float(momentum),
            eps=float(eps),
            weight_decay=float(weight_decay),
            precondition_frequency=int(precondition_frequency),
            max_preconditioner_dim=int(max_preconditioner_dim),
        )
        defaults = {
            "lr": float(lr),
            "momentum": float(momentum),
            "eps": float(eps),
            "weight_decay": float(weight_decay),
            "precondition_frequency": int(precondition_frequency),
            "max_preconditioner_dim": int(max_preconditioner_dim),
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Any | None = None) -> Any | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = float(group["lr"])
            momentum = float(group["momentum"])
            eps = float(group["eps"])
            weight_decay = float(group["weight_decay"])
            precondition_frequency = int(group["precondition_frequency"])
            max_preconditioner_dim = int(group["max_preconditioner_dim"])
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                if grad.is_sparse:
                    raise RuntimeError("Shampoo does not support sparse gradients.")
                state = self.state[param]
                if len(state) == 0:
                    state["step"] = 0
                    state["preconditioners"] = _init_preconditioners_for_param(
                        grad,
                        max_preconditioner_dim=max_preconditioner_dim,
                    )
                    state["inv_preconditioners"] = [None for _ in state["preconditioners"]]
                    if momentum != 0.0:
                        state["momentum_buffer"] = torch.zeros_like(param, memory_format=torch.preserve_format)

                state["step"] = int(state["step"]) + 1
                preconditioners = state["preconditioners"]
                _accumulate_shampoo_statistics(preconditioners, grad)
                num_preconditioned = _preconditioned_dim_count(preconditioners)

                if num_preconditioned and (
                    int(state["step"]) == 1 or int(state["step"]) % precondition_frequency == 0
                ):
                    root_power = -1.0 / (2.0 * float(num_preconditioned))
                    state["inv_preconditioners"] = _preconditioner_roots(
                        preconditioners,
                        root_power=root_power,
                        eps=eps,
                    )

                update = grad.detach().float()
                if num_preconditioned:
                    update = _apply_dim_matrices(update, state["inv_preconditioners"])

                if momentum != 0.0:
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(update.to(dtype=buf.dtype), alpha=1.0 - momentum)
                    update = buf.float()

                if weight_decay != 0.0:
                    param.mul_(1.0 - lr * weight_decay)
                param.add_(update.to(dtype=param.dtype), alpha=-lr)

        return loss


class SOAP(torch.optim.Optimizer):
    """Single-device SOAP-style optimizer.

    SOAP runs Adam-style moments in the eigenspaces induced by Shampoo statistics.
    For tractability in this app, basis matrices are bounded by
    ``max_preconditioner_dim``. Basis refreshes migrate the first moment and keep
    the second moment in the refreshed coordinate order.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        *,
        lr: float = 3e-4,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        precondition_frequency: int = 10,
        max_preconditioner_dim: int = 1024,
    ) -> None:
        _check_optimizer_hparams(
            lr=float(lr),
            betas=(float(betas[0]), float(betas[1])),
            eps=float(eps),
            weight_decay=float(weight_decay),
            precondition_frequency=int(precondition_frequency),
            max_preconditioner_dim=int(max_preconditioner_dim),
        )
        defaults = {
            "lr": float(lr),
            "betas": (float(betas[0]), float(betas[1])),
            "eps": float(eps),
            "weight_decay": float(weight_decay),
            "precondition_frequency": int(precondition_frequency),
            "max_preconditioner_dim": int(max_preconditioner_dim),
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Any | None = None) -> Any | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = float(group["lr"])
            beta1, beta2 = group["betas"]
            beta1 = float(beta1)
            beta2 = float(beta2)
            eps = float(group["eps"])
            weight_decay = float(group["weight_decay"])
            precondition_frequency = int(group["precondition_frequency"])
            max_preconditioner_dim = int(group["max_preconditioner_dim"])
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                if grad.is_sparse:
                    raise RuntimeError("SOAP does not support sparse gradients.")
                state = self.state[param]
                if len(state) == 0:
                    state["step"] = 0
                    state["preconditioners"] = _init_preconditioners_for_param(
                        grad,
                        max_preconditioner_dim=max_preconditioner_dim,
                    )
                    state["basis"] = [None for _ in state["preconditioners"]]
                    state["exp_avg"] = torch.zeros_like(param, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(param, memory_format=torch.preserve_format)
                    state["moment_step"] = 0

                state["step"] = int(state["step"]) + 1
                preconditioners = state["preconditioners"]
                num_preconditioned = _preconditioned_dim_count(preconditioners)
                has_basis = any(basis is not None for basis in state["basis"])

                if num_preconditioned and not has_basis:
                    _accumulate_shampoo_statistics(preconditioners, grad)
                    state["basis"], _ = _refresh_soap_basis_and_second_moment(
                        preconditioners,
                        state["basis"],
                        state["exp_avg_sq"],
                        eps=eps,
                        can_use_power_iteration=False,
                    )
                    continue

                grad_projected = grad.detach().float()
                grad_projected = _apply_dim_matrices(grad_projected, state["basis"], transpose=True)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["moment_step"] = int(state["moment_step"]) + 1
                exp_avg.mul_(beta1).add_(grad_projected.to(dtype=exp_avg.dtype), alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(
                    grad_projected.to(dtype=exp_avg_sq.dtype),
                    grad_projected.to(dtype=exp_avg_sq.dtype),
                    value=1.0 - beta2,
                )

                bias_correction1 = 1.0 - beta1 ** int(state["moment_step"])
                bias_correction2 = 1.0 - beta2 ** int(state["moment_step"])
                update = exp_avg.float() / max(bias_correction1, 1e-16)
                denom = exp_avg_sq.float().div(max(bias_correction2, 1e-16)).sqrt().add_(eps)
                update = update / denom

                update = _apply_dim_matrices(update, state["basis"])

                if weight_decay != 0.0:
                    param.mul_(1.0 - lr * weight_decay)
                param.add_(update.to(dtype=param.dtype), alpha=-lr)

                _accumulate_shampoo_statistics(preconditioners, grad)
                if num_preconditioned and int(state["step"]) % precondition_frequency == 0:
                    old_basis = state["basis"]
                    new_basis, reordered_exp_avg_sq = _refresh_soap_basis_and_second_moment(
                        preconditioners,
                        old_basis,
                        state["exp_avg_sq"],
                        eps=eps,
                        can_use_power_iteration=True,
                    )
                    migrated = _projected_first_moment_in_new_basis(
                        state["exp_avg"],
                        old_basis,
                        new_basis,
                    )
                    state["exp_avg"].copy_(migrated.to(dtype=state["exp_avg"].dtype))
                    state["exp_avg_sq"].copy_(reordered_exp_avg_sq)
                    state["basis"] = new_basis

        return loss
