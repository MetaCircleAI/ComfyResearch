"""Training loop for LMN."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from comfy_research.lmech.lmn.json_util import finite_float
from comfy_research.lmech.lmn.model import LMN
from comfy_research.lmech.lmn.primitives import NUM_PRIMITIVES, PRIMITIVE_SPECS, PRIMITIVE_NAMES

ProgressCallback = Callable[["TrainProgress"], None]


@dataclass
class TrainProgress:
    """Training progress snapshot for CLI tqdm or web SSE."""

    current: int
    total: int
    phase: str = "train"  # train | auto_k | train_done | finetize
    restart: int = 0
    num_restarts: int = 1
    epoch: int = 0
    num_epochs: int = 0
    k: int = 1
    k_max: int = 1
    mse: Optional[float] = None
    message: str = ""

    @property
    def fraction(self) -> float:
        if self.total <= 0:
            return 0.0
        return min(1.0, self.current / self.total)

    def to_dict(self) -> Dict[str, float | int | str | None]:
        d = asdict(self)
        d["fraction"] = self.fraction
        if d.get("mse") is not None:
            d["mse"] = finite_float(d["mse"])
        return d


@dataclass
class TrainConfig:
    num_phases: int = 4
    epochs: int = 3000
    lr: float = 1e-2
    beta_start: float = 1.0
    beta_end: float = 200.0
    lambda_pi: float = 1e-3
    lambda_w: float = 1e-4
    restarts: int = 3
    seed: int = 42
    verbose: bool = False
    show_progress: bool = False
    early_stop_patience: int = 0  # 0 = disabled
    early_stop_min_delta: float = 1e-6

    @classmethod
    def fast(cls, num_phases: int = 4, **kwargs) -> "TrainConfig":
        """Lightweight config for unit / CI tests."""
        defaults = dict(
            num_phases=num_phases,
            epochs=300,
            restarts=1,
            beta_end=100.0,
            early_stop_patience=50,
            early_stop_min_delta=1e-5,
        )
        defaults.update(kwargs)
        return cls(**defaults)


@dataclass
class TrainResult:
    model: LMN
    history: List[Dict[str, float]] = field(default_factory=list)
    best_mse: float = float("inf")
    bic: Optional[float] = None


def _beta_schedule(epoch: int, total: int, beta_start: float, beta_end: float) -> float:
    if total <= 1:
        return beta_end
    frac = epoch / (total - 1)
    return beta_start + (beta_end - beta_start) * frac


def _make_tqdm_callback(
    total: int, desc: str = "Training LMN"
) -> Tuple[ProgressCallback, Callable[[], None]]:
    from tqdm import tqdm

    bar = tqdm(total=total, desc=desc, unit="step", dynamic_ncols=True)

    def callback(p: TrainProgress) -> None:
        bar.n = min(p.current, total)
        postfix: Dict[str, str] = {}
        if p.mse is not None:
            postfix["mse"] = f"{p.mse:.4f}"
        if p.k_max > 1:
            postfix["K"] = f"{p.k}/{p.k_max}"
        if p.num_restarts > 1:
            postfix["restart"] = f"{p.restart + 1}/{p.num_restarts}"
        if p.message:
            postfix["stage"] = p.message
        if postfix:
            bar.set_postfix(postfix, refresh=False)
        bar.refresh()

    def close() -> None:
        bar.n = total
        bar.refresh()
        bar.close()

    return callback, close


def count_parameters(num_phases: int) -> int:
    """Effective parameter count for BIC."""
    if num_phases <= 1:
        return NUM_PRIMITIVES  # one phase, one primitive typically
    p = (num_phases - 1)  # breakpoints
    # per phase: pick 1 primitive with avg param count ~3
    avg_params = sum(PRIMITIVE_SPECS[n].num_params for n in PRIMITIVE_NAMES) / len(
        PRIMITIVE_NAMES
    )
    p += num_phases * avg_params
    return int(p)


def compute_bic(mse: float, n: int, num_phases: int) -> float:
    sse = mse * n
    p = count_parameters(num_phases)
    return n * math.log(max(sse / n, 1e-12)) + p * math.log(n)


def train_lmn(
    t: torch.Tensor,
    y: torch.Tensor,
    config: TrainConfig,
    device: Optional[torch.device] = None,
    progress_callback: Optional[ProgressCallback] = None,
    progress_offset: int = 0,
    progress_total: Optional[int] = None,
) -> TrainResult:
    if device is None:
        device = torch.device("cpu")

    t = t.to(device)
    y = y.to(device)
    t_max = float(t.max().item())

    best_result: Optional[TrainResult] = None
    best_mse = float("inf")

    local_total = config.restarts * config.epochs
    total = progress_total if progress_total is not None else local_total

    cb = progress_callback
    close_fn: Callable[[], None] = lambda: None
    if cb is None and config.show_progress:
        cb, close_fn = _make_tqdm_callback(total=total)

    try:
        for restart in range(config.restarts):
            seed = config.seed + restart
            torch.manual_seed(seed)

            model = LMN(
                num_phases=config.num_phases,
                t_max=t_max,
                lambda_pi=config.lambda_pi,
                lambda_w=config.lambda_w,
            ).to(device)
            model.initialize_from_data(t, y)

            optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
            history: List[Dict[str, float]] = []
            best_epoch_mse = float("inf")
            stale_epochs = 0
            best_state = None

            for epoch in range(config.epochs):
                beta = _beta_schedule(
                    epoch, config.epochs, config.beta_start, config.beta_end
                )
                optimizer.zero_grad()
                loss, metrics = model.loss(t, y, beta=beta)
                loss.backward()
                optimizer.step()
                metrics["beta"] = beta
                metrics["epoch"] = epoch
                history.append(metrics)

                if cb is not None:
                    step = restart * config.epochs + epoch + 1
                    cb(
                        TrainProgress(
                            current=progress_offset + step,
                            total=total,
                            phase="train",
                            restart=restart,
                            num_restarts=config.restarts,
                            epoch=epoch + 1,
                            num_epochs=config.epochs,
                            k=config.num_phases,
                            mse=metrics["mse"],
                            message=f"K={config.num_phases}",
                        )
                    )

                if config.verbose and epoch % 500 == 0:
                    print(
                        f"restart={restart} epoch={epoch} mse={metrics['mse']:.6f} beta={beta:.1f}"
                    )

                if metrics["mse"] < best_epoch_mse - config.early_stop_min_delta:
                    best_epoch_mse = metrics["mse"]
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                    stale_epochs = 0
                elif config.early_stop_patience > 0:
                    stale_epochs += 1
                    if stale_epochs >= config.early_stop_patience:
                        break

            if best_state is not None:
                model.load_state_dict(best_state)
            final_mse = best_epoch_mse if best_state is not None else history[-1]["mse"]
            if final_mse < best_mse:
                best_mse = final_mse
                result = TrainResult(model=model, history=history, best_mse=final_mse)
                result.bic = compute_bic(final_mse, len(t), config.num_phases)
                best_result = result
    finally:
        close_fn()

    assert best_result is not None
    return best_result


def train_auto_k(
    t: torch.Tensor,
    y: torch.Tensor,
    k_max: int = 6,
    base_config: Optional[TrainConfig] = None,
    device: Optional[torch.device] = None,
    progress_callback: Optional[ProgressCallback] = None,
) -> TrainResult:
    """Train models for K=1..k_max and pick best by BIC."""
    if base_config is None:
        base_config = TrainConfig()

    best: Optional[TrainResult] = None
    best_bic = float("inf")

    total = k_max * base_config.restarts * base_config.epochs
    cb = progress_callback
    close_fn: Callable[[], None] = lambda: None
    if cb is None and base_config.show_progress:
        cb, close_fn = _make_tqdm_callback(total=total, desc="Auto-K training")

    try:
        for k in range(1, k_max + 1):
            cfg = TrainConfig(
                num_phases=k,
                epochs=base_config.epochs,
                lr=base_config.lr,
                beta_start=base_config.beta_start,
                beta_end=base_config.beta_end,
                lambda_pi=base_config.lambda_pi,
                lambda_w=base_config.lambda_w,
                restarts=base_config.restarts,
                seed=base_config.seed,
                verbose=base_config.verbose,
                show_progress=False,
                early_stop_patience=base_config.early_stop_patience,
                early_stop_min_delta=base_config.early_stop_min_delta,
            )
            offset = (k - 1) * base_config.restarts * base_config.epochs

            def _inner_progress(p: TrainProgress, _k: int = k) -> None:
                if cb is None:
                    return
                cb(
                    TrainProgress(
                        current=p.current,
                        total=total,
                        phase="auto_k",
                        restart=p.restart,
                        num_restarts=p.num_restarts,
                        epoch=p.epoch,
                        num_epochs=p.num_epochs,
                        k=_k,
                        k_max=k_max,
                        mse=p.mse,
                        message=f"K={_k}/{k_max}",
                    )
                )

            result = train_lmn(
                t,
                y,
                cfg,
                device=device,
                progress_callback=_inner_progress if cb else None,
                progress_offset=offset,
                progress_total=total,
            )
            if result.bic is not None and result.bic < best_bic:
                best_bic = result.bic
                best = result
    finally:
        close_fn()

    assert best is not None
    return best
