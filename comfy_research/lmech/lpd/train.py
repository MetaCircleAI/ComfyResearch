"""Training loop for CurveDETR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader

from comfy_research.lmech.lpd.dataset import LPDDataset, collate_fn, prepare_targets
from comfy_research.lmech.lpd.loss import SetCriterion
from comfy_research.lmech.lpd.matcher import BreakpointMatcher
from comfy_research.lmech.lpd.model import CurveDETR


@dataclass
class TrainLPDConfig:
    data_dir: str = "data/lpd/fake"
    seq_len: int = 256
    num_queries: int = 8
    d_model: int = 128
    batch_size: int = 4
    train_steps: int = 3000
    lr: float = 1e-3
    device: str = "cpu"
    seed: int = 42
    log_interval: int = 50


def train_lpd(
    config: Optional[TrainLPDConfig] = None,
    progress: bool = False,
) -> tuple[CurveDETR, Dict[str, List[float]]]:
    if config is None:
        config = TrainLPDConfig()

    torch.manual_seed(config.seed)
    device = torch.device(config.device)

    dataset = LPDDataset(config.data_dir, seq_len=config.seq_len)
    loader = DataLoader(
        dataset,
        batch_size=min(config.batch_size, len(dataset)),
        shuffle=True,
        collate_fn=collate_fn,
    )

    model = CurveDETR(num_queries=config.num_queries, d_model=config.d_model).to(device)
    matcher = BreakpointMatcher()
    criterion = SetCriterion(matcher).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    history: Dict[str, List[float]] = {"loss": [], "loss_bp": [], "loss_score": []}

    step_bar = None
    if progress:
        from tqdm import tqdm

        step_bar = tqdm(total=config.train_steps, desc="Train LPD", unit="step")

    def forward_loss(batch):
        curves = batch["curves"].to(device)
        targets = prepare_targets(batch["breakpoints"])
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        outputs = model(curves)
        return criterion(outputs, targets)

    def log_step(step: int, step_loss: Dict[str, float]) -> None:
        if progress and (
            step == 0
            or step % config.log_interval == 0
            or step == config.train_steps
        ):
            print(
                f"step {step}/{config.train_steps} "
                f"loss={step_loss['loss']:.4f} "
                f"loss_bp={step_loss['loss_bp']:.4f} "
                f"loss_score={step_loss['loss_score']:.4f}",
                flush=True,
            )

    model.train()
    data_iter = iter(loader)
    try:
        batch = next(data_iter)
    except StopIteration:
        data_iter = iter(loader)
        batch = next(data_iter)

    with torch.no_grad():
        losses = forward_loss(batch)
        step_loss = {k: float(losses[k].item()) for k in history}
        for k in history:
            history[k].append(step_loss[k])
        log_step(0, step_loss)

    for step in range(1, config.train_steps + 1):
        if step > 1:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)

        losses = forward_loss(batch)

        optimizer.zero_grad()
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        step_loss = {k: float(losses[k].item()) for k in history}
        for k in history:
            history[k].append(step_loss[k])

        if step_bar is not None:
            step_bar.update(1)
            step_bar.set_postfix(
                loss=f"{step_loss['loss']:.4f}",
                bp=f"{step_loss['loss_bp']:.4f}",
            )

        log_step(step, step_loss)

    if step_bar is not None:
        step_bar.close()

    return model, history
