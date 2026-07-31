"""PyTorch dataset for LPD curves."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from comfy_research.lmech.lpd.schema import CurveSample, segment_type_id
from comfy_research.lmech.lpd.synthetic import interval_to_box


def curve_feature_channels(y_norm: torch.Tensor) -> torch.Tensor:
    """Build (3, L) input: normalized y, dy, d²y (each scaled to unit range)."""
    y = y_norm
    dy = torch.zeros_like(y)
    d2y = torch.zeros_like(y)
    if y.numel() >= 2:
        dy[1:] = y[1:] - y[:-1]
        dy[0] = dy[1]
    if y.numel() >= 3:
        d2y[1:] = dy[1:] - dy[:-1]
        d2y[0] = d2y[1]

    def _scale(ch: torch.Tensor) -> torch.Tensor:
        span = ch.abs().max()
        if float(span) < 1e-6:
            return torch.zeros_like(ch)
        return ch / span

    return torch.stack([y, _scale(dy), _scale(d2y)], dim=0)


def segments_to_breakpoints(
    segments: List[Any],
    t_min: float,
    t_max: float,
) -> torch.Tensor:
    span = max(t_max - t_min, 1e-6)
    bps = [(segments[i].end - t_min) / span for i in range(len(segments) - 1)]
    return torch.tensor(bps, dtype=torch.float32)


class LPDDataset(Dataset):
    def __init__(
        self,
        data_dir: str | Path,
        seq_len: int = 256,
        normalize_y: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.seq_len = seq_len
        self.normalize_y = normalize_y

        manifest_path = self.data_dir / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest = json.load(f)
            self.sample_ids = list(manifest["samples"])
        else:
            self.sample_ids = sorted(
                p.stem for p in self.data_dir.glob("sample_*.json")
            )

        if not self.sample_ids:
            raise ValueError(f"No samples found in {self.data_dir}")

    def __len__(self) -> int:
        return len(self.sample_ids)

    def _load_sample(self, idx: int) -> CurveSample:
        path = self.data_dir / f"{self.sample_ids[idx]}.json"
        with open(path) as f:
            return CurveSample.from_dict(json.load(f))

    def _resample(self, t: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        t_new = torch.linspace(t[0], t[-1], self.seq_len)
        y_new = torch.zeros(self.seq_len)
        for i in range(self.seq_len):
            ti = t_new[i]
            j = torch.searchsorted(t, ti).clamp(1, len(t) - 1)
            t0, t1 = t[j - 1], t[j]
            y0, y1 = y[j - 1], y[j]
            frac = (ti - t0) / (t1 - t0) if t1 > t0 else 0.0
            y_new[i] = y0 + frac * (y1 - y0)
        return t_new, y_new

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self._load_sample(idx)
        t = torch.tensor(sample.t, dtype=torch.float32)
        y = torch.tensor(sample.y, dtype=torch.float32)

        t_min = float(t[0].item())
        t_max = float(t[-1].item())

        t, y = self._resample(t, y)
        if self.normalize_y:
            y_min, y_max = y.min(), y.max()
            span = y_max - y_min
            if span < 1e-6 or not torch.isfinite(span):
                y_norm = torch.zeros_like(y)
            else:
                y_norm = (y - y_min) / span
        else:
            y_norm = y

        labels: List[int] = []
        boxes: List[Tuple[float, float]] = []
        for seg in sample.segments:
            labels.append(segment_type_id(seg.type))
            boxes.append(interval_to_box(seg.start, seg.end, t_min, t_max))

        breakpoints = segments_to_breakpoints(sample.segments, t_min, t_max)

        return {
            "curve": curve_feature_channels(y_norm),
            "breakpoints": breakpoints,
            "labels": torch.tensor(labels, dtype=torch.long),
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "sample_id": sample.id,
            "t_min": t_min,
            "t_max": t_max,
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    curves = torch.stack([b["curve"] for b in batch])
    breakpoints = [b["breakpoints"] for b in batch]
    labels = [b["labels"] for b in batch]
    boxes = [b["boxes"] for b in batch]
    return {
        "curves": curves,
        "breakpoints": breakpoints,
        "labels": labels,
        "boxes": boxes,
    }


def prepare_targets(
    breakpoints_list: List[torch.Tensor],
) -> List[Dict[str, torch.Tensor]]:
    return [{"breakpoints": b} for b in breakpoints_list]
