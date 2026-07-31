"""In-memory store for activation tensors (lazy fetch from browser)."""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch


def _nbytes_dict(tensors: dict[str, np.ndarray]) -> int:
    return int(sum(int(v.nbytes) for v in tensors.values()))


@dataclass
class _ActivationRun:
    tensors: dict[str, np.ndarray]
    created: float = field(default_factory=time.monotonic)


_LOCK = threading.Lock()
_RUNS: dict[str, _ActivationRun] = {}

# Evict oldest runs first; soft caps to avoid unbounded RAM on the server.
_MAX_RUNS = 24
_MAX_BYTES = 3 * 1024**3


def _evict_unlocked(need_bytes: int = 0) -> None:
    """Drop oldest runs until under count/byte limits."""
    while len(_RUNS) > _MAX_RUNS:
        oldest = min(_RUNS.items(), key=lambda kv: kv[1].created)[0]
        del _RUNS[oldest]
    total = sum(_nbytes_dict(r.tensors) for r in _RUNS.values())
    while total + need_bytes > _MAX_BYTES and _RUNS:
        oldest = min(_RUNS.items(), key=lambda kv: kv[1].created)[0]
        total -= _nbytes_dict(_RUNS[oldest].tensors)
        del _RUNS[oldest]


def store_activation_tensors(tensors_torch: dict[str, torch.Tensor]) -> str:
    """Copy torch tensors to CPU float32 numpy, return run id."""
    tensors_np: dict[str, np.ndarray] = {}
    need = 0
    for k, t in tensors_torch.items():
        arr = t.detach().cpu().float().numpy()
        if not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr)
        tensors_np[str(k)] = arr
        need += int(arr.nbytes)

    with _LOCK:
        _evict_unlocked(need_bytes=need)
        run_id = uuid.uuid4().hex
        _RUNS[run_id] = _ActivationRun(tensors=tensors_np)
        _evict_unlocked()
    return run_id


def get_activation_numpy(run_id: str, rep_id: str) -> Optional[np.ndarray]:
    with _LOCK:
        run = _RUNS.get(run_id)
        if run is None:
            return None
        return run.tensors.get(rep_id)


def manifest_for_run(run_id: str) -> Optional[dict[str, dict[str, list[int]]]]:
    with _LOCK:
        run = _RUNS.get(run_id)
        if run is None:
            return None
        return {k: {"shape": list(v.shape)} for k, v in run.tensors.items()}
