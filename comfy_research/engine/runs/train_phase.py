"""Optional train-phase reporter for NDJSON status lines (remote prepare / dataset I/O)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

_emitter: Callable[[dict[str, Any]], None] | None = None

CIFAR10_DATASET_NAME = "CIFAR-10"
MNIST_DATASET_NAME = "MNIST"

REMOTE_PHASE_PREFIX = "remote: "


def remote_phase_message(text: str) -> str:
    return f"{REMOTE_PHASE_PREFIX}{text}"


def bind_train_phase_emitter(fn: Callable[[dict[str, Any]], None] | None) -> None:
    global _emitter
    _emitter = fn


def emit_train_phase(phase: str, message: str, *, meta: dict[str, Any] | None = None) -> None:
    if _emitter is None:
        return
    payload: dict[str, Any] = {"phase": phase, "message": message}
    if meta:
        payload["meta"] = meta
    _emitter(payload)


def emit_dataset_download(dataset_name: str, *, meta: dict[str, Any] | None = None) -> None:
    emit_train_phase("dataset_download", remote_phase_message(f"downloading {dataset_name}"), meta=meta)


def emit_dataset_extract(dataset_name: str, *, meta: dict[str, Any] | None = None) -> None:
    emit_train_phase("dataset_extract", remote_phase_message(f"extracting {dataset_name}"), meta=meta)


def emit_dataset_load(dataset_name: str, *, meta: dict[str, Any] | None = None) -> None:
    emit_train_phase("dataset_load", remote_phase_message(f"loading {dataset_name}"), meta=meta)
