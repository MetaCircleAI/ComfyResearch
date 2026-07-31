"""Tests for train phase NDJSON events."""

from __future__ import annotations

import json

from comfy_research.engine.runs.train_phase import bind_train_phase_emitter, emit_dataset_download


def test_emit_dataset_download_includes_meta() -> None:
    seen: list[dict] = []

    bind_train_phase_emitter(seen.append)
    try:
        emit_dataset_download("CIFAR-10", meta={"downloaded": True, "downloadMs": 42})
    finally:
        bind_train_phase_emitter(None)

    assert seen[0]["message"] == "remote: downloading CIFAR-10"
    assert seen[0]["meta"] == {"downloaded": True, "downloadMs": 42}
    line = json.dumps({"type": "phase", **seen[0]}, separators=(",", ":"))
    assert '"meta"' in line
