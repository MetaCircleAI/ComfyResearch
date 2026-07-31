"""Schema-only smoke test for the differential harness.

Runs the harness's `smoke` suite in-process and validates the JSON trace
contract. Deliberately does NOT run run_differential.sh or create worktrees —
the two-branch diff is a manual gate, not a CI job.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPO_ROOT = Path(__file__).resolve().parents[2]
HARNESS = REPO_ROOT / "scripts" / "differential_harness.py"


def test_smoke_suite_emits_schema_valid_trace(tmp_path: Path) -> None:
    out = tmp_path / "trace.json"
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT))
    proc = subprocess.run(
        [sys.executable, str(HARNESS), "--suite", "smoke", str(out)],
        capture_output=True, text=True, cwd=REPO_ROOT, env=env,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    payload = json.loads(out.read_text())

    meta = payload["meta"]
    assert isinstance(meta["git_head"], str) and len(meta["git_head"]) >= 7
    assert isinstance(meta["tracked_dirty"], bool)
    assert isinstance(meta["untracked_present"], bool)
    assert meta["suite"] == "smoke"

    results = payload["results"]
    assert len(results) >= 4  # 2 smoke fixtures + 2 custom shapes
    for name, trace in results.items():
        assert trace["event_types"], name
        if "loss_history" in trace:
            assert isinstance(trace["loss_history"], list)
    assert isinstance(payload["failures"], dict)

    # order stability: serializing the parsed payload with the same settings
    # must reproduce the file byte-for-byte
    assert out.read_text() == json.dumps(payload, indent=1, sort_keys=True) + "\n"
