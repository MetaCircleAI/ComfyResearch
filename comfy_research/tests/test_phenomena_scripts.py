"""scripts/phenomena 防腐:--quick --json 必须能跑通且输出 schema 字段(不断言现象本身)。"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")

REPO_ROOT = Path(__file__).resolve().parents[2]

SCRIPT_SCHEMAS = {
    "scripts/phenomena/staggered_sv.py": {
        "experiment", "quick", "top_k", "steps",
        "half_max_steps", "ordered", "logit_fit_r2", "final_svs",
    },
    "scripts/phenomena/edge_of_stability.py": {
        "experiment", "quick", "steps", "lr", "two_over_lr", "crossing_step",
        "pre_cross_monotone_frac", "post_cross_up_frac", "post_cross_sharpness_mean",
        "eos_reached", "oscillates", "final_sharpness",
    },
    "scripts/phenomena/rich_vs_lazy.py": {
        "experiment", "quick", "steps", "alphas", "displacement", "alignment",
        "rich_more_displaced", "rich_better_aligned",
    },
}


@pytest.mark.parametrize("script,required", sorted(SCRIPT_SCHEMAS.items()))
def test_quick_json_schema(script: str, required: set[str]) -> None:
    out = subprocess.run(
        [sys.executable, script, "--quick", "--json"],
        capture_output=True, text=True, timeout=300, check=True, cwd=REPO_ROOT,
    )
    payload = json.loads(out.stdout.strip().splitlines()[-1])
    assert required <= set(payload), payload
    assert payload["quick"] is True
