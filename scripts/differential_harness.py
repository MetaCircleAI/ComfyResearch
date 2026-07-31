"""End-to-end differential harness for behavior-preserving trainer refactors.

Runs curated graph fixtures through ``iter_trainer_events`` for a few real
CPU training steps and writes a STABLE, sorted JSON trace. Run it on two
branches (see ``scripts/run_differential.sh``) and diff the traces: identical
files == identical behavior, float-exact, including error paths.

Usage:
    PYTHONPATH=. python scripts/differential_harness.py [--suite SUITE] OUTPUT.json

Suites (--suite, default ``smoke``):
    smoke       2 train-smoke fixtures + 2 custom shapes (fast sanity)
    observable  smoke fixtures + the 2 observable sweeps (21 wired kinds) —
                the gate for recorder/dispatch changes
    full        all fixtures: 30 materialize goldens + smoke + 5 custom
                complex shapes + 2 observable sweeps — the gate for
                high-risk refactors

JSON contract (the file is the interface; stdout/stderr are informational):
    {"meta": {"git_head": str, "tracked_dirty": bool,
              "untracked_present": bool, "suite": str},
     "results": {fixture_name: trace, ...},
     "failures": {fixture_name: "ExcType: message", ...}}

A fixture that raises is characterized in ``failures`` — an identical
exception on both branches is identical behavior. Determinism: CPU device,
fixed seeds, fixed 5 steps; fixtures are only mutated to force
trainingSteps/logFrequency.
"""
from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

# The engine under measurement is resolved via PYTHONPATH/cwd, NOT this
# file's location: run_differential.sh executes THIS harness (one measuring
# instrument) against two different trees by pointing PYTHONPATH at each.
# 跨版本 import:harness 以 HEAD 的 scripts 同时驱动 HEAD 与 base
# worktree;base 可能仍采用旧布局,双路径 fallback 保证两侧都能跑。
try:
    from comfy_research.engine.runs.trainer_run import iter_trainer_events  # noqa: E402
except ModuleNotFoundError as e:  # legacy layout in a base worktree
    if e.name not in ("comfy_research.engine.runs", "comfy_research.engine.runs.trainer_run"):
        raise  # 新模块内部缺依赖不得静默回退旧路径
    from comfy_research.engine.trainer_run import iter_trainer_events  # noqa: E402
from comfy_research.schemas.graph import Edge, Node  # noqa: E402

from comfy_research.tests.test_trainer_materialize_golden import (  # noqa: E402
    _all_fixtures,
    _build_graph_objects,
)
from comfy_research.tests.test_train_smoke_golden import _fixtures as _smoke_fixtures  # noqa: E402

STEPS = 5
SUITE_NAMES = ("smoke", "observable", "full")


# ---------------------------------------------------------------------------
# Custom fixtures: complex shapes NOT covered by the materialize golden.
# ---------------------------------------------------------------------------
def _custom_fixtures() -> dict[str, dict]:
    base_nodes = [
        {"id": "dataset", "type": "linear_dataset",
         "data": {"inputDim": 3, "outputDim": 2, "trainSize": 16, "testSize": 8,
                  "noiseLevel": 0, "seed": 7, "samplingMode": "fixed"}},
        {"id": "model", "type": "mlp_model",
         "data": {"inputDim": 3, "outputDim": 2, "depth": 2, "width": 8,
                  "activation": "relu", "seed": 7}},
        {"id": "optimizer", "type": "adam_optimizer", "data": {"learningRate": 0.01}},
        {"id": "loss", "type": "mse_loss", "data": {}},
        {"id": "trainer", "type": "trainer",
         "data": {"trainingSteps": STEPS, "logFrequency": 1, "batchSize": -1,
                  "computeDevice": "cpu"}},
    ]
    std_edges = [
        ["e1", "dataset", "trainer", "dataset", "dataset"],
        ["e2", "model", "trainer", "model", "model"],
        ["e3", "optimizer", "trainer", "optimizer", "optimizer"],
        ["e4", "loss", "trainer", "loss", "loss"],
    ]

    out: dict[str, dict] = {}

    # 1. legacy separate train/test dataset wires
    fx = {"trainer_node_id": "trainer", "nodes": copy.deepcopy(base_nodes), "edges": [
        ["e1", "dataset", "trainer", "dataset", "train_dataset"],
        ["e1b", "dataset2", "trainer", "dataset", "test_dataset"],
        ["e2", "model", "trainer", "model", "model"],
        ["e3", "optimizer", "trainer", "optimizer", "optimizer"],
        ["e4", "loss", "trainer", "loss", "loss"],
    ]}
    fx["nodes"].append({"id": "dataset2", "type": "linear_dataset",
                        "data": {"inputDim": 3, "outputDim": 2, "trainSize": 16, "testSize": 8,
                                 "noiseLevel": 0, "seed": 7, "samplingMode": "fixed"}})
    out["legacy_train_test_wires"] = fx

    # 2. cosine schedule + warmup + grad clip
    fx = {"trainer_node_id": "trainer", "nodes": copy.deepcopy(base_nodes),
          "edges": copy.deepcopy(std_edges)}
    for n in fx["nodes"]:
        if n["id"] == "trainer":
            n["data"].update({"lrSchedule": "cosine", "lrWarmupSteps": 2,
                              "cosineLrMinFraction": 0.1, "gradClipMaxNorm": 1.0})
    out["cosine_warmup_clip"] = fx

    # 3. muP init + muP LR schedule
    fx = {"trainer_node_id": "trainer", "nodes": copy.deepcopy(base_nodes),
          "edges": copy.deepcopy(std_edges)}
    for n in fx["nodes"]:
        if n["id"] == "trainer":
            n["data"].update({"useMupInit": True, "useMupLrSchedule": True})
    out["mup_init_and_lr"] = fx

    # 4. atomic layer tip wired to trainer
    fx = {"trainer_node_id": "trainer", "nodes": [
        {"id": "dataset", "type": "linear_dataset",
         "data": {"inputDim": 3, "outputDim": 2, "trainSize": 16, "testSize": 8,
                  "noiseLevel": 0, "seed": 7, "samplingMode": "fixed"}},
        {"id": "lin1", "type": "linear_layer", "data": {"inFeatures": 3, "outFeatures": 2, "seed": 7}},
        {"id": "optimizer", "type": "adam_optimizer", "data": {"learningRate": 0.01}},
        {"id": "loss", "type": "mse_loss", "data": {}},
        {"id": "trainer", "type": "trainer",
         "data": {"trainingSteps": STEPS, "logFrequency": 1, "batchSize": -1,
                  "computeDevice": "cpu"}},
    ], "edges": [
        ["e1", "dataset", "trainer", "dataset", "dataset"],
        ["e2", "lin1", "trainer", "model", "model"],
        ["e3", "optimizer", "trainer", "optimizer", "optimizer"],
        ["e4", "loss", "trainer", "loss", "loss"],
    ]}
    out["atomic_linear_tip"] = fx

    # 5. observable sweep, vector side
    fx = {"trainer_node_id": "trainer", "nodes": copy.deepcopy(base_nodes),
          "edges": copy.deepcopy(std_edges)}
    obs_v = [
        ("o_w2", "observable_weight_l2", {"normAggregation": "top_level_module"}),
        ("o_w2t", "observable_weight_l2", {"normAggregation": "tensor"}),
        ("o_w1", "observable_weight_l1", {}),
        ("o_gn", "observable_gradient_norm", {"normAggregation": "top_level_module"}),
        ("o_gap", "observable_train_test_gap", {}),
        ("o_cap", "observable_capacity", {}),
        ("o_relu", "observable_relu_nonlinear_count", {"hiddenLayerIndex": 0}),
        ("o_hess", "observable_hessian_eigenvalues", {}),
    ]
    for oid, otype, od in obs_v:
        fx["nodes"].append({"id": oid, "type": otype, "data": od})
        fx["edges"].append([f"e-{oid}", oid, "trainer", "observables", "observables"])
    out["obs_vector_sweep"] = fx

    # 6. observable sweep, token/attention side
    fx = {"trainer_node_id": "trainer", "nodes": [
        {"id": "dataset", "type": "token_prediction_dataset",
         "data": {"vocabSize": 5, "contextLength": 3, "whichToken": -1, "trainSize": 8,
                  "testSize": 4, "seed": 0, "samplingMode": "fixed"}},
        {"id": "model", "type": "transformer_token_model",
         "data": {"vocabSize": 5, "contextLength": 3, "modelDim": 8, "numLayers": 2,
                  "numHeads": 2, "seed": 0}},
        {"id": "optimizer", "type": "adam_optimizer", "data": {"learningRate": 0.01}},
        {"id": "loss", "type": "cross_entropy_loss", "data": {}},
        {"id": "trainer", "type": "trainer",
         "data": {"trainingSteps": STEPS, "logFrequency": 1, "batchSize": -1,
                  "computeDevice": "cpu"}},
    ], "edges": copy.deepcopy(std_edges)}
    obs_t = [
        ("o_acc", "observable_accuracy", {}),
        ("o_sink", "observable_sink_attention_mass", {"sinkAttentionMassLayers": "all_layers"}),
        ("o_ent", "observable_attention_entropy_mean", {"observableEncoderLayers": "all_layers"}),
        ("o_maxw", "observable_attention_max_weight_mean", {}),
        ("o_hsink", "observable_attention_head_sink_max", {}),
        ("o_bias", "observable_attention_position_bias_ratio", {}),
        ("o_astats", "observable_activation_stats", {"activationStatsLayers": "all_layers"}),
        ("o_anorm", "observable_activation_norm_mean", {}),
        ("o_aout", "observable_activation_outlier_ratio", {}),
        ("o_erank", "observable_embedding_effective_rank", {}),
        ("o_edrift", "observable_embedding_feature_drift", {}),
        ("o_eevo", "observable_embedding_evolution", {}),
        ("o_etraj", "observable_embedding_trajectory", {}),
    ]
    for oid, otype, od in obs_t:
        fx["nodes"].append({"id": oid, "type": otype, "data": od})
        fx["edges"].append([f"e-{oid}", oid, "trainer", "observables", "observables"])
    out["obs_token_sweep"] = fx

    # 7. memB routed to the token task: characterized fall-through (its memB
    # draws are consumed, then the fallback circle-walk overwrites x_t/y_t).
    fx = {"trainer_node_id": "trainer", "nodes": [
        {"id": "dataset", "type": "memorization_b_dataset",
         "data": {"vocabSize": 6, "trainSize": 8, "testSize": 4, "seed": 0, "samplingMode": "fixed"}},
        {"id": "model", "type": "mlp_token_model",
         "data": {"vocabSize": 6, "embedDim": 4, "tokensPerInput": 1, "depth": 1, "width": 6,
                  "activation": "relu", "tieWeights": "no", "seed": 0}},
        {"id": "optimizer", "type": "adam_optimizer", "data": {"learningRate": 0.01}},
        {"id": "loss", "type": "cross_entropy_loss", "data": {}},
        {"id": "trainer", "type": "trainer",
         "data": {"trainingSteps": STEPS, "logFrequency": 1, "batchSize": -1,
                  "computeDevice": "cpu"}},
    ], "edges": copy.deepcopy(std_edges)}
    out["memb_token_fallthrough"] = fx
    return out


# ---------------------------------------------------------------------------
# Execution + trace capture
# ---------------------------------------------------------------------------
def _as_objects(fx: dict) -> tuple[list[Node], list[Edge], str]:
    nodes = [Node(id=n["id"], type=n["type"], position={"x": 0, "y": 0}, data=n.get("data") or {})
             for n in fx["nodes"]]
    edges = [Edge(id=e[0], source=e[1], target=e[2], sourceHandle=e[3], targetHandle=e[4])
             for e in fx["edges"]]
    return nodes, edges, fx["trainer_node_id"]


def _shrink_steps(nodes: list[Node]) -> None:
    for n in nodes:
        if str(n.type) == "trainer":
            n.data = dict(n.data or {}, trainingSteps=STEPS, logFrequency=1)


def run_one(nodes: list[Node], edges: list[Edge], tid: str) -> dict[str, Any]:
    _shrink_steps(nodes)
    events = list(iter_trainer_events(nodes, edges, tid))
    complete = [e for e in events if e.get("type") == "complete"]
    trace: dict[str, Any] = {"event_types": [e.get("type") for e in events]}
    if complete:
        c = complete[0]
        trace["loss_history"] = c.get("loss_history")
        trace["step_ticks"] = c.get("step_ticks")
        trace["complete_keys"] = sorted(c.keys())
        omh = c.get("observable_metric_histories")
        if isinstance(omh, dict):
            trace["observable_metric_histories"] = {k: omh[k] for k in sorted(omh)}
    return trace


def _thunks() -> dict[str, Callable[[], dict[str, Any]]]:
    """name -> zero-arg runner; nothing executes until the suite selects it."""
    thunks: dict[str, Callable[[], dict[str, Any]]] = {}
    for name, (fx, _streaming) in _all_fixtures().items():
        def _run_golden(fx=fx):
            nodes, edges = _build_graph_objects(fx)
            return run_one(nodes, edges, fx.get("trainer_node_id", "trainer"))
        thunks[f"golden::{name}"] = _run_golden
    for name, fx in _smoke_fixtures().items():
        thunks[f"smoke::{name}"] = lambda fx=fx: run_one(*_as_objects(fx))
    for name, fx in _custom_fixtures().items():
        thunks[f"custom::{name}"] = lambda fx=fx: run_one(*_as_objects(fx))
    return thunks


def _suite_members(suite: str, all_names: list[str]) -> list[str]:
    smoke = [n for n in all_names if n.startswith("smoke::")] + [
        "custom::legacy_train_test_wires", "custom::atomic_linear_tip"]
    if suite == "smoke":
        return [n for n in smoke if n in all_names]
    if suite == "observable":
        return [n for n in all_names
                if n.startswith("smoke::") or n.startswith("custom::obs_")]
    return all_names  # full


def _git_meta(suite: str) -> dict[str, Any]:
    # cwd (not this file's repo) so a worktree run reports the worktree's HEAD
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    porcelain = subprocess.run(["git", "status", "--porcelain"],
                               capture_output=True, text=True).stdout.splitlines()
    return {
        "git_head": head,
        "tracked_dirty": any(not line.startswith("??") for line in porcelain),
        "untracked_present": any(line.startswith("??") for line in porcelain),
        "suite": suite,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", choices=SUITE_NAMES, default="smoke")
    ap.add_argument("output", help="path for the JSON trace")
    args = ap.parse_args()

    thunks = _thunks()
    wanted = _suite_members(args.suite, list(thunks))
    results: dict[str, Any] = {}
    failures: dict[str, str] = {}
    for name in wanted:
        try:
            results[name] = thunks[name]()
        except Exception as e:  # noqa: BLE001 — characterized, not swallowed
            failures[name] = f"{type(e).__name__}: {e}"

    payload = {"meta": _git_meta(args.suite), "results": results, "failures": failures}
    Path(args.output).write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n",
                                 encoding="utf-8")
    print(f"[harness] suite={args.suite} traces={len(results)} failures={len(failures)} "
          f"head={payload['meta']['git_head']} tracked_dirty={payload['meta']['tracked_dirty']}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
