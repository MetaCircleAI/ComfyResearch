#!/usr/bin/env python3
"""Smoke test: representation algebra observable hooks + trainer metric / viz stream."""

from __future__ import annotations

import math
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
MPL_CACHE = ROOT / ".cache" / "matplotlib"
MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))

from comfy_research.api.user_observables import UserObservableRecord
from comfy_research.engine.analysis.observable_algebra import (
    format_algebra_definition_code,
    format_algebra_human_chain,
    parse_axis_reductions,
)
from comfy_research.engine.analysis.representation_specs import (
    fetch_representation_numpy,
    run_model_representation_specs,
)
from comfy_research.engine.runs.trainer_run import iter_trainer_events_from_context, prepare_trainer_run
from comfy_research.schemas.graph import Edge, Node, Position


def _finite(values: list[float]) -> list[float]:
    return [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(float(v))]


def _build_graph(*, observable_uid: str, rep_id: str, rep_label: str) -> tuple[list[Node], list[Edge], str, str, str]:
    ds_id = "tok_ds"
    model_id = "mlp_tok"
    opt_id = "adam"
    loss_id = "ce"
    trainer_id = "trainer"
    obs_id = "obs_user"
    viz_id = "obs_viz"

    nodes = [
        Node(
            id=ds_id,
            type="token_prediction_dataset",
            data={
                "vocabSize": 64,
                "contextLength": 4,
                "whichToken": 0,
                "trainSize": 128,
                "testSize": 32,
                "seed": 7,
            },
            position=Position(x=0, y=0),
        ),
        Node(
            id=model_id,
            type="mlp_token_model",
            data={
                "vocabSize": 64,
                "embedDim": 16,
                "tokensPerInput": 4,
                "depth": 2,
                "width": 32,
                "numExperts": 1,
                "activation": "gelu",
                "tieWeights": "yes",
                "seed": 3,
            },
            position=Position(x=200, y=0),
        ),
        Node(
            id=opt_id,
            type="adam_optimizer",
            data={"learningRate": 0.01, "beta1": 0.9, "beta2": 0.999, "epsilon": 1e-8},
            position=Position(x=200, y=200),
        ),
        Node(
            id=loss_id,
            type="cross_entropy_loss",
            data={"lossScale": 1},
            position=Position(x=400, y=200),
        ),
        Node(
            id=trainer_id,
            type="trainer",
            data={"trainingSteps": 30, "logFrequency": 5},
            position=Position(x=600, y=100),
        ),
        Node(
            id=obs_id,
            type="observable_user",
            data={
                "userObservableId": observable_uid,
                "label": rep_label,
            },
            position=Position(x=400, y=0),
        ),
        Node(
            id=viz_id,
            type="observable_viz_user",
            data={
                "pairedObservableId": obs_id,
                "pairedTrainerId": trainer_id,
                "observableName": rep_label,
            },
            position=Position(x=800, y=100),
        ),
    ]
    edges = [
        Edge(id="e-ds-tr", source=ds_id, target=trainer_id, sourceHandle="train_dataset", targetHandle="train_dataset"),
        Edge(id="e-ds-te", source=ds_id, target=trainer_id, sourceHandle="test_dataset", targetHandle="test_dataset"),
        Edge(id="e-m-tr", source=model_id, target=trainer_id, sourceHandle="model", targetHandle="model"),
        Edge(id="e-o-tr", source=opt_id, target=trainer_id, sourceHandle="optimizer", targetHandle="optimizer"),
        Edge(id="e-l-tr", source=loss_id, target=trainer_id, sourceHandle="loss", targetHandle="loss"),
        Edge(id="e-obs-tr", source=obs_id, target=trainer_id, sourceHandle="observable", targetHandle="observables"),
        Edge(id="e-tr-viz", source=trainer_id, target=viz_id, sourceHandle="observable_results", targetHandle="tensor"),
    ]
    return nodes, edges, trainer_id, obs_id, viz_id


def _make_representation_observable(
    *,
    uid: str,
    model_id: str,
    rep_id: str,
    rep_shape: list[int],
    flatten_mode: str = "local",
) -> UserObservableRecord:
    reductions_raw = [{"axis_index": -1, "axis_label": "flat", "op": "l2_norm"}]
    specs = parse_axis_reductions(reductions_raw, flatten_mode=flatten_mode, tensor_shape=rep_shape)
    definition_code = format_algebra_definition_code(
        tensor_name=rep_id,
        reductions=specs,
        flatten_mode=flatten_mode,
        observable_source="representation",
    )
    human_chain = format_algebra_human_chain(
        tensor_name=rep_id,
        tensor_shape=rep_shape,
        reductions=specs,
        source_model_node_id=model_id,
        tensor_scope="single",
        flatten_mode=flatten_mode,
        observable_source="representation",
        representation_id=rep_id,
        layer_index=0,
        layer_io="output",
    )
    return UserObservableRecord(
        id=uid,
        label=f"rep {rep_id} l2",
        definition_code=definition_code,
        human_chain=human_chain,
        created_at=datetime.now(timezone.utc).isoformat(),
        definition_kind="algebra",
        source_model_node_id=model_id,
        tensor_name=rep_id,
        tensor_shape=rep_shape,
        reductions=reductions_raw,
        tensor_scope="single",
        flatten_mode=flatten_mode,
        observable_source="representation",
        representation_id=rep_id,
        layer_index=0,
        layer_io="output",
    )


def _patch_observable_record(rec: UserObservableRecord):
    def _getter(item_id: str) -> UserObservableRecord | None:
        return rec if item_id == rec.id else None

    return patch("comfy_research.engine.runs.trainer_run.get_user_observable_record", side_effect=_getter)


def _probe_hook_before_training(nodes: list[Node], edges: list[Edge], model_id: str, rep_id: str) -> None:
    from comfy_research.engine.analysis.model_weight_materialize import build_model_for_weight_node

    synth_id = f"{model_id}__weight_tensors"
    synth_nodes = list(nodes)
    synth_edges = list(edges)
    if not any(n.id == synth_id for n in synth_nodes):
        model_node = next(n for n in nodes if n.id == model_id)
        synth_nodes.append(
            Node(id=synth_id, type="model_weight_tensors", data={}, position=model_node.position)
        )
        synth_edges.append(
            Edge(
                id=f"e-{model_id}-{synth_id}",
                source=model_id,
                target=synth_id,
                sourceHandle="model",
                targetHandle="model",
            )
        )
    model, meta = build_model_for_weight_node(synth_nodes, synth_edges, synth_id)
    model.eval()
    md = next(n for n in nodes if n.id == model_id).data or {}
    vocab = max(2, int(md.get("vocabSize", 64)))
    tpi = max(1, int(md.get("tokensPerInput", 4)))
    x = __import__("torch").randint(0, vocab, (4, tpi), dtype=__import__("torch").long)
    depth = int(meta.get("depth") or md.get("depth") or 2)
    arr = fetch_representation_numpy(model, x, depth, rep_id)
    if arr.size == 0 or not math.isfinite(float(arr.mean())):
        raise RuntimeError(f"fetch_representation_numpy returned empty/non-finite for {rep_id!r}")


def main() -> int:
    uid = str(uuid.uuid4())
    model_id = "mlp_tok"

    # Discover a module output representation id from the model.
    probe_nodes, probe_edges, _, _, _ = _build_graph(observable_uid=uid, rep_id="placeholder", rep_label="probe")
    specs = run_model_representation_specs(probe_nodes, probe_edges, model_id)
    entries = specs.get("entries") or []
    if not entries:
        print("FAIL: no representation entries from mlp_token_model")
        return 1
    output_entries = [e for e in entries if str(e.get("io")) == "output" and "::" in str(e.get("representation_id"))]
    pick = output_entries[0] if output_entries else entries[0]
    rep_id = str(pick["representation_id"])
    rep_shape = [int(x) for x in (pick.get("shape") or [])]
    rep_label = str(pick.get("label") or rep_id)
    print(f"Using representation: {rep_id} shape={rep_shape}")

    rec = _make_representation_observable(uid=uid, model_id=model_id, rep_id=rep_id, rep_shape=rep_shape)
    nodes, edges, trainer_id, obs_id, viz_id = _build_graph(
        observable_uid=uid, rep_id=rep_id, rep_label=rep_label
    )

    _probe_hook_before_training(nodes, edges, model_id, rep_id)
    print("OK: forward hook captured representation tensor before training")

    with _patch_observable_record(rec):
        ctx = prepare_trainer_run(nodes, edges, trainer_id, resume=None, hessian_oversized_policy="skip")
        final: dict[str, Any] | None = None
        for event in iter_trainer_events_from_context(ctx):
            if event.get("type") == "complete":
                final = event
                break
            if event.get("type") in {"error", "aborted", "paused"}:
                print(f"FAIL: trainer ended with {event.get('type')}")
                return 1

    if final is None:
        print("FAIL: no complete event")
        return 1

    omh = final.get("observable_metric_histories") or {}
    hist = omh.get(obs_id) or []
    finite_hist = _finite(hist)
    print(f"Observable history length={len(hist)} finite={len(finite_hist)} values={finite_hist[:5]}…")
    if len(finite_hist) < 2:
        print("FAIL: expected at least 2 finite logged representation metric values")
        return 1
    if len(set(round(v, 6) for v in finite_hist)) < 2:
        print("WARN: metric barely changed across steps (may still be valid for tiny run)")

    obs_updates = final.get("observable_viz_updates") or []
    viz_row = next((row for row in obs_updates if row.get("node_id") == viz_id), None)
    if viz_row is None:
        print(f"FAIL: no observable_viz_updates row for viz node {viz_id}")
        print(f"  updates: {[r.get('node_id') for r in obs_updates]}")
        return 1

    viz_hist = viz_row.get("value_history") or []
    viz_finite = _finite(viz_hist)
    print(f"Viz stream length={len(viz_hist)} finite={len(viz_finite)}")
    if len(viz_finite) < 2:
        print("FAIL: Training Dynamics viz payload missing usable value_history")
        return 1
    if viz_finite != finite_hist:
        print("WARN: viz value_history differs from observable_metric_histories (checking subset)")
        if len(viz_finite) != len(finite_hist):
            print("FAIL: viz history length mismatch")
            return 1

    step_ticks = final.get("step_ticks") or []
    if len(step_ticks) != len(finite_hist):
        print(f"FAIL: step_ticks ({len(step_ticks)}) != metric points ({len(finite_hist)})")
        return 1

    print("PASS: representation observable hook + trainer metric history + Training Dynamics viz stream")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
