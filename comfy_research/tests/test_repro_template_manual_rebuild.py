"""Simulate deleting template schedule nodes and re-adding them like a user.

Two layers:
1. **API / graph rebuild (feasible now)** — drop nodes+edges, recreate from catalog
   defaults + paper params, rewire handles, ``prepare_trainer_run(validate_only=True)``.
   This matches what the backend sees after a human rebuilds the graph; it does **not**
   drive the Add-Node modal UI.
2. **Headless browser (not gated yet)** — Playwright would need stable ``data-testid``s
   and React Flow handle targeting; see ``frontend/e2e/README.md``. Marked skip here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from comfy_research.engine.runs.trainer_run import prepare_trainer_run
from comfy_research.generated.node_params import validate_node_params
from comfy_research.main import app
from comfy_research.schemas.graph import Edge, GraphDocument, Node

_REPO = Path(__file__).resolve().parents[2]
_FIG1 = _REPO / "data" / "graph_library" / "templates" / "repro-jastrzbski-fig1-vgg11.json"

# validate_only=True must not touch the vision dataset build (CIFAR download).
_VISION_BUILD = "comfy_research.engine.trainer.dataset_materialize.build_vision_numpy_arrays"

# Paper parameters used when rebuilding the two schedule nodes.
_CBS_DEFAULTS: dict[str, Any] = {
    "batchMin": 128,
    "batchMax": 640,
    "cycleLengthEpochs": 10,
    "refBatchSize": 128,
    "cycleLengthSteps": 0,
    "scheduleMode": "square_epoch",
}
_CLR_DEFAULTS: dict[str, Any] = {
    "lrMin": 0.001,
    "lrMax": 0.005,
    "cycleLengthEpochs": 10,
    "refBatchSize": 128,
    "cycleLengthSteps": 0,
    "scheduleMode": "square_epoch",
}


def _load_fig1() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    entry = json.loads(_FIG1.read_text(encoding="utf-8"))
    doc = entry["document"]
    return list(doc["nodes"]), list(doc["edges"])


def _as_models(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> tuple[list[Node], list[Edge]]:
    return [Node.model_validate(n) for n in nodes], [Edge.model_validate(e) for e in edges]


def _strip_cyclic_schedules(
    nodes: list[dict[str, Any]], edges: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    """Remove cyclic schedule nodes and any edges touching them. Return remembered trainer/opt ids."""
    drop_ids = {n["id"] for n in nodes if n["type"] in ("cyclic_batch_schedule", "cyclic_lr_schedule")}
    remembered = {
        "tr_cbs": next(n["id"] for n in nodes if n["type"] == "trainer" and "CBS" in str((n.get("data") or {}).get("instanceTitle", ""))),
        "tr_clr": next(n["id"] for n in nodes if n["type"] == "trainer" and "CLR" in str((n.get("data") or {}).get("instanceTitle", ""))),
        "sgd_clr": next(n["id"] for n in nodes if n["type"] == "sgd_optimizer" and n["id"].endswith("sgd_clr")),
    }
    nodes2 = [n for n in nodes if n["id"] not in drop_ids]
    edges2 = [e for e in edges if e["source"] not in drop_ids and e["target"] not in drop_ids]
    assert not any(n["type"].startswith("cyclic_") for n in nodes2)
    return nodes2, edges2, remembered


def _add_node(nodes: list[dict[str, Any]], *, type_: str, data: dict[str, Any], position: dict[str, float]) -> str:
    """Mimic Add-Node modal: new id, catalog type, defaults (+ user edits already in ``data``)."""
    validate_node_params(type_, data)
    nid = f"manual-{type_}-{uuid4().hex[:8]}"
    nodes.append(
        {
            "id": nid,
            "type": type_,
            "position": position,
            "data": {**data, "instanceTitle": f"manual {type_}"},
            "parentId": None,
            "extent": None,
            "hidden": None,
            "style": None,
        }
    )
    return nid


def _add_edge(
    edges: list[dict[str, Any]],
    *,
    source: str,
    target: str,
    source_handle: str,
    target_handle: str,
) -> None:
    edges.append(
        {
            "id": f"e-{uuid4().hex[:10]}",
            "source": source,
            "target": target,
            "sourceHandle": source_handle,
            "targetHandle": target_handle,
        }
    )


def test_catalog_lists_cyclic_nodes_for_add_modal() -> None:
    """What the Add-Node search modal is backed by (HTTP), not the canvas itself."""
    client = TestClient(app)
    r = client.get("/api/node-categories")
    assert r.status_code == 200
    payload = r.json()
    types: set[str] = set()
    for cat in payload.get("categories", []):
        for child in cat.get("children", []):
            tid = child.get("id")
            if tid:
                types.add(str(tid))
    assert "cyclic_lr_schedule" in types
    assert "cyclic_batch_schedule" in types
    assert "vgg11_cifar_model" in types


def test_fig1_rebuild_cyclic_nodes_via_graph_api_shape() -> None:
    """Delete CBS/CLR schedule nodes, re-add from catalog defaults, rewire, compile."""
    nodes, edges = _load_fig1()
    nodes, edges, ids = _strip_cyclic_schedules(nodes, edges)

    cbs_id = _add_node(nodes, type_="cyclic_batch_schedule", data=dict(_CBS_DEFAULTS), position={"x": 280, "y": 280})
    clr_id = _add_node(nodes, type_="cyclic_lr_schedule", data=dict(_CLR_DEFAULTS), position={"x": 280, "y": 620})
    _add_edge(edges, source=cbs_id, target=ids["tr_cbs"], source_handle="batch_schedule", target_handle="batch_schedule")
    _add_edge(edges, source=clr_id, target=ids["sgd_clr"], source_handle="lr_schedule", target_handle="lr_schedule")

    n_models, e_models = _as_models(nodes, edges)
    for node in n_models:
        if node.type == "trainer":
            node.data["computeDevice"] = "cpu"
            node.data["remoteGpu"] = False
    with patch(_VISION_BUILD):
        ctx_cbs = prepare_trainer_run(n_models, e_models, ids["tr_cbs"], validate_only=True)
        ctx_clr = prepare_trainer_run(n_models, e_models, ids["tr_clr"], validate_only=True)

    assert ctx_cbs.cyclic_batch_cycle_steps > 0
    assert ctx_cbs.training_steps == 63450
    assert ctx_clr.cyclic_lr_cycle_steps > 0
    assert ctx_clr.training_steps == 105600


def test_post_train_accepts_rebuilt_fig1_graph_body() -> None:
    """POST /api/train with rebuilt graph: request validates; prepare runs validate_only via patch."""
    nodes, edges = _load_fig1()
    nodes, edges, ids = _strip_cyclic_schedules(nodes, edges)
    cbs_id = _add_node(nodes, type_="cyclic_batch_schedule", data=dict(_CBS_DEFAULTS), position={"x": 1, "y": 1})
    _add_edge(edges, source=cbs_id, target=ids["tr_cbs"], source_handle="batch_schedule", target_handle="batch_schedule")
    for n in nodes:
        if n["id"] == ids["tr_cbs"]:
            n.setdefault("data", {})["computeDevice"] = "cpu"
            n["data"]["remoteGpu"] = False

    client = TestClient(app)

    def _prep(nodes_arg, edges_arg, trainer_node_id, **kwargs):
        return prepare_trainer_run(nodes_arg, edges_arg, trainer_node_id, validate_only=True)

    with (
        patch("comfy_research.api.train.prepare_trainer_run", side_effect=_prep),
        patch("comfy_research.api.train.iter_trainer_events_from_context", return_value=iter([{"type": "complete"}])),
        patch(_VISION_BUILD),
    ):
        r = client.post(
            "/api/train",
            json={"trainer_node_id": ids["tr_cbs"], "nodes": nodes, "edges": edges},
        )
    assert r.status_code == 200, r.text[:500]
    # NDJSON stream: at least one complete event
    lines = [ln for ln in r.text.splitlines() if ln.strip()]
    assert lines
    assert json.loads(lines[-1]).get("type") == "complete"


@pytest.mark.skip(reason="Playwright Add-Node + React Flow wire not instrumented (no data-testid); see frontend/e2e/README.md")
def test_fig1_rebuild_via_headless_browser_placeholder() -> None:
    """Future: double-click canvas → search cyclic → add → drag handles → Train."""
    raise AssertionError("unimplemented")
