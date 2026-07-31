"""Regression checks for the double-descent graph-library template."""

from __future__ import annotations

import json
from pathlib import Path

from comfy_research.engine.runs.trainer_run import prepare_trainer_run
from comfy_research.generated.node_params import validate_node_params
from comfy_research.schemas.graph import Edge, GraphDocument, Node, NodeKind
from comfy_research.schemas.saved_graph_library import SavedGraphEntry


_REPO = Path(__file__).resolve().parents[2]
_TEMPLATE = _REPO / "data" / "graph_library" / "templates" / "84b21298-67aa-456d-b823-b97ce9352892.json"


def _load() -> tuple[SavedGraphEntry, list[Node], list[Edge]]:
    entry = SavedGraphEntry.model_validate(json.loads(_TEMPLATE.read_text(encoding="utf-8")))
    document = GraphDocument.model_validate(entry.document)
    return entry, list(document.nodes), list(document.edges)


def test_double_descent_template_validates_and_has_the_expected_sweep_pipeline() -> None:
    entry, nodes, edges = _load()

    assert entry.name == "double descent"
    assert entry.tier == "small"
    assert entry.savedAt > 0
    for node in nodes:
        kind = node.type.value if isinstance(node.type, NodeKind) else str(node.type)
        NodeKind(kind)
        validate_node_params(kind, node.data)

    by_id = {node.id: node for node in nodes}
    assert by_id["dd-dataset"].data["samplingMode"] == "fixed"
    assert by_id["dd-dataset"].data["noiseLevel"] == 0.5
    assert by_id["dd-model"].data["width"] == [
        1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256,
    ]
    assert by_id["dd-final-plot"].data["plotXParamKey"] == "model.width"
    assert by_id["dd-final-plot"].data["logScaleX"] is True

    connections = {(edge.source, edge.target, edge.sourceHandle, edge.targetHandle) for edge in edges}
    assert ("dd-trainer", "dd-training-viz", "loss_results", "tensor_list") in connections
    assert ("dd-training-viz", "dd-test-loss-selector", "out_tensor_list", "tensor_list") in connections
    assert ("dd-test-loss-selector", "dd-final-test-loss", "tensor_1", "tensor") in connections
    assert ("dd-final-test-loss-viz", "dd-sweep-table", "out_tensor", "stream") in connections
    assert ("dd-sweep-table", "dd-final-plot", "table", "table") in connections


def test_double_descent_template_prepares_a_training_run() -> None:
    _, nodes, edges = _load()

    context = prepare_trainer_run(nodes, edges, "dd-trainer", validate_only=True)

    assert context.trainer_task == "mse_regression"
    assert context.training_steps == 5000
    assert context.log_frequency == 1000
    assert context.train_size == 128
    assert context.test_size == 5000
