from __future__ import annotations

import math
from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch

from comfy_research.engine.trainer.observable_metrics import _fourier_component_observable_value
from comfy_research.engine.trainer.observable_viz import observable_viz_metric_updates
from comfy_research.nodes.definitions.observables.fourier_component import record
from comfy_research.schemas.graph import Edge, Node, NodeKind


class _HalfWave(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * torch.sin(2.0 * math.pi * 5.0 * x[:, :1])


class _ExplodingModel(torch.nn.Module):
    def forward(self, _x: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("deliberate recorder failure")


def _wave() -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.linspace(0.0, 1.0, 201, dtype=torch.float32).reshape(-1, 1)
    return x, torch.sin(2.0 * math.pi * 5.0 * x)


def _rec(model: torch.nn.Module, *, task: str = "mse_regression") -> SimpleNamespace:
    x, y = _wave()
    return SimpleNamespace(
        model=model,
        trainer_task=task,
        _xr=x,
        _yr=y,
        observable_metric_histories=defaultdict(list),
    )


def _node(**data: object) -> Node:
    return Node(
        id="fourier",
        type=NodeKind.observable_fourier_component,
        data={"frequency": 5.0, "metric": "relative_projection_mse", "inputAxis": 0, "outputIndex": 0, **data},
    )


def test_fourier_component_math_covers_amplitude_ratio_and_relative_mse() -> None:
    x, y = _wave()
    pred = 0.5 * y
    assert _fourier_component_observable_value(x, y, pred, frequency=5.0, metric="amplitude_ratio", input_axis=0, output_index=0) == pytest.approx(0.5, abs=1e-6)
    assert _fourier_component_observable_value(x, y, pred, frequency=5.0, metric="relative_projection_mse", input_axis=0, output_index=0) == pytest.approx(0.25, abs=1e-6)


@pytest.mark.parametrize("input_axis,output_index", [(-1, 0), (1, 0), (0, -1), (0, 1)])
def test_fourier_component_invalid_axis_or_index_is_non_fatal_nan(input_axis: int, output_index: int) -> None:
    x, y = _wave()
    assert math.isnan(_fourier_component_observable_value(x, y, y, frequency=5.0, metric="relative_projection_mse", input_axis=input_axis, output_index=output_index))


def test_recorder_records_mse_and_non_mse_branches() -> None:
    mse = _rec(_HalfWave())
    record(mse, _node())
    assert mse.observable_metric_histories["fourier"] == pytest.approx([0.25], abs=1e-6)

    non_mse = _rec(_ExplodingModel(), task="cross_entropy_dense")
    record(non_mse, _node())
    assert len(non_mse.observable_metric_histories["fourier"]) == 1
    assert math.isnan(non_mse.observable_metric_histories["fourier"][0])


def test_recorder_degrades_calculation_exception_to_nan() -> None:
    rec = _rec(_ExplodingModel())
    record(rec, _node())
    assert len(rec.observable_metric_histories["fourier"]) == 1
    assert math.isnan(rec.observable_metric_histories["fourier"][0])


def test_recorder_history_routes_to_user_observable_viz_payload() -> None:
    trainer = Node(id="trainer", type=NodeKind.trainer, data={})
    observable = _node()
    viz = Node(id="viz", type=NodeKind.observable_viz, data={"pairedObservableId": "fourier", "vizVariant": "user"})
    edges = [
        Edge(id="obs-trainer", source="fourier", target="trainer", sourceHandle="observable", targetHandle="observables"),
        Edge(id="trainer-viz", source="trainer", target="viz", sourceHandle="observable_results", targetHandle="tensor"),
    ]
    rec = _rec(_HalfWave())
    record(rec, observable)
    updates = observable_viz_metric_updates(edges, {node.id: node for node in (trainer, observable, viz)}, "trainer", rec.observable_metric_histories, {})
    assert len(updates) == 1
    assert updates[0]["node_id"] == "viz"
    assert updates[0]["paired_observable_id"] == "fourier"
    assert updates[0]["value_history"] == pytest.approx([0.25], abs=1e-6)
