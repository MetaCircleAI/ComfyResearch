from __future__ import annotations

from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import patch

import torch

from comfy_research.engine.trainer.information_plane import information_plane_for_model
from comfy_research.engine.trainer.observable_viz import observable_viz_metric_updates
from comfy_research.nodes.definitions.observables.information_plane import record
from comfy_research.schemas.graph import Edge, Node, NodeKind


class _TinyNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.Sequential(torch.nn.Linear(2, 4), torch.nn.Tanh(), torch.nn.Linear(4, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class _LinearOnly(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :1]


class _Exploding(torch.nn.Module):
    def forward(self, _x: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("expected measurement failure")


def _arrays(count: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.linspace(-1.0, 1.0, count * 2).reshape(count, 2)
    y = (x[:, 0] > 0).long()
    return x, y


def _node(**data: object) -> Node:
    return Node(id="plane", type=NodeKind.observable_information_plane, data={"bins": 8, "maxSamples": 16, "includeOutput": True, **data})


def _rec(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor) -> SimpleNamespace:
    return SimpleNamespace(model=model, x_test_t=x, y_test_t=y, _xr=x, _yr=y, observable_embedding_histories=defaultdict(list))


def test_information_plane_collects_bounded_activation_and_restores_training_mode() -> None:
    model = _TinyNet()
    model.train()
    x, y = _arrays()
    points = information_plane_for_model(model, x, y, bins=8)
    assert model.training is True
    assert len(points) == 2
    assert all(len(point) == 2 and all(value >= 0 for value in point) for point in points)


def test_recorder_samples_deterministically_and_routes_embedding_payload() -> None:
    x, y = _arrays(48)
    observable = _node(maxSamples=16)
    rec = _rec(_TinyNet(), x, y)
    record(rec, observable)
    assert len(rec.observable_embedding_histories["plane"]) == 1
    assert rec.observable_embedding_histories["plane"][0]

    trainer = Node(id="trainer", type=NodeKind.trainer, data={})
    viz = Node(id="viz", type=NodeKind.observable_viz, data={"pairedObservableId": "plane", "vizVariant": "information_plane"})
    edges = [
        Edge(id="obs-trainer", source="plane", target="trainer", sourceHandle="observable", targetHandle="observables"),
        Edge(id="trainer-viz", source="trainer", target="viz", sourceHandle="observable_results", targetHandle="tensor"),
    ]
    updates = observable_viz_metric_updates(edges, {node.id: node for node in (trainer, observable, viz)}, "trainer", {}, rec.observable_embedding_histories)
    assert updates == [{"node_id": "viz", "embedding_history": rec.observable_embedding_histories["plane"], "paired_observable_id": "plane"}]


def test_recorder_forwards_protocol_binning_and_output_mapping() -> None:
    x, y = _arrays(16)
    rec = _rec(_TinyNet(), x, y)
    observable = _node(
        binning="idnns_equal_points",
        outputMapping="probability",
    )
    with patch(
        "comfy_research.engine.trainer.information_plane.information_plane_for_model",
        return_value=[[1.0, 2.0]],
    ) as measure:
        record(rec, observable)

    assert rec.observable_embedding_histories["plane"] == [[[1.0, 2.0]]]
    assert measure.call_args.kwargs["binning"] == "idnns_equal_points"
    assert measure.call_args.kwargs["output_mapping"] == "probability"


def test_recorder_degrades_missing_hook_shape_and_calculation_failures_to_empty_frames() -> None:
    x, y = _arrays()
    no_hook = _rec(_LinearOnly(), x, y)
    record(no_hook, _node(includeOutput=False))
    assert no_hook.observable_embedding_histories["plane"] == [[]]

    bad_shape = _rec(_TinyNet(), x, y[:-1])
    record(bad_shape, _node())
    assert bad_shape.observable_embedding_histories["plane"] == [[]]

    exploding = _rec(_Exploding(), x, y)
    record(exploding, _node())
    assert exploding.observable_embedding_histories["plane"] == [[]]
