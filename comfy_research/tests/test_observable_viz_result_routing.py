from comfy_research.engine.trainer.observable_viz import observable_viz_metric_updates
from comfy_research.schemas.graph import Edge, Node, NodeKind


def test_observable_results_do_not_require_downstream_visualization_nodes() -> None:
    trainer = Node(id="trainer", type=NodeKind.trainer, data={})
    observable = Node(id="obs", type=NodeKind.observable_weight_l2, data={})
    edges = [
        Edge(
            id="obs-trainer",
            source="obs",
            target="trainer",
            sourceHandle="observable",
            targetHandle="observables",
        )
    ]

    updates = observable_viz_metric_updates(
        edges,
        {node.id: node for node in (trainer, observable)},
        "trainer",
        {"obs": [1.0, 2.0]},
        {},
    )

    assert len(updates) == 1
    assert updates[0]["paired_observable_id"] == "obs"
    assert updates[0]["value_histories"] == [[1.0, 2.0]]


def test_legacy_visualization_target_does_not_duplicate_observable_results() -> None:
    trainer = Node(id="trainer", type=NodeKind.trainer, data={})
    observable = Node(id="obs", type=NodeKind.observable_weight_l2, data={})
    visualization = Node(
        id="viz",
        type="observable_viz_weight_l2",
        data={"pairedObservableId": "obs"},
    )
    edges = [
        Edge(
            id="obs-trainer",
            source="obs",
            target="trainer",
            sourceHandle="observable",
            targetHandle="observables",
        ),
        Edge(
            id="trainer-viz",
            source="trainer",
            target="viz",
            sourceHandle="observable_results",
            targetHandle="tensor",
        ),
    ]

    updates = observable_viz_metric_updates(
        edges,
        {node.id: node for node in (trainer, observable, visualization)},
        "trainer",
        {"obs": [1.0, 2.0]},
        {},
    )

    assert len(updates) == 1
    assert updates[0]["node_id"] == "viz"
