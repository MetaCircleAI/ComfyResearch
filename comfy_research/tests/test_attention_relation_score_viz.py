from comfy_research.engine.trainer.observable_viz import observable_viz_metric_updates
from comfy_research.engine.trainer.attention_relation_metrics import attention_relation_pair_key
from comfy_research.schemas.graph import Edge, Node


def test_attention_relation_score_viz_receives_paired_multi_series() -> None:
    trainer = Node(id="trainer", type="trainer")
    score = Node(
        id="score",
        type="observable_attention_relation_score",
        data={"layerIndex": [1, 0], "headIndex": [0, 1]},
    )
    viz = Node(id="viz", type="observable_viz", data={"pairedObservableId": "score", "vizVariant": "user"})
    edges = [
        Edge(id="score-in", source="score", target="trainer", sourceHandle="observables", targetHandle="observables"),
        Edge(id="score-out", source="trainer", target="viz", sourceHandle="observable_results", targetHandle="tensor"),
    ]

    updates = observable_viz_metric_updates(
        edges,
        {node.id: node for node in (trainer, score, viz)},
        "trainer",
        {
            attention_relation_pair_key("score", 0): [0.1, 0.2],
            attention_relation_pair_key("score", 1): [0.3, 0.4],
        },
        {},
    )

    assert updates == [{
        "node_id": "viz",
        "paired_observable_id": "score",
        "value_histories": [[0.1, 0.2], [0.3, 0.4]],
        "series_labels": ["layer 1, head 0", "layer 0, head 1"],
    }]
