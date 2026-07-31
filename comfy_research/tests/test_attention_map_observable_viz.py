import pytest

from comfy_research.engine.trainer.attention_map_history import (
    MAX_ATTENTION_MAP_FRAMES,
    append_frame,
    selected_tuples,
)
from comfy_research.engine.trainer.observable_viz import observable_viz_metric_updates
from comfy_research.schemas.graph import Edge, Node


def test_attention_map_viz_receives_only_its_paired_history() -> None:
    trainer = Node(id="trainer", type="trainer")
    attention = Node(id="attention", type="observable_attention_map")
    other = Node(id="other", type="observable_attention_map")
    paired_viz = Node(
        id="paired-viz",
        type="observable_viz",
        data={"pairedObservableId": "attention", "vizVariant": "attention_map"},
    )
    other_viz = Node(
        id="other-viz",
        type="observable_viz",
        data={"pairedObservableId": "other", "vizVariant": "attention_map"},
    )
    edges = [
        Edge(id="obs", source="attention", target="trainer", sourceHandle="observables", targetHandle="observables"),
        Edge(id="other-obs", source="other", target="trainer", sourceHandle="observables", targetHandle="observables"),
        Edge(id="paired-result", source="trainer", target="paired-viz", sourceHandle="observable_results", targetHandle="tensor"),
        Edge(id="other-result", source="trainer", target="other-viz", sourceHandle="observable_results", targetHandle="tensor"),
    ]

    updates = observable_viz_metric_updates(
        edges,
        {node.id: node for node in (trainer, attention, other, paired_viz, other_viz)},
        trainer.id,
        {},
        {},
        {
            "attention": [
                {
                    "step": 3,
                    "slices": [
                        {
                            "layer": 0,
                            "batch": 0,
                            "head": 0,
                            "map": [[0.25, 0.75], [0.5, 0.5]],
                            "token_ids": [17, 9],
                            "source_shape": [2, 2],
                            "row_start": 0,
                            "col_start": 0,
                        }
                    ],
                }
            ]
        },
    )

    assert updates == [
        {
            "node_id": "paired-viz",
            "attention_map_frames": [
                {
                    "step": 3,
                    "slices": [
                        {
                            "layer": 0,
                            "batch": 0,
                            "head": 0,
                            "map": [[0.25, 0.75], [0.5, 0.5]],
                            "token_ids": [17, 9],
                            "source_shape": [2, 2],
                            "row_start": 0,
                            "col_start": 0,
                        }
                    ],
                }
            ],
            "paired_observable_id": "attention",
        }
    ]


def test_attention_map_viz_fallback_keeps_frames_when_execution_graph_omits_viz() -> None:
    """Execution-graph serialization excludes downstream viz nodes before training."""
    trainer = Node(id="trainer", type="trainer")
    attention = Node(id="attention", type="observable_attention_map")
    frames = [
        {
            "step": 3,
            "slices": [
                {
                    "layer": 0,
                    "batch": 0,
                    "head": 0,
                    "map": [[0.25, 0.75], [0.5, 0.5]],
                    "token_ids": [17, 9],
                    "source_shape": [2, 2],
                    "row_start": 0,
                    "col_start": 0,
                }
            ],
        }
    ]
    edges = [
        Edge(id="obs", source="attention", target="trainer", sourceHandle="observables", targetHandle="observables"),
    ]

    updates = observable_viz_metric_updates(
        edges,
        {node.id: node for node in (trainer, attention)},
        trainer.id,
        {},
        {},
        {attention.id: frames},
    )

    assert updates == [
        {
            "node_id": "trainer::__observable_result__attention",
            "attention_map_frames": frames,
            "paired_observable_id": attention.id,
        }
    ]

def test_attention_map_selection_is_deduplicated_bounded_and_strict() -> None:
    assert selected_tuples(
        "attention",
        {"attentionLayerIndices": "1, 0, 1", "attentionBatchIndices": "2", "attentionHeadIndices": "0, 1"},
    ) == [(0, 2, 0), (0, 2, 1), (1, 2, 0), (1, 2, 1)]
    with pytest.raises(Exception, match="limit is 20"):
        selected_tuples(
            "attention",
            {"attentionLayerIndices": "0,1,2", "attentionBatchIndices": "0,1,2", "attentionHeadIndices": "0,1,2"},
        )


def test_attention_map_history_evicts_whole_oldest_frames() -> None:
    history: list[dict[str, object]] = []
    for step in range(MAX_ATTENTION_MAP_FRAMES + 1):
        append_frame(history, {"step": step, "slices": [{"layer": 0}, {"layer": 1}]})
    assert len(history) == MAX_ATTENTION_MAP_FRAMES
    assert history[0]["step"] == 1
    assert all(len(frame["slices"]) == 2 for frame in history)
