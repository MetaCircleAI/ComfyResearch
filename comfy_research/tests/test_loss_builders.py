import pytest

torch = pytest.importorskip("torch")

from comfy_research.engine.losses.loss_builders import (  # noqa: E402
    LossCriterionContext,
    build_loss_criterion_for_node,
    dense_ce_memorization_a_slot_groups,
)
from comfy_research.schemas.graph import Node, NodeKind  # noqa: E402


def _node(node_type: NodeKind, data: dict | None = None) -> Node:
    return Node(id=str(node_type.value), type=node_type, data=dict(data or {}))


def test_registered_primary_loss_builders_construct_torch_criteria() -> None:
    mse = build_loss_criterion_for_node(
        _node(NodeKind.mse_loss),
        LossCriterionContext(trainer_task="mse_regression", target_flat_dim=2),
    )
    assert isinstance(mse, torch.nn.MSELoss)

    ce = build_loss_criterion_for_node(
        _node(NodeKind.cross_entropy_loss, {"labelSmoothing": 0.1}),
        LossCriterionContext(trainer_task="cross_entropy_dense", num_logits=3),
    )
    assert isinstance(ce, torch.nn.CrossEntropyLoss)
    assert ce.label_smoothing == pytest.approx(0.1)

    diffusion = build_loss_criterion_for_node(
        _node(NodeKind.diffusion_mse_loss),
        LossCriterionContext(trainer_task="diffusion_noise"),
    )
    assert isinstance(diffusion, torch.nn.MSELoss)


def test_dense_ce_slot_group_helper_preserves_mask_semantics() -> None:
    assert dense_ce_memorization_a_slot_groups(
        {"lossMaskMode": "all"},
        trainer_task="cross_entropy_dense",
        ds_type=NodeKind.memorization_a_dataset,
    ) == 1
    assert dense_ce_memorization_a_slot_groups(
        {"lossMaskMode": "last_context", "lossMaskContextLength": 3},
        trainer_task="cross_entropy_dense",
        ds_type=NodeKind.memorization_a_dataset,
    ) == 3
