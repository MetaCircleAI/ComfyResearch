from comfy_research.engine.losses.loss_selection import choose_loss_kind
from comfy_research.schemas.graph import NodeKind


def test_choose_loss_kind_uses_capability_families() -> None:
    assert choose_loss_kind(NodeKind.diffusion_score_model, NodeKind.linear_dataset) == NodeKind.diffusion_mse_loss

    assert choose_loss_kind(NodeKind.mlp_token_model, NodeKind.linear_dataset) == NodeKind.cross_entropy_loss
    assert choose_loss_kind(NodeKind.linear_attention_model, NodeKind.linear_dataset) == NodeKind.cross_entropy_loss
    assert choose_loss_kind(NodeKind.resnet_model, NodeKind.gaussian_blob_dataset) == NodeKind.cross_entropy_loss
    assert choose_loss_kind(NodeKind.vit_model, NodeKind.mnist_dataset) == NodeKind.cross_entropy_loss

    assert choose_loss_kind(NodeKind.mlp_model, NodeKind.memorization_a_dataset) == NodeKind.cross_entropy_loss
    assert choose_loss_kind(NodeKind.mlp_model, NodeKind.memorization_b_dataset) == NodeKind.cross_entropy_loss

    assert choose_loss_kind(NodeKind.mlp_model, NodeKind.linear_dataset) == NodeKind.mse_loss
