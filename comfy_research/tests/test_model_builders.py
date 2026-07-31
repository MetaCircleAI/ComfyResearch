import pytest

torch = pytest.importorskip("torch")

from comfy_research.engine.models.model_builders import ModelBuildContext, build_model_for_node  # noqa: E402
from comfy_research.schemas.graph import Node, NodeKind  # noqa: E402


def _node(node_type: NodeKind, data: dict | None = None) -> Node:
    return Node(id=str(node_type.value), type=node_type, data=dict(data or {}))


def test_registered_vision_model_builders_construct_modules() -> None:
    context = ModelBuildContext(input_channels=1, image_size=16, num_classes=3)

    resnet = build_model_for_node(_node(NodeKind.resnet_model, {"variant": "resnet18"}), context)
    assert isinstance(resnet, torch.nn.Module)
    assert tuple(resnet(torch.zeros(2, 1, 16, 16)).shape) == (2, 3)

    vit = build_model_for_node(_node(NodeKind.vit_model, {"variant": "tiny", "patchSize": 4}), context)
    assert isinstance(vit, torch.nn.Module)
    assert tuple(vit(torch.zeros(2, 1, 16, 16)).shape) == (2, 3)


def test_vgg11_cifar_uses_paper_initialization() -> None:
    model = build_model_for_node(
        _node(NodeKind.vgg11_cifar_model, {"seed": 0}),
        ModelBuildContext(input_channels=3, image_size=32, num_classes=10),
    )
    affine = [m for m in model.modules() if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear))]
    batch_norms = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    assert affine
    assert batch_norms
    assert all(m.bias is None or torch.count_nonzero(m.bias).item() == 0 for m in affine)
    assert all(torch.equal(m.weight, torch.ones_like(m.weight)) for m in batch_norms)
    assert all(torch.equal(m.bias, torch.zeros_like(m.bias)) for m in batch_norms)


def test_registered_canvas_md_model_builders_construct_modules() -> None:
    for node_type in (
        NodeKind.afno_lite_spatiotemporal_model,
        NodeKind.mpp_spatiotemporal_model,
        NodeKind.numeric_hyena_model,
        NodeKind.numeric_transformer_model,
    ):
        model = build_model_for_node(_node(node_type), ModelBuildContext())
        assert isinstance(model, torch.nn.Module)


def test_registered_mlp_model_builders_construct_modules() -> None:
    context = ModelBuildContext(input_dim=2, output_dim=3)
    for node_type in (NodeKind.mlp_model, NodeKind.gated_mlp_model, NodeKind.moe_mlp_model):
        model = build_model_for_node(
            _node(node_type, {"depth": 1, "width": 4, "numExperts": 2}),
            context,
        )
        assert isinstance(model, torch.nn.Module)
        assert tuple(model(torch.zeros(2, 2)).shape) == (2, 3)


def test_registered_mlp_token_model_builders_construct_modules() -> None:
    for node_type in (NodeKind.mlp_token_model, NodeKind.gated_mlp_token_model, NodeKind.moe_mlp_token_model):
        model = build_model_for_node(
            _node(
                node_type,
                {
                    "vocabSize": 7,
                    "embedDim": 3,
                    "tokensPerInput": 2,
                    "depth": 1,
                    "width": 4,
                    "numExperts": 2,
                },
            ),
            ModelBuildContext(),
        )
        assert isinstance(model, torch.nn.Module)
        assert tuple(model(torch.zeros(2, 2, dtype=torch.long)).shape) == (2, 7)


def test_registered_token_sequence_model_builders_construct_modules() -> None:
    common = {
        "vocabSize": 7,
        "embedDim": 4,
        "contextLength": 3,
        "numHeads": 1,
        "depth": 1,
        "numLayers": 1,
        "convKernel": 3,
        "ffMult": 1,
        "numSlots": 2,
        "slotIters": 1,
    }
    for node_type in (
        NodeKind.attention_only_model,
        NodeKind.linear_attention_model,
        NodeKind.diagonal_ssm_token_model,
        NodeKind.rwkv_time_mix_token_model,
        NodeKind.hyena_like_conv_model,
        NodeKind.slot_attention_token_model,
        NodeKind.transformer_token_model,
    ):
        model = build_model_for_node(_node(node_type, common), ModelBuildContext())
        assert isinstance(model, torch.nn.Module)
        assert tuple(model(torch.zeros(2, 3, dtype=torch.long)).shape) in {(2, 7), (2, 3, 7)}

    multi = build_model_for_node(
        _node(NodeKind.transformer_multi_token_model, {**common, "tokensPerPosition": 2}),
        ModelBuildContext(),
    )
    assert isinstance(multi, torch.nn.Module)
    assert tuple(multi(torch.zeros(2, 3, 2, dtype=torch.long)).shape) == (2, 2, 7)


def test_registered_diffusion_score_model_builder_constructs_module() -> None:
    model = build_model_for_node(
        _node(NodeKind.diffusion_score_model, {"hiddenDim": 8, "depth": 1, "timeEmbedDim": 8, "diffusionTimesteps": 4}),
        ModelBuildContext(input_dim=3),
    )
    assert isinstance(model, torch.nn.Module)
    pred = model(torch.zeros(2, 3), torch.zeros(2, dtype=torch.long))
    assert tuple(pred.shape) == (2, 3)



def test_registered_legacy_residual_model_builders_construct_modules() -> None:
    residual = build_model_for_node(
        _node(NodeKind.residual_ln_model, {"dim": 4, "depth": 2, "alpha": 0.5, "activation": "relu"}),
        ModelBuildContext(),
    )
    assert isinstance(residual, torch.nn.Module)
    assert tuple(residual(torch.zeros(2, 4)).shape) == (2, 4)

    crl = build_model_for_node(
        _node(
            NodeKind.crl_residual_mlp,
            {
                "stateDim": 3,
                "actionDim": 2,
                "goalDim": 1,
                "actorWidth": 8,
                "criticWidth": 8,
                "actorDepth": 4,
                "criticDepth": 4,
                "embedDim": 8,
            },
        ),
        ModelBuildContext(),
    )
    assert isinstance(crl, torch.nn.Module)
