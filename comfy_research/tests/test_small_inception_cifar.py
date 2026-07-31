from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from comfy_research.engine.models.cifar_models import build_small_inception_cifar  # noqa: E402
from comfy_research.engine.models.model_builders import ModelBuildContext, build_model_for_node  # noqa: E402
from comfy_research.nodes.registry import load_definitions, model_defs_builders  # noqa: E402
from comfy_research.schemas.graph import Node, NodeKind  # noqa: E402


def test_small_inception_builds_through_registered_provider() -> None:
    load_definitions()
    assert NodeKind.small_inception_cifar_model.value in model_defs_builders()
    model = build_model_for_node(
        Node(id="model", type=NodeKind.small_inception_cifar_model, data={"seed": 7}),
        ModelBuildContext(input_channels=3, image_size=32, num_classes=10),
    )
    assert isinstance(model, torch.nn.Module)


@pytest.mark.parametrize("image_size", (28, 32))
def test_small_inception_accepts_paper_and_default_cifar_sizes(image_size: int) -> None:
    model = build_small_inception_cifar(in_channels=3, num_classes=10).eval()
    with torch.no_grad():
        output = model(torch.zeros(2, 3, image_size, image_size))
    assert output.shape == (2, 10)


def test_small_inception_matches_paper_parameter_count_for_ten_classes() -> None:
    model = build_small_inception_cifar(in_channels=3, num_classes=10)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 1_649_402
