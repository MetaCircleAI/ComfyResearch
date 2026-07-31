import pytest

torch = pytest.importorskip("torch")

from comfy_research.engine.optimizers.optimizer_builders import (  # noqa: E402
    OptimizerBuildConfig,
    build_optimizer_for_node,
)
from comfy_research.schemas.graph import Node, NodeKind  # noqa: E402


def _node(node_type: NodeKind) -> Node:
    return Node(id=str(node_type.value), type=node_type, data={})


def _config() -> OptimizerBuildConfig:
    return OptimizerBuildConfig(
        lr=1e-3,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        momentum=0.1,
        weight_decay=0.0,
        muon_ns_steps=1,
    )


def test_registered_optimizer_builders_construct_torch_optimizers() -> None:
    model = torch.nn.Linear(2, 1)

    adam = build_optimizer_for_node(_node(NodeKind.adam_optimizer), model, _config())
    assert isinstance(adam, torch.optim.Adam)

    sgd = build_optimizer_for_node(_node(NodeKind.sgd_optimizer), model, _config())
    assert isinstance(sgd, torch.optim.SGD)

    muon = build_optimizer_for_node(_node(NodeKind.muon_optimizer), model, _config())
    assert isinstance(muon, torch.optim.Optimizer)
