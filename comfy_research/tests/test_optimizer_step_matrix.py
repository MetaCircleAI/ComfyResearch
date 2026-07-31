from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from comfy_research.engine.optimizers.optimizer_builders import (  # noqa: E402
    OptimizerBuildConfig,
    build_optimizer_for_node,
)
from comfy_research.schemas.graph import Node  # noqa: E402


OPTIMIZER_TYPES = (
    "adam_optimizer",
    "adamw_optimizer",
    "sgd_optimizer",
    "signsgd_optimizer",
    "muon_optimizer",
    "shampoo_optimizer",
    "soap_optimizer",
)


def _config() -> OptimizerBuildConfig:
    return OptimizerBuildConfig(
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        momentum=0.1,
        weight_decay=0.0,
        muon_ns_steps=1,
        precondition_frequency=1,
        max_preconditioner_dim=64,
        use_mup_lr_schedule=False,
        mup_embed_lr_mult=1.0,
        mup_hidden_lr_mult=1.0,
        mup_output_lr_mult=1.0,
    )


@pytest.mark.parametrize("optimizer_type", OPTIMIZER_TYPES)
def test_optimizer_runs_three_finite_cpu_steps(optimizer_type: str) -> None:
    torch.manual_seed(0)
    model = torch.nn.Linear(2, 1)
    optimizer = build_optimizer_for_node(
        Node(id="optimizer", type=optimizer_type, data={}),
        model,
        _config(),
    )
    inputs = torch.tensor([[1.0, -1.0], [0.5, 2.0]])
    initial_params = [parameter.detach().clone() for parameter in model.parameters()]

    for _ in range(3):
        optimizer.zero_grad()
        loss = model(inputs).square().mean()
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()

    if optimizer_type == "signsgd_optimizer":
        assert not optimizer.state
    else:
        assert optimizer.state
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
    assert any(
        not torch.equal(initial, current)
        for initial, current in zip(initial_params, model.parameters())
    )
