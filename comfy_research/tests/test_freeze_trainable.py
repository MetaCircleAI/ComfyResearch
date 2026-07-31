"""Tests for per-layer freeze/trainable support.

Covers:
- Four freeze markers each take effect (freeze, trainable, requiresGrad, requires_grad).
- Frozen layer parameters are not updated during training.
- Adam, SGD, AdamW, SignSGD, Shampoo, SOAP all exclude frozen parameters.
- All-params-frozen behaviour is consistent per optimizer.
- muP Adam path raises when all params are frozen.
"""

from __future__ import annotations

from typing import cast

import pytest

torch = pytest.importorskip("torch")

import torch.nn as nn

from comfy_research.engine.models.atomic_layer_chain import _is_frozen, build_atomic_layer_module
from comfy_research.engine.optimizers.optimizer_builders import (
    OptimizerBuildConfig,
    _build_adam_optimizer,
    _build_adamw_optimizer,
    _build_sgd_optimizer,
    _build_shampoo_optimizer,
    _build_signsgd_optimizer,
    _build_soap_optimizer,
    _trainable_parameters,
)
from comfy_research.schemas.graph import Node, NodeKind

# ── helpers ──────────────────────────────────────────────────────────

_FREEZE_TRUE = {"freeze": True}
_TRAINABLE_FALSE = {"trainable": False}
_REQUIRES_GRAD_FALSE = {"requiresGrad": False}
_REQUIRES_GRAD_SNAKE_FALSE = {"requires_grad": False}
_NO_FREEZE = {}
_ALL_MARKERS = [_FREEZE_TRUE, _TRAINABLE_FALSE, _REQUIRES_GRAD_FALSE, _REQUIRES_GRAD_SNAKE_FALSE]

_MARKER_NAMES = ["freeze: true", "trainable: false", "requiresGrad: false", "requires_grad: false"]


def _make_linear_node(data: dict | None = None) -> Node:
    return Node(
        id="test-linear",
        type=NodeKind.linear_layer,
        data={"inFeatures": 4, "outFeatures": 2, "bias": True, **(data or {})},
    )


def _config() -> OptimizerBuildConfig:
    return OptimizerBuildConfig(
        lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8,
        momentum=0.1, weight_decay=0.0, muon_ns_steps=1,
    )


def _param_ids(optimizer: torch.optim.Optimizer) -> set[int]:
    return {id(p) for group in optimizer.param_groups for p in group["params"]}


# ── 1. _is_frozen detects all four markers ───────────────────────────

@pytest.mark.parametrize("marker, name", list(zip(_ALL_MARKERS, _MARKER_NAMES)))
def test_is_frozen_true_for_marker(marker: dict, name: str) -> None:
    assert _is_frozen(marker), f"{name} should be detected as frozen"


def test_is_frozen_false_without_markers() -> None:
    assert not _is_frozen(_NO_FREEZE)


def test_is_frozen_false_when_freeze_is_false() -> None:
    assert not _is_frozen({"freeze": False})
    assert not _is_frozen({"trainable": True})
    assert not _is_frozen({"requiresGrad": True})
    assert not _is_frozen({"requires_grad": True})


# ── 2. build_atomic_layer_module with freeze ─────────────────────────

@pytest.mark.parametrize("marker, name", list(zip(_ALL_MARKERS, _MARKER_NAMES)))
def test_build_atomic_layer_module_frozen(marker: dict, name: str) -> None:
    """Each freeze marker causes the built module to have requires_grad=False."""
    module = build_atomic_layer_module(_make_linear_node(marker))
    assert all(not p.requires_grad for p in module.parameters()), (
        f"{name}: all parameters should have requires_grad=False"
    )


def test_build_atomic_layer_module_trainable_by_default() -> None:
    module = build_atomic_layer_module(_make_linear_node())
    assert all(p.requires_grad for p in module.parameters())


# ── 3. _trainable_parameters filter ──────────────────────────────────

def test_trainable_parameters_excludes_frozen() -> None:
    """_trainable_parameters returns only requires_grad=True params."""
    m = nn.Linear(4, 2)
    m.requires_grad_(False)
    trainable = list(_trainable_parameters(m))
    assert len(trainable) == 0


def test_trainable_parameters_includes_trainable() -> None:
    m = nn.Sequential(nn.Linear(4, 2), nn.ReLU(), nn.Linear(2, 1))
    # Only freeze first layer
    cast(nn.Linear, m[0]).requires_grad_(False)
    trainable = list(_trainable_parameters(m))
    # ReLU has no params; only the second Linear is trainable
    assert len(trainable) > 0
    assert all(p.requires_grad for p in trainable)


# ── 4. each optimizer builder excludes frozen params ──────────────────

_ALL_BUILDERS = [
    ("adam", _build_adam_optimizer),
    ("adamw", _build_adamw_optimizer),
    ("sgd", _build_sgd_optimizer),
    ("signsgd", _build_signsgd_optimizer),
    ("shampoo", _build_shampoo_optimizer),
    ("soap", _build_soap_optimizer),
]


@pytest.mark.parametrize("name, builder", _ALL_BUILDERS)
def test_optimizer_excludes_frozen_params(name: str, builder) -> None:
    """Each optimizer receives only trainable params."""
    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 2),
    )
    # freeze the first Linear
    cast(nn.Linear, model[0]).requires_grad_(False)

    opt = builder(model, _config())
    param_ids = _param_ids(opt)
    frozen_ids = {id(p) for p in model[0].parameters()}
    trainable_ids = {id(p) for p in list(model[1].parameters()) + list(model[2].parameters())}

    # None of the frozen params should be in the optimizer
    assert param_ids.isdisjoint(frozen_ids), (
        f"{name}: frozen parameters should not be passed to the optimizer"
    )
    # All trainable params should be in the optimizer
    assert trainable_ids.issubset(param_ids), (
        f"{name}: all trainable parameters should be passed to the optimizer"
    )


# ── 5. all params frozen → consistent error ──────────────────────────

@pytest.mark.parametrize("name, builder", _ALL_BUILDERS)
def test_optimizer_all_frozen_raises(name: str, builder) -> None:
    """When all params are frozen, each optimizer should raise a clear error."""
    model = nn.Linear(4, 2)
    model.requires_grad_(False)
    with pytest.raises((ValueError, RuntimeError)):
        builder(model, _config())


# ── 6. μP Adam with all frozen ───────────────────────────────────────

def test_mup_adam_all_frozen_raises_valueerror() -> None:
    """build_mup_adam_param_groups should raise ValueError when all params frozen,
    not silently pass frozen params."""
    from comfy_research.engine.optimizers.mup_init import build_mup_adam_param_groups

    model = nn.Linear(4, 2)
    model.requires_grad_(False)
    with pytest.raises(ValueError, match="no trainable parameters"):
        build_mup_adam_param_groups(
            model,
            base_lr=1e-3,
            embed_mult=1.0,
            hidden_mult=1.0,
            output_mult=1.0,
            weight_decay=0.0,
        )


# ── 7. frozen params do not update during training ───────────────────

def test_frozen_params_do_not_update() -> None:
    """Run one optimizer step; frozen weights stay unchanged, trainable ones move."""
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    # Freeze first Linear
    frozen_layer: nn.Linear = cast(nn.Linear, model[0])
    frozen_layer.requires_grad_(False)

    # Snapshot initial weights
    frozen_weight_before = frozen_layer.weight.clone()
    trainable_layer: nn.Linear = cast(nn.Linear, model[2])
    trainable_weight_before = trainable_layer.weight.clone()

    opt = _build_adam_optimizer(model, _config())
    x = torch.randn(2, 4)
    y = torch.randn(2, 2)
    loss = nn.MSELoss()(model(x), y)
    loss.backward()
    opt.step()

    assert torch.equal(frozen_layer.weight, frozen_weight_before), (
        "Frozen layer weights should not change"
    )
    assert not torch.equal(trainable_layer.weight, trainable_weight_before), (
        "Trainable layer weights should update"
    )


# ── 8. multiple freeze markers together ──────────────────────────────

def test_multiple_freeze_markers_redundant() -> None:
    """When multiple freeze markers are set, the module is still frozen."""
    data = {"freeze": True, "trainable": False, "requiresGrad": False}
    module = build_atomic_layer_module(_make_linear_node(data))
    assert all(not p.requires_grad for p in module.parameters())


# ── 9. activation / norm layers also respect freeze ──────────────────

def test_activation_layer_freeze() -> None:
    node = Node(id="test-act", type=NodeKind.activation_layer, data={"activation": "relu", "freeze": True})
    module = build_atomic_layer_module(node)
    assert all(not p.requires_grad for p in module.parameters())


def test_layer_norm_freeze() -> None:
    node = Node(id="test-ln", type=NodeKind.layer_norm_layer, data={"normalizedShape": 4, "trainable": False})
    module = build_atomic_layer_module(node)
    assert all(not p.requires_grad for p in module.parameters())
