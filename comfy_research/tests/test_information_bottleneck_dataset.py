from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pytest

from comfy_research.engine.datasets.information_bottleneck_dataset import (
    VAR_U_PATH,
    VAR_U_SHA256,
    build_information_bottleneck_arrays,
    load_var_u,
)
from comfy_research.engine.reproductions.information_bottleneck import (
    _sample_training_subsets,
)
from comfy_research.nodes.definitions.datasets.information_bottleneck import DEF, materialize
from comfy_research.nodes.provider_types import DatasetMaterializeContext
from comfy_research.nodes.registry import dataset_defs_materializers, load_definitions


def test_var_u_asset_is_hash_checked_and_has_expected_structure() -> None:
    x, y = load_var_u()
    assert VAR_U_PATH.is_file()
    assert VAR_U_SHA256 == "0ba9551878a855396a8de0cbaae620788ede1e7b1eb2b8373bbe017d5ea02036"
    assert x.shape == (4096, 12)
    assert y.shape == (4096,)
    assert set(np.unique(x)) == {0.0, 1.0}
    assert set(np.unique(y)) == {0, 1}


def test_hash_mismatch_rejects_replaced_asset(tmp_path) -> None:
    bad = tmp_path / "var_u.mat"
    bad.write_bytes(VAR_U_PATH.read_bytes() + b"replaced")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_var_u(bad)


def test_dataset_node_is_registered_and_materializes_direct_arrays() -> None:
    load_definitions()
    assert DEF.type in dataset_defs_materializers()
    arrays = materialize(DatasetMaterializeContext(
        ds_type=DEF.type,
        rng=np.random.default_rng(5),
        dd_train={}, dd_test={}, train_size=16, test_size=32,
        input_dim=12, output_dim=2, ds_test_raw=None,
    ))
    assert arrays.x_np.shape == (16, 12)
    assert arrays.y_np.shape == (16,)
    assert arrays.x_test_np is not None and arrays.x_test_np.shape == (32, 12)


def test_graph_subset_draw_matches_single_reference_repeat_exactly() -> None:
    seed = 1708
    train_size = 205
    x_all, y_all = load_var_u()
    reference_indices = _sample_training_subsets(
        np.random.default_rng(seed),
        repeats=1,
        total=len(x_all),
        train_size=train_size,
    )[0]
    x_train, y_train, _x_test, _y_test = build_information_bottleneck_arrays(
        np.random.default_rng(seed),
        train_size=train_size,
        test_size=len(x_all),
    )
    assert np.array_equal(x_train, x_all[reference_indices])
    assert np.array_equal(y_train, y_all[reference_indices])


def _tiny_cpu_run() -> tuple[OrderedDict[str, object], object]:
    torch = pytest.importorskip("torch")
    x_np, y_np, x_test_np, y_test_np = build_information_bottleneck_arrays(
        np.random.default_rng(41), train_size=32, test_size=64,
    )
    assert x_test_np is not None and y_test_np is not None
    torch.manual_seed(19)
    model = torch.nn.Sequential(torch.nn.Linear(12, 5), torch.nn.Tanh(), torch.nn.Linear(5, 2))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    x = torch.from_numpy(x_np)
    y = torch.from_numpy(y_np)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        torch.nn.functional.cross_entropy(model(x), y).backward()
        optimizer.step()
    with torch.no_grad():
        logits = model(torch.from_numpy(x_test_np)).detach().clone()
    return OrderedDict((name, value.detach().clone()) for name, value in model.state_dict().items()), logits


def test_small_cpu_double_run_is_deterministic() -> None:
    state_a, logits_a = _tiny_cpu_run()
    state_b, logits_b = _tiny_cpu_run()
    assert logits_a.equal(logits_b)
    assert state_a.keys() == state_b.keys()
    for name in state_a:
        assert state_a[name].equal(state_b[name])
