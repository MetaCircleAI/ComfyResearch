from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest
from fastapi import HTTPException

from comfy_research.engine.datasets import synthetic_dataset_builders as sdb

SNAPSHOT_PATH = Path(__file__).with_name("snapshots") / "synthetic_dataset_builders_golden.json"
UPDATE_ENV = "COMFYRESEARCH_UPDATE_SYNTHETIC_DATASET_BUILDERS_GOLDEN"


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


# --- fixtures: name -> zero-arg callable returning the builder's raw result ---
def _fixtures() -> dict[str, Callable[[], Any]]:
    x_lin = np.array([[1.0, -1.0], [-1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]], dtype=np.float32)
    x_lin3 = np.array(
        [[0.5, -0.5, 1.0], [1.0, 0.0, -1.0], [-1.0, 1.0, 0.5], [0.25, 0.75, -0.25]],
        dtype=np.float32,
    )
    return {
        "memorization_labels": lambda: sdb._sample_memorization_labels(
            _rng(0), 6, 4, "power_law_class_probs", 1.0
        ),
        "memorization_b_xy": lambda: sdb._memorization_b_xy_numpy(
            _rng(1), 5, 4, "uniform_class_probs", 0.0
        ),
        "memorization_class_probs": lambda: sdb._memorization_class_probs(
            4, "exponential_class_probs", 0.7
        ),
        "unigram_token_probs": lambda: sdb._unigram_token_probs(
            5, "power_law_class_probs", 1.0
        ),
        "memorization_b_vocab_size": lambda: np.array(
            [sdb._memorization_b_vocab_size({"vocabSize": 6})], dtype=np.int64
        ),
        "bigram_train_test": lambda: sdb._build_bigram_low_rank_arrays(
            {"vocabSize": 8, "rank": 3, "seed": 0, "initSeed": 0, "alpha": 1.0},
            {"vocabSize": 8, "rank": 3, "seed": 0, "initSeed": 0, "alpha": 1.0},
            4,
            2,
        ),
        "circular_train_test": lambda: sdb._build_circular_motion_arrays(
            {"vocabSize": 16, "contextLength": 3, "seed": 0},
            {"vocabSize": 16, "contextLength": 3, "seed": 0},
            4,
            2,
        ),
        "uniform_motion_train_test": lambda: sdb._build_uniform_linear_motion_arrays(
            _rng(0), {"positionDim": 2, "contextLength": 3}, {"positionDim": 2, "contextLength": 3}, 4, 2
        ),
        "uniform_motion_no_test": lambda: sdb._build_uniform_linear_motion_arrays(
            _rng(0), {"positionDim": 2, "contextLength": 3}, {}, 4, 0
        ),
        "kepler_train_test": lambda: sdb._build_kepler_2d_arrays(
            _rng(0), {"contextLength": 4}, {"contextLength": 4}, 4, 2
        ),
        "kepler_no_test": lambda: sdb._build_kepler_2d_arrays(
            _rng(0), {"contextLength": 4}, {}, 4, 0
        ),
        "linear_xor2d": lambda: sdb._linear_like_targets_from_x(
            {"targetKind": "xor2d", "seed": 0}, x_lin, 2, 1, sigma=0.0, additive=False, rng=_rng(0)
        ),
        "linear_plain_shared_weight_seed": lambda: sdb._linear_like_targets_from_x(
            {"targetKind": "linear", "seed": 7}, x_lin3, 3, 2, sigma=0.1, additive=True, rng=_rng(0)
        ),
    }


# --- error cases: name -> (callable, expected status_code, expected detail) ---
def _error_cases() -> dict[str, tuple[Callable[[], Any], int, str]]:
    return {
        "bigram_bad_decay_type": (
            lambda: sdb._bigram_spectrum_decay_type({"decayType": "bad"}),
            400,
            "Bigram low-rank dataset decayType must be 'power_law' or 'exponential'.",
        ),
        "linear_weights_zero_output_dim": (
            lambda: sdb._random_linear_weights(_rng(0), 2, 0),
            400,
            "Output dimension must be >= 1",
        ),
    }


def _describe(obj: Any) -> Any:
    """Serialize a builder result element for snapshotting."""
    if obj is None:
        return {"kind": "none"}
    if isinstance(obj, np.ndarray):
        return {"kind": "ndarray", "dtype": str(obj.dtype), "shape": list(obj.shape), "data": obj.tolist()}
    if isinstance(obj, (list, tuple)):
        return {"kind": "seq", "items": [_describe(x) for x in obj]}
    if isinstance(obj, (int, np.integer)):
        return {"kind": "int", "value": int(obj)}
    if isinstance(obj, (float, np.floating)):
        return {"kind": "float", "value": float(obj)}
    raise TypeError(f"Unsnapshottable type: {type(obj)!r}")


def _build_snapshot() -> dict[str, Any]:
    return {name: _describe(fn()) for name, fn in _fixtures().items()}


def _compare(expected: Any, actual: Any, path: str) -> None:
    assert expected["kind"] == actual["kind"], f"{path}: kind {actual['kind']} != {expected['kind']}"
    kind = expected["kind"]
    if kind == "none":
        return
    if kind == "int":
        assert actual["value"] == expected["value"], f"{path}: int {actual['value']} != {expected['value']}"
        return
    if kind == "float":
        np.testing.assert_allclose(actual["value"], expected["value"], rtol=1e-12, atol=1e-12, err_msg=path)
        return
    if kind == "seq":
        assert len(actual["items"]) == len(expected["items"]), f"{path}: length mismatch"
        for i, (e, a) in enumerate(zip(expected["items"], actual["items"])):
            _compare(e, a, f"{path}[{i}]")
        return
    if kind == "ndarray":
        assert actual["shape"] == expected["shape"], f"{path}: shape {actual['shape']} != {expected['shape']}"
        assert actual["dtype"] == expected["dtype"], f"{path}: dtype {actual['dtype']} != {expected['dtype']}"
        np.testing.assert_allclose(
            np.array(actual["data"]), np.array(expected["data"]), rtol=1e-12, atol=1e-12, err_msg=path
        )
        return
    raise AssertionError(f"{path}: unknown kind {kind!r}")


def test_synthetic_dataset_builders_golden() -> None:
    current = _build_snapshot()
    if os.environ.get(UPDATE_ENV) == "1":
        SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
        SNAPSHOT_PATH.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        pytest.skip(f"Updated snapshot at {SNAPSHOT_PATH}")
    if not SNAPSHOT_PATH.exists():
        pytest.skip(f"bootstrap snapshot with {UPDATE_ENV}=1")
    expected = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
    assert set(expected) == set(current), "fixture set changed; re-bootstrap the snapshot"
    for name in expected:
        _compare(expected[name], current[name], name)


@pytest.mark.parametrize("name", list(_error_cases()))
def test_synthetic_dataset_builders_error_semantics(name: str) -> None:
    fn, status, detail = _error_cases()[name]
    with pytest.raises(HTTPException) as exc:
        fn()
    assert exc.value.status_code == status
    assert exc.value.detail == detail


ALL_BUILDER_NAMES = [
    "_apply_linear_map",
    "_bigram_spectrum_alpha",
    "_bigram_spectrum_decay_type",
    "_bigram_spectrum_lamb",
    "_build_bigram_low_rank_arrays",
    "_build_circular_motion_arrays",
    "_build_kepler_2d_arrays",
    "_build_uniform_linear_motion_arrays",
    "_linear_like_targets_from_x",
    "_linear_regression_weight_rng",
    "_memorization_b_vocab_size",
    "_memorization_b_xy_numpy",
    "_memorization_class_probs",
    "_random_linear_weights",
    "_sample_memorization_labels",
    "_unigram_token_probs",
]


def test_trainer_run_reexports_all_builders() -> None:
    # Other modules (api/dataset_tensor, kan_visualize, activation_collect, ...)
    # import these from trainer_run, so it MUST remain a complete re-export surface.
    # Importing trainer_run pulls in torch; skip where torch is unavailable.
    import importlib

    pytest.importorskip("torch")
    trainer_run = importlib.import_module("comfy_research.engine.runs.trainer_run")
    missing = [name for name in ALL_BUILDER_NAMES if not hasattr(trainer_run, name)]
    assert not missing, f"trainer_run no longer re-exports: {missing}"


def test_module_does_not_import_trainer_run() -> None:
    import ast

    tree = ast.parse(Path(sdb.__file__).read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    offenders = [m for m in imported if "trainer_run" in m]
    assert not offenders, f"synthetic_dataset_builders must not import trainer_run; found {offenders}"
