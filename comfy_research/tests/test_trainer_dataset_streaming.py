"""Structural guards for the three-layer streaming rematerialization boundary."""

import ast
from pathlib import Path

_ENGINE = Path(__file__).resolve().parents[1] / "engine"
_STREAMING = _ENGINE / "datasets" / "trainer_dataset_streaming.py"
_TRAINER_RUN = _ENGINE / "runs" / "trainer_run.py"


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    mods: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            mods.add(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                mods.add(alias.name)
    return mods


def _defined_functions(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    return {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}


def test_trainer_dataset_streaming_does_not_import_trainer_run():
    assert not any(
        "trainer_run" in m for m in _imported_modules(_STREAMING)
    ), "trainer_dataset_streaming must not import trainer_run (one-way edge)."


def test_trainer_run_no_longer_defines_streaming_train_step_seed():
    assert "_streaming_train_step_seed" not in _defined_functions(_TRAINER_RUN), (
        "streaming_train_step_seed must live only in streaming_seed.py (single source)."
    )


def test_streaming_module_has_no_hardcoded_device():
    src = _STREAMING.read_text()
    for needle in ('torch.device("cpu")', "torch.device('cpu')", ".cpu()"):
        assert needle not in src, (
            f"trainer_dataset_streaming must not hardcode device ({needle!r}); "
            "device flows only through the injected `device` param."
        )


def test_streaming_module_exposes_entry_point():
    assert "rematerialize_train_tensors_for_step" in _defined_functions(_STREAMING)
