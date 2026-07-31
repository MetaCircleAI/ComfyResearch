"""RNG-call-sequence golden for trainer build stages.

Drives the prepare stages manually, swaps s.rng for a RecordingGenerator
proxy right before materialize_dataset_stage (which with
build_training_model_stage replaced build_training_objects_stage),
and snapshots the exact sequence of shared-rng calls plus torch.manual_seed
events. Structure only —
no sampled values (value equivalence is the materialize/prepare goldens' and
the differential harness's job). Also stores per-fixture summaries
(tensor shape/dtype/sha256, model class, task) so a divergence triages as
RNG-order vs data vs model-path.

Regenerate ONLY on deliberate behavior changes:
COMFYRESEARCH_UPDATE_TRAINER_RNG_SEQUENCE_GOLDEN=1 python -m pytest -q \
    comfy_research/tests/test_trainer_rng_sequence_golden.py
"""
from __future__ import annotations

import hashlib
import json
import os
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from fastapi import HTTPException  # noqa: E402

from comfy_research.engine.trainer.prepare_state import PrepareState  # noqa: E402
from comfy_research.engine.trainer.prepare_graph import (  # noqa: E402
    determine_task_stage,
    resolve_graph_stage,
    validate_compatibility_stage,
)
from comfy_research.engine.trainer.prepare_config import (  # noqa: E402
    init_build_placeholders_stage,
    parse_hyperparams_stage,
)
from comfy_research.engine.trainer.prepare_finalize import (  # noqa: E402
    build_training_model_stage,
    materialize_dataset_stage,
)
from comfy_research.engine.runs.trainer_run import get_user_observable_record  # noqa: E402
from comfy_research.schemas.graph import Edge, Node, NodeKind  # noqa: E402

from comfy_research.tests.test_trainer_prepare_golden import _fixtures, _gr  # noqa: E402

SNAPSHOT_PATH = Path(__file__).parent / "snapshots" / "trainer_rng_sequence_golden.json"
UPDATE_ENV = "COMFYRESEARCH_UPDATE_TRAINER_RNG_SEQUENCE_GOLDEN"

RECORDED_METHODS = ("choice", "integers", "random", "standard_normal",
                    "permutation", "shuffle", "normal", "uniform")


def _summ(v: Any) -> str:
    """Structural summary of an argument/result — never values."""
    if isinstance(v, np.ndarray):
        return f"ndarray{list(v.shape)}:{v.dtype}"
    if isinstance(v, (list, tuple)):
        return f"{type(v).__name__}[{len(v)}]"
    if isinstance(v, (bool, int, float, np.integer, np.floating)):
        return type(v).__name__  # scalar magnitudes are values — omit them
    return type(v).__name__


def _callsite() -> str:
    for frame in reversed(traceback.extract_stack(limit=14)[:-2]):
        if "/comfy_research/engine/" in frame.filename.replace("\\", "/"):
            return f"{Path(frame.filename).name}:{frame.lineno}"
    return "external"


class RecordingGenerator:
    """Proxy around a real np.random.Generator recording call structure."""

    def __init__(self, inner: np.random.Generator, log: list[dict], diag: list[str]):
        self._inner = inner
        self._log = log
        self._diag = diag

    def __getattr__(self, name: str):
        attr = getattr(self._inner, name)
        if name not in RECORDED_METHODS or not callable(attr):
            return attr

        def wrapper(*args: Any, **kwargs: Any):
            result = attr(*args, **kwargs)
            # The snapshot stores ONLY invariants (method + structure): call
            # SITES shift with any line edit and would churn the golden on
            # every refactor. Callsites live in a parallel diagnostic list
            # used solely in failure messages.
            self._log.append({
                "method": name,
                "args": [_summ(a) for a in args],
                "kwargs": {k: _summ(v) for k, v in sorted(kwargs.items())},
                "result": _summ(result),
            })
            self._diag.append(_callsite())
            return result

        return wrapper


def _tensor_summary(t: Any) -> Any:
    if t is None:
        return None
    data = t.detach().cpu().numpy()
    # Float tensors that passed through BLAS (e.g. y = x @ w) are NOT bitwise
    # reproducible across OS/arch (caught by CI on this very PR: x_t matched,
    # y_t didn't). Round floats to 4 decimals before hashing — the same
    # tolerance class the train-smoke golden uses; integer tensors hash exactly.
    if np.issubdtype(data.dtype, np.floating):
        data = np.round(data.astype(np.float64), 4)
    return {"shape": list(t.shape), "dtype": str(t.dtype),
            "sha256": hashlib.sha256(data.tobytes()).hexdigest()[:16]}


def _capture(fixture: dict) -> dict[str, Any]:
    nodes = [Node(id=n["id"], type=NodeKind(n["type"]), data=dict(n.get("data") or {}))
             for n in fixture["nodes"]]
    edges = [Edge(id=e[0], source=e[1], target=e[2], sourceHandle=e[3], targetHandle=e[4])
             for e in fixture["edges"]]
    log: list[dict] = []
    diag: list[str] = []
    seed_log: list[dict] = []
    seed_diag: list[str] = []

    real_manual_seed = torch.manual_seed

    def recording_manual_seed(seed):  # plain module attr — safely patchable
        seed_log.append({"event": "torch.manual_seed", "seed": int(seed)})
        seed_diag.append(_callsite())
        return real_manual_seed(seed)

    torch.manual_seed = recording_manual_seed
    try:
        s = PrepareState.initial(
            nodes, edges, "trainer", resume=None, hessian_oversized_policy=None,
            get_user_observable_record=get_user_observable_record,
        )
        resolve_graph_stage(s)
        determine_task_stage(s)
        validate_compatibility_stage(s)
        parse_hyperparams_stage(s)
        init_build_placeholders_stage(s)
        s.rng = RecordingGenerator(s.rng, log, diag)  # brackets the build-stage RNG calls
        materialize_dataset_stage(s)
        build_training_model_stage(s)
    except HTTPException as e:
        return {"error": {"status": e.status_code, "detail": str(e.detail)}}
    finally:
        torch.manual_seed = real_manual_seed

    return {
        "rng_calls": log,
        "_rng_callsites": diag,        # diagnostic only — stripped before compare/save
        "seed_events": seed_log,
        "_seed_callsites": seed_diag,  # diagnostic only
        "summary": {
            "x_t": _tensor_summary(s.x_t),
            "y_t": _tensor_summary(s.y_t),
            "x_test_t": _tensor_summary(s.x_test_t),
            "y_test_t": _tensor_summary(s.y_test_t),
            "model_class": type(s.model).__name__,
            "trainer_task": str(s.trainer_task),
        },
    }


def _strip_diag(entry: dict) -> dict:
    return {k: v for k, v in entry.items() if not k.startswith("_")}


def test_trainer_rng_sequence_golden() -> None:
    raw = {name: _capture(fx) for name, fx in sorted(_fixtures().items())}
    current = {name: _strip_diag(e) for name, e in raw.items()}
    if os.environ.get(UPDATE_ENV) == "1":
        SNAPSHOT_PATH.write_text(json.dumps(current, indent=1, sort_keys=True) + "\n",
                                 encoding="utf-8")
        pytest.skip(f"Updated snapshot at {SNAPSHOT_PATH}")
    if not SNAPSHOT_PATH.exists():
        pytest.skip(f"bootstrap snapshot with {UPDATE_ENV}=1")
    expected = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
    assert set(expected) == set(current), (
        f"fixture set changed: +{sorted(set(current) - set(expected))} "
        f"-{sorted(set(expected) - set(current))}")
    for name in sorted(expected):
        exp, cur = expected[name], current[name]
        if exp.get("rng_calls") != cur.get("rng_calls"):
            sites = raw[name].get("_rng_callsites", [])
            for i, (a, b) in enumerate(zip(exp.get("rng_calls", []), cur.get("rng_calls", []))):
                site = sites[i] if i < len(sites) else "?"
                assert a == b, (f"{name}: first diverging rng call #{i} (current side at {site}):\n"
                                f" expected {json.dumps(a)}\n current  {json.dumps(b)}")
            raise AssertionError(
                f"{name}: rng call count {len(exp.get('rng_calls', []))} -> "
                f"{len(cur.get('rng_calls', []))}")
        for key in ("seed_events", "summary"):
            assert exp.get(key) == cur.get(key), (
                f"{name}: {key} diverged:\n expected {json.dumps(exp.get(key), sort_keys=True)}\n"
                f" current  {json.dumps(cur.get(key), sort_keys=True)}")
        assert exp == cur, f"{name}: unexpected extra keys diverged"


def test_memb_token_is_a_real_family() -> None:
    """memB routed to the token task uses its own draws, exactly
    the four choice calls (train x/y + test x/y), NO fallback circle-walk
    integers/random, and the tensors are the memB pairs."""
    entry = _capture(_fixtures()["tok_memb_fallthrough"])
    assert "error" not in entry, entry
    methods = [c["method"] for c in entry["rng_calls"]]
    assert methods == ["choice", "choice", "choice", "choice"], methods
    assert entry["summary"]["x_t"]["shape"] == [8, 1], entry["summary"]
    assert entry["summary"]["y_t"]["shape"] == [8], entry["summary"]


def test_memb_vector_testsplit_aligns_with_atomic() -> None:
    """memB on the dense vector path draws train and test data and builds;
    no terminal 400 (the historical vector y-test dispatch gap is fixed; the
    memB dense helper is branch-free again). The graphs are inline, NOT in
    _fixtures(), so this guards behavior independent of the snapshots.
    """
    def _run(test_size: int):
        fx = _gr(
            {"type": "memorization_b_dataset", "data": {"vocabSize": 4, "inputDim": 4, "outputDim": 4, "trainSize": 8, "testSize": test_size, "seed": 0, "samplingMode": "fixed"}},
            {"type": "mlp_model", "data": {"inputDim": 4, "outputDim": 4, "depth": 1, "width": 6, "activation": "relu", "seed": 0}},
            "cross_entropy_loss",
        )
        nodes = [Node(id=n["id"], type=NodeKind(n["type"]), data=dict(n.get("data") or {}))
                 for n in fx["nodes"]]
        edges = [Edge(id=e[0], source=e[1], target=e[2], sourceHandle=e[3], targetHandle=e[4])
                 for e in fx["edges"]]
        log: list[dict] = []
        s = PrepareState.initial(
            nodes, edges, "trainer", resume=None, hessian_oversized_policy=None,
            get_user_observable_record=get_user_observable_record,
        )
        resolve_graph_stage(s)
        determine_task_stage(s)
        validate_compatibility_stage(s)
        parse_hyperparams_stage(s)
        init_build_placeholders_stage(s)
        assert str(s.trainer_task) == "cross_entropy_dense", s.trainer_task  # vector-dense routing
        s.rng = RecordingGenerator(s.rng, log, [])
        err: HTTPException | None = None
        try:
            materialize_dataset_stage(s)
            build_training_model_stage(s)
        except HTTPException as e:
            err = e
        return [c["method"] for c in log], err, s

    # trainonly baseline: two train draws, builds fine.
    methods, err, s = _run(0)
    assert err is None, err
    assert methods == ["choice", "choice"], methods

    # testSize > 0: train AND test draws, then a NORMAL build with the test
    # split materialized (atomic-identical).
    methods, err, s = _run(4)
    assert err is None, err
    assert methods == ["choice", "choice", "choice", "choice"], methods
    # memB dense features are [n, vocab] (vocab=4 here — hence mlp inputDim 4)
    assert list(s.x_test_t.shape) == [4, 4] and list(s.y_test_t.shape) == [4], (
        s.x_test_t.shape, s.y_test_t.shape)


def test_recorded_methods_are_not_missing_known_consumers() -> None:
    """Canary against RECORDED_METHODS gaps: ``uniform`` was missing
    and kepler-driven fixtures showed EMPTY rng_calls despite consuming the
    shared rng). Each entry pins one fixture to a method its dataset family
    is known to call — a future gap surfaces here, not as a silent hole."""
    expectations = {
        "vec_numeric_transformer": "uniform",   # kepler arrays draw via rng.uniform
        "vec_kan": "standard_normal",           # linear-like x draws
        "tok_transformer": "integers",          # token_prediction draws
    }
    fixtures = _fixtures()
    for name, method in expectations.items():
        entry = _capture(fixtures[name])
        methods = {c["method"] for c in entry.get("rng_calls", [])}
        assert method in methods, (
            f"{name}: expected shared-rng method {method!r} in recorded calls, "
            f"got {sorted(methods)} — is RECORDED_METHODS missing something?")
