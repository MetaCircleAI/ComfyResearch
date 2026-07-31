"""Unit tests for the expected-changes differential comparator."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from compare_differential_traces import compare_traces  # noqa: E402


def _trace(results: dict, failures: dict | None = None) -> dict:
    return {"meta": {"git_head": "x", "tracked_dirty": False},
            "results": results, "failures": failures or {}}


def test_identical_traces_report_ok_and_empty_lists() -> None:
    base = _trace({"a": 1, "b": 2}, {"c": "E: boom"})
    rep = compare_traces(base, _trace({"a": 1, "b": 2}, {"c": "E: boom"}), set())
    assert rep["ok"] and rep["changed"] == [] and rep["unexpected"] == []
    assert rep["stale_expectation"] == [] and rep["unchanged"] == ["a", "b", "c"]


def test_named_change_is_expected_everything_else_byte_identical() -> None:
    base = _trace({"a": 1, "b": 2})
    cur = _trace({"a": 1, "b": 3})
    rep = compare_traces(base, cur, {"b"})
    assert rep["ok"] and rep["changed"] == ["b"] and rep["unchanged"] == ["a"]


def test_unexpected_change_fails() -> None:
    rep = compare_traces(_trace({"a": 1, "b": 2}), _trace({"a": 9, "b": 3}), {"b"})
    assert not rep["ok"] and rep["unexpected"] == ["a"]


def test_stale_expectation_fails() -> None:
    rep = compare_traces(_trace({"a": 1}), _trace({"a": 1}), {"a"})
    assert not rep["ok"] and rep["stale_expectation"] == ["a"]


def test_result_to_failure_flip_counts_as_changed() -> None:
    rep = compare_traces(_trace({"a": 1}), _trace({}, {"a": "E: x"}), {"a"})
    assert rep["ok"] and rep["changed"] == ["a"]


def test_key_present_on_one_side_counts_as_changed() -> None:
    rep = compare_traces(_trace({"a": 1}), _trace({"a": 1, "b": 2}), set())
    assert not rep["ok"] and rep["unexpected"] == ["b"]
