"""Expected-changes comparator for differential traces.

Plain mode (no expectation) reproduces the historical gate: IDENTICAL or
DIVERGED. Expectation mode prints four lists and passes ONLY when the
changed trace keys equal the named set exactly:
    changed            keys whose (result|failure) entry differs
    expected           the named keys
    unexpected         changed - expected      -> FAIL if non-empty
    stale_expectation  expected - changed      -> FAIL if non-empty
Unchanged keys are byte-identical by construction of `changed`.
"""
from __future__ import annotations

import json
import sys


def _entry(trace: dict, name: str) -> tuple:
    if name in trace["results"]:
        return ("result", trace["results"][name])
    if name in trace["failures"]:
        return ("failure", trace["failures"][name])
    return ("absent", None)


def compare_traces(base: dict, cur: dict, expected: set[str]) -> dict:
    names = (set(base["results"]) | set(base["failures"])
             | set(cur["results"]) | set(cur["failures"]))
    changed = sorted(n for n in names if _entry(base, n) != _entry(cur, n))
    unexpected = sorted(set(changed) - expected)
    stale = sorted(expected - set(changed))
    return {
        "changed": changed,
        "expected": sorted(expected),
        "unexpected": unexpected,
        "stale_expectation": stale,
        "unchanged": sorted(names - set(changed)),
        "ok": not unexpected and not stale,
    }


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print("usage: compare_differential_traces.py BASE.json CUR.json [EXPECT_CSV]",
              file=sys.stderr)
        return 2
    base, cur = (json.load(open(p, encoding="utf-8")) for p in argv[1:3])
    if base["meta"]["git_head"] == cur["meta"]["git_head"]:
        sys.exit("[diff] FATAL: both sides are the same commit - self-comparison (checkout failed?)")
    for side, t in (("base", base), ("current", cur)):
        if t["meta"]["tracked_dirty"]:
            sys.exit(f"[diff] FATAL: {side} tree has tracked modifications - commit or stash first")
    expectation_mode = len(argv) > 3
    expected = {k for k in (argv[3].split(",") if expectation_mode else []) if k}
    rep = compare_traces(base, cur, expected)
    head = (f"[diff] base={base['meta']['git_head']} current={cur['meta']['git_head']} "
            f"suite={cur['meta'].get('suite', '?')}")
    if not expectation_mode:
        print(f"{head} -> {'IDENTICAL' if not rep['changed'] else 'DIVERGED'}")
        return 0 if not rep["changed"] else 1
    print(f"{head} -> expect-changed mode")
    for key in ("changed", "expected", "unexpected", "stale_expectation"):
        print(f"[diff] {key}: {rep[key]}")
    print(f"[diff] unchanged: {len(rep['unchanged'])} keys byte-identical")
    print(f"[diff] {'PASS' if rep['ok'] else 'FAIL'} (expected-changes contract)")
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
