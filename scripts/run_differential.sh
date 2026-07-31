#!/usr/bin/env bash
# Two-branch differential gate for behavior-preserving trainer refactors.
#
#   bash scripts/run_differential.sh [--suite smoke|observable|full] \
#       [--expect-changed k1,k2] [BASE_REF]
#
# Without --expect-changed the gate is the historical IDENTICAL-or-fail.
# With it, the comparator prints changed/expected/unexpected/
# stale_expectation lists and passes ONLY when the changed trace keys equal
# the named set exactly (behavior-fix PRs paste this output).
#
# Runs scripts/differential_harness.py in (a) a fresh worktree at BASE_REF
# (default origin/main) and (b) the current tree, then diffs the traces'
# results+failures. Hard-fails if both sides resolve to the same commit
# (self-comparison) or if either side has tracked modifications. On failure
# or divergence the worktree and both traces are KEPT and their paths
# printed. Only ever removes worktrees it created under .diff-worktrees/.
set -euo pipefail

SUITE="smoke"
EXPECT_SET=0
EXPECT=""
while [[ "${1:-}" == --* ]]; do
  case "$1" in
    --suite) SUITE="$2"; shift 2 ;;
    --expect-changed) EXPECT="$2"; EXPECT_SET=1; shift 2 ;;
    *) echo "unknown flag $1" >&2; exit 2 ;;
  esac
done
BASE_REF="${1:-origin/main}"

ROOT=$(git rev-parse --show-toplevel)
cd "$ROOT"
DIFF_DIR="$ROOT/.diff-worktrees"
mkdir -p "$DIFF_DIR"
# mktemp guarantees a unique id even for parallel/same-second invocations
# (a second-resolution timestamp collided under parallel runs).
WT=$(mktemp -d "$DIFF_DIR/base-XXXXXX")
RUN_ID=$(basename "$WT")
OUT_BASE="$DIFF_DIR/trace-$RUN_ID-base.json"
OUT_CUR="$DIFF_DIR/trace-$RUN_ID-current.json"

git fetch -q origin 2>/dev/null || true
git worktree add -q "$WT" "$BASE_REF"

echo "[diff] base worktree: $WT ($BASE_REF)"
# Both sides run THIS tree's harness (one measuring instrument); the engine
# under test comes from PYTHONPATH/cwd. This also lets the base side predate
# the harness itself.
(cd "$WT" && PYTHONPATH="$WT" python "$ROOT/scripts/differential_harness.py" --suite "$SUITE" "$OUT_BASE")
PYTHONPATH="$ROOT" python scripts/differential_harness.py --suite "$SUITE" "$OUT_CUR"

STATUS=0
if [[ $EXPECT_SET -eq 1 ]]; then
  python scripts/compare_differential_traces.py "$OUT_BASE" "$OUT_CUR" "$EXPECT" || STATUS=$?
else
  python scripts/compare_differential_traces.py "$OUT_BASE" "$OUT_CUR" || STATUS=$?
fi

if [[ $STATUS -eq 0 ]]; then
  git worktree remove -f "$WT"
  rm -f "$OUT_BASE" "$OUT_CUR"
  echo "[diff] PASS - cleaned up"
else
  echo "[diff] KEPT for debugging:"
  echo "  worktree: $WT"
  echo "  traces:   $OUT_BASE"
  echo "            $OUT_CUR"
fi
exit $STATUS
