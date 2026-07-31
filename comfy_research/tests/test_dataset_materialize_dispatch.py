"""Three-way dispatch comparison table for dataset materialization.

The maps below are the authoritative statement
of which build branch handles which dataset family and where the branch
copies GENUINELY differ — compiled by mechanical pairwise diff of the
_materialize_* helpers after the trainer was split into modules, and now
asserted directly against the unified
comfy_research/engine/trainer/dataset_materialize.py. Helpers whose
atomic/vector copies were byte-identical exist once; each genuine divergence
sits behind an explicit ``branch`` parameter and is exercised by named
prepare-golden fixtures (or, where unreachable through the full pipeline,
pinned at source level here).
"""
from __future__ import annotations

import ast
from pathlib import Path

TRAINER_PKG = Path(__file__).resolve().parents[1] / "engine" / "trainer"
UNIFIED_MODULE = "dataset_materialize.py"

# Per-branch dispatch functions inside the unified module (not family
# helpers). The first four are the DATASET_PROVIDERS entries.
BRANCH_DISPATCHERS: tuple[str, ...] = (
    "_materialize_vision_branch",
    "_materialize_for_atomic",
    "_materialize_for_vector",
    "_materialize_for_token",
    "_materialize_dense_families",
)

# family key -> unified helper name, per branch.
VISION_FAMILIES: dict[str, str] = {
    "vision": "_materialize_vision",
}
ATOMIC_FAMILIES: dict[str, str] = {
    "teacher": "_materialize_teacher",
    "mixer": "_materialize_mixer",
    "uniform_linear_motion": "_materialize_paired_split",
    "kepler_2d": "_materialize_paired_split",
    "pde_field": "_materialize_paired_split",
    "memorization_b_dense": "_materialize_memorization_b_dense",
    "memorization_a_dense": "_materialize_memorization_a_dense",
    "vision_flatten_dense": "_materialize_vision_flatten_dense",
    "linear_like": "_materialize_linear_like",
    "symbolic_func": "_materialize_symbolic_func",
}
VECTOR_FAMILIES: dict[str, str] = dict(ATOMIC_FAMILIES)  # same helpers, branch="vector"
TOKEN_FAMILIES: dict[str, str] = {
    "memorization_b_token": "_materialize_memorization_b_token",
    "unigram": "_materialize_unigram",
    "token_prediction": "_materialize_token_prediction",
    "bigram_low_rank": "_materialize_bigram_low_rank",
    "circle_random_walk": "_materialize_circle_random_walk",
    "circular_motion": "_materialize_circular_motion",
    "in_context_associative_recall": "_materialize_in_context_associative_recall",
    "modular_addition": "_materialize_modular_addition",
    "toy_language": "_materialize_toy_language",
    "fallback_circle_walk": "_materialize_fallback_circle_walk",
}

# Helpers that take an explicit branch parameter because the atomic/vector
# copies genuinely diverged; every other shared helper was byte-identical.
# _materialize_memorization_b_dense left this set after the vector
# test-split 400 was fixed; the helper is branch-free again).
BRANCH_PARAMETERIZED_HELPERS: tuple[str, ...] = (
    "_materialize_teacher",
    "_materialize_mixer",
)

# (family, branches) -> what differs, why it must stay, and what pins it.
# "fixtures" entries must exist in the prepare golden; divergences that are
# unreachable through the full prepare pipeline (graph validation intercepts
# with its own message first) use
# "unreachable": True and are pinned by the source-wording assertion instead.
KNOWN_DIVERGENCES: dict[tuple[str, str], dict] = {
    # The memorization_b_dense atomic/vector divergence was fixed: the
    # vector test-split 400 was a historical y-dispatch gap; memB dense now
    # returns the test split on both branches — pinned by
    # test_memb_vector_testsplit_aligns_with_atomic + the
    # err_memb_vector_testsplit golden entry (key name kept; it names the
    # historical bug). NOTE: "full-model MLP regression" stays in
    # DIVERGENT_WORDINGS — the dense chain's terminal else still uses it.
    # The memorization_b_token fallthrough was fixed: memB
    # is a real token family now (its own draws; no fallback overwrite) —
    # pinned by test_memb_token_is_a_real_family + the tok_memb_fallthrough
    # golden entries (key name kept; it names the historical bug).
    ("teacher", "atomic-vs-vector"): {
        "what": (
            "dim-mismatch 400 wording: 'Student chain I/O (m, n) must match the "
            "teacher (i, o).' (atomic) vs 'Student model dimensions (m, n) must "
            "match the teacher (i, o).' (vector). Bodies otherwise identical."
        ),
        "fixtures": ("err_teacher_dim_mismatch_atomic", "err_teacher_dim_mismatch_vec"),
    },
    ("mixer", "atomic-vs-vector"): {
        "what": (
            "dim-mismatch 400 wording: '...must match student chain dimensions "
            "(m, n).' (atomic) vs '...must match student model dimensions (m, n).' "
            "(vector). Bodies otherwise identical."
        ),
        "fixtures": ("err_mixer_dim_mismatch_atomic", "err_mixer_dim_mismatch_vec"),
    },
    ("circle_random_walk", "token-family-vs-fallback"): {
        "what": (
            "the fallback circle-walk sources vocab_ds from the CALLER (dataset "
            "vocabSize default 4) while the family helper reads "
            "scalar(vocabSize, 10); the test-mismatch messages differ "
            "('Test circle random walk dataset...' fallback vs 'Circle random walk "
            "test dataset...' family). Two verbatim near-copies stay. "
            "(memB became a real family) NO token-task dataset type reaches the "
            "fallback through the pipeline — probed empirically: every member of "
            "_TOKEN_CLASSIFICATION_DATASET_TYPES has a family arm. The fallback "
            "side is defense in depth, pinned by DIVERGENT_WORDINGS; the family "
            "side by tok_circle_walk."
        ),
        "fixtures": ("tok_circle_walk",),
        "fallback_unreachable": True,
    },
    ("unsupported_dataset", "atomic-vs-vector"): {
        "what": (
            "the dense family chain's terminal 400 wording: '...not supported for "
            "atomic-chain MLP regression...' vs '...full-model MLP regression...'."
        ),
        "fixtures": (),
        "unreachable": True,
    },
    ("teacher", "vector-diffusion-precheck"): {
        "what": (
            "only the vector branch pre-checks teacher_dataset against diffusion "
            "models ('diffusion_score_model cannot be trained with teacher_dataset.')."
        ),
        "fixtures": (),
        "unreachable": True,
    },
}

# Exact wording pairs the unified source must keep verbatim — covers the
# unreachable divergences that no fixture can pin.
DIVERGENT_WORDINGS: tuple[str, ...] = (
    "Student chain I/O",
    "Student model dimensions",
    "student chain dimensions",
    "student model dimensions",
    "atomic-chain MLP regression",
    "full-model MLP regression",
    "Test circle random walk dataset vocabSize/contextLength must match train.",
    "Circle random walk test dataset vocabSize/contextLength must match train.",
    "diffusion_score_model cannot be trained with teacher_dataset.",
)


def _module_functions(module_file: str) -> dict[str, ast.FunctionDef]:
    src = (TRAINER_PKG / module_file).read_text(encoding="utf-8")
    return {n.name: n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef)}


def test_family_maps_match_the_unified_module_exactly() -> None:
    fns = _module_functions(UNIFIED_MODULE)
    helpers = {n for n in fns if n.startswith("_materialize") and n not in BRANCH_DISPATCHERS}
    mapped = (set(VISION_FAMILIES.values()) | set(ATOMIC_FAMILIES.values())
              | set(VECTOR_FAMILIES.values()) | set(TOKEN_FAMILIES.values()))
    assert mapped == helpers, (
        f"family maps out of sync with {UNIFIED_MODULE}: "
        f"unmapped helpers={sorted(helpers - mapped)}, phantom entries={sorted(mapped - helpers)}"
    )
    for dispatcher in BRANCH_DISPATCHERS + ("materialize_dataset_for_training",):
        assert dispatcher in fns, f"{dispatcher} missing from {UNIFIED_MODULE}"


def test_divergent_helpers_take_an_explicit_branch_parameter() -> None:
    fns = _module_functions(UNIFIED_MODULE)
    for name in BRANCH_PARAMETERIZED_HELPERS:
        kwonly = {a.arg for a in fns[name].args.kwonlyargs}
        assert "branch" in kwonly, f"{name} must take keyword-only 'branch'"
    # and no OTHER family helper grows a branch switch silently
    for name, fn in fns.items():
        if name in BRANCH_PARAMETERIZED_HELPERS or name in BRANCH_DISPATCHERS:
            continue
        if not name.startswith("_materialize"):
            continue
        kwonly = {a.arg for a in fn.args.kwonlyargs}
        assert "branch" not in kwonly, (
            f"{name} gained a 'branch' parameter — add it to BRANCH_PARAMETERIZED_HELPERS "
            f"and KNOWN_DIVERGENCES first"
        )


def test_build_modules_keep_no_materialize_helpers() -> None:
    for module_file in ("prepare_build_vision.py", "prepare_build_atomic.py",
                        "prepare_build_vector.py", "prepare_build_token.py"):
        leftovers = [n for n in _module_functions(module_file) if n.startswith("_materialize")]
        assert not leftovers, f"{module_file} still defines {leftovers} (moved to {UNIFIED_MODULE})"


def test_divergent_wordings_survive_in_unified_source() -> None:
    src = (TRAINER_PKG / UNIFIED_MODULE).read_text(encoding="utf-8")
    for wording in DIVERGENT_WORDINGS:
        assert wording in src, f"divergent wording lost from {UNIFIED_MODULE}: {wording!r}"


def test_every_divergence_is_pinned_by_fixtures() -> None:
    from comfy_research.tests.test_trainer_prepare_golden import _fixtures

    names = set(_fixtures())
    for key, entry in KNOWN_DIVERGENCES.items():
        if entry.get("unreachable"):
            assert not entry["fixtures"], f"{key}: unreachable divergences cannot have fixtures"
            continue
        missing = [f for f in entry["fixtures"] if f not in names]
        assert not missing, f"{key}: fixtures {missing} not in the prepare golden set"
        assert entry["fixtures"], f"{key}: reachable divergence must name at least one fixture"
