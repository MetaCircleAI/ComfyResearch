"""Smoke tests for toy language dataset builders (deterministic seeds, tensor shapes)."""

from __future__ import annotations

import numpy as np

from comfy_research.engine.datasets.toy_language_dyck_runtime import build_dyck_lm_arrays_from_seed
from comfy_research.engine.datasets.toy_language_external_runtime import (
    build_cogs_arrays_from_seed,
    build_listops_arrays_from_seed,
    build_scan_arrays_from_seed,
    phi_style_lm_from_seed,
    tiny_stories_lm_from_seed,
)
from comfy_research.engine.datasets.toy_language_formal_runtime import build_formal_suite_lm_arrays_from_seed
from comfy_research.engine.datasets.toy_language_ngram_runtime import build_ngram_lm_arrays_from_seed
from comfy_research.engine.datasets.toy_language_pcfg_runtime import build_pcfg_lm_arrays_from_seed
from comfy_research.engine.datasets.toy_language_inspect import toy_language_word_inspect_lines


def _assert_lm_quad(x_tr, y_tr, x_te, y_te, ctx: int, train_n: int, test_n: int) -> None:
    assert x_tr.shape == (train_n, ctx)
    assert y_tr.shape == (train_n, ctx)
    assert x_te.shape == (test_n, ctx)
    assert y_te.shape == (test_n, ctx)
    assert x_tr.dtype == np.int64


def _max_open_stack_height_dyck_prefix(full_row: np.ndarray, ctx: int, k: int, v: int) -> int:
    """Max unmatched-open depth on the balanced prefix (first ``even_prefix`` tokens), matching runtime id layout."""
    target_len = ctx + 1
    even_prefix = target_len if target_len % 2 == 0 else target_len - 1
    open_ids = np.minimum(np.arange(k, dtype=np.int64) * 2, v - 1)
    close_ids = np.minimum(open_ids + 1, v - 1)
    stack: list[int] = []
    max_h = 0
    for t in full_row[:even_prefix]:
        ti = int(t)
        opened = False
        for bt in range(k):
            if ti == int(open_ids[bt]):
                stack.append(bt)
                opened = True
                max_h = max(max_h, len(stack))
                break
        if opened:
            continue
        closed = False
        for bt in range(k):
            if ti == int(close_ids[bt]):
                assert stack and stack[-1] == bt
                stack.pop()
                closed = True
                break
        assert closed, f"unexpected token {ti} in dyck prefix"
    return max_h


def test_pcfg_dyck_ngram_formal_smoke() -> None:
    ctx = 12
    train_n, test_n = 16, 5
    base = {"contextLength": ctx, "vocabSize": 24, "seed": 7}

    x_tr, y_tr, x_te, y_te = build_pcfg_lm_arrays_from_seed({**base, "pcfgMaxDepth": 6}, train_n, test_n)
    _assert_lm_quad(x_tr, y_tr, x_te, y_te, ctx, train_n, test_n)

    x_tr, y_tr, x_te, y_te = build_pcfg_lm_arrays_from_seed(
        {**base, "pcfgGenMode": "cfg_sentence", "pcfgGrammarId": "world_model", "vocabSize": 32},
        train_n,
        test_n,
    )
    _assert_lm_quad(x_tr, y_tr, x_te, y_te, ctx, train_n, test_n)

    x_tr, y_tr, x_te, y_te = build_dyck_lm_arrays_from_seed({**base, "numBracketTypes": 2}, train_n, test_n)
    _assert_lm_quad(x_tr, y_tr, x_te, y_te, ctx, train_n, test_n)
    # With v=2k and even context, the (L+1)-th token must not collapse to a constant (regression guard).
    x_pad, y_pad, _, _ = build_dyck_lm_arrays_from_seed(
        {"contextLength": 16, "vocabSize": 4, "numBracketTypes": 2, "seed": 42},
        400,
        0,
    )
    assert int(np.unique(y_pad.reshape(-1)).size) > 1

    cap = 3
    x_tr2, y_tr2, x_te2, y_te2 = build_dyck_lm_arrays_from_seed(
        {**base, "numBracketTypes": 2, "maxNestingDepth": cap},
        train_n,
        test_n,
    )
    _assert_lm_quad(x_tr2, y_tr2, x_te2, y_te2, ctx, train_n, test_n)
    k_eff = 2
    v_dyck = 2 * k_eff
    for i in range(train_n):
        full = np.concatenate([x_tr2[i], np.asarray([y_tr2[i, -1]], dtype=np.int64)])
        assert _max_open_stack_height_dyck_prefix(full, ctx, k_eff, v_dyck) <= cap
    for i in range(test_n):
        full = np.concatenate([x_te2[i], np.asarray([y_te2[i, -1]], dtype=np.int64)])
        assert _max_open_stack_height_dyck_prefix(full, ctx, k_eff, v_dyck) <= cap

    x_tr, y_tr, x_te, y_te = build_ngram_lm_arrays_from_seed({**base, "orderN": 3, "dirichletAlpha": 1.0}, train_n, test_n)
    _assert_lm_quad(x_tr, y_tr, x_te, y_te, ctx, train_n, test_n)

    x_tr, y_tr, x_te, y_te = build_formal_suite_lm_arrays_from_seed(
        {**base, "languageType": "anbn", "vocabSize": 8},
        train_n,
        test_n,
    )
    _assert_lm_quad(x_tr, y_tr, x_te, y_te, ctx, train_n, test_n)


def test_word_inspect_scan_and_cogs() -> None:
    ctx = 10
    train_n, test_n = 6, 2
    common = {"contextLength": ctx, "vocabSize": 48, "seed": 11, "dataSource": "synthetic", "trainSize": train_n, "testSize": test_n}
    lines_in, shape_in, _note = toy_language_word_inspect_lines("scan_dataset", common, "train", "input")
    assert shape_in == [train_n, ctx]
    assert len(lines_in) == train_n
    assert any(w in lines_in[0].lower() for w in ("jump", "walk", "look", "run", "turn", "left", "right"))
    lines_c, shape_c, _ = toy_language_word_inspect_lines("cogs_dataset", common, "train", "input")
    assert shape_c == [train_n, ctx]
    assert "the" in lines_c[0] or "a" in lines_c[0]


def test_external_suite_smoke() -> None:
    ctx = 10
    train_n, test_n = 12, 4
    common = {"contextLength": ctx, "vocabSize": 48, "seed": 11, "dataSource": "synthetic"}

    x_tr, y_tr, x_te, y_te = build_scan_arrays_from_seed(common, train_n, test_n)
    _assert_lm_quad(x_tr, y_tr, x_te, y_te, ctx, train_n, test_n)

    x_tr, y_tr, x_te, y_te = build_cogs_arrays_from_seed(common, train_n, test_n)
    _assert_lm_quad(x_tr, y_tr, x_te, y_te, ctx, train_n, test_n)

    x_tr, y_tr, x_te, y_te = build_listops_arrays_from_seed(common, train_n, test_n)
    _assert_lm_quad(x_tr, y_tr, x_te, y_te, ctx, train_n, test_n)

    corp = {
        **common,
        "vocabCap": 64,
        "tokenizerMode": "char",
        "seqLen": 32,
        "stride": 16,
    }
    x_tr, y_tr, x_te, y_te = tiny_stories_lm_from_seed(corp, train_n, test_n)
    _assert_lm_quad(x_tr, y_tr, x_te, y_te, ctx, train_n, test_n)

    x_tr, y_tr, x_te, y_te = phi_style_lm_from_seed({**corp, "domainMix": "mixed"}, train_n, test_n)
    _assert_lm_quad(x_tr, y_tr, x_te, y_te, ctx, train_n, test_n)


def test_deterministic_repeat() -> None:
    data = {"contextLength": 8, "vocabSize": 16, "seed": 99, "languageType": "palindrome"}
    a = build_formal_suite_lm_arrays_from_seed(data, 20, 5)
    b = build_formal_suite_lm_arrays_from_seed(data, 20, 5)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
