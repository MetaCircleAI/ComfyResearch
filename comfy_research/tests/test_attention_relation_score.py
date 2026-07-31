from __future__ import annotations

import math

import pytest
import torch
from fastapi import HTTPException

from comfy_research.engine.trainer.attention_relation_dsl import (
    AttentionRelationDslError,
    compile_attention_relation_predicate,
)
from comfy_research.engine.trainer.attention_relation_metrics import attention_relation_score
from comfy_research.engine.trainer.attention_relation_metrics import attention_relation_pair_label, attention_relation_pairs


def _predicate(source: str, *, required: bool = True):
    return compile_attention_relation_predicate(source, field="predicate", required=required)


def test_predicate_examples_and_negative_position_shorthand() -> None:
    induction = _predicate("pos(k) > 0 and pos(k) < pos(q) and tok(k - 1) == tok(q)")
    assert induction.evaluate(q=3, k=2, token_ids=[7, 8, 9, 8], seq_len=4)
    assert _predicate("pos(q) == -1").evaluate(q=3, k=0, token_ids=None, seq_len=4)
    assert _predicate("-2 <= pos(k)").evaluate(q=0, k=2, token_ids=None, seq_len=4)
    assert _predicate("pos(q) - 1 == 2").evaluate(q=3, k=0, token_ids=None, seq_len=4)
    assert _predicate("", required=False).evaluate(q=0, k=0, token_ids=None, seq_len=1)


@pytest.mark.parametrize("source", ["x", "pos(x)", "foo()", "a[0]", "[x for x in y]", "pos(q) / 2"])
def test_predicate_rejects_unsafe_syntax(source: str) -> None:
    with pytest.raises(AttentionRelationDslError):
        _predicate(source)


def test_attention_relation_score_reductions_and_query_mean() -> None:
    attn = torch.tensor([[[[0.1, 0.2, 0.7], [0.2, 0.3, 0.5], [0.4, 0.4, 0.2]]]])
    all_queries = _predicate("true")
    previous = _predicate("pos(k) == pos(q) - 1")
    value = attention_relation_score([attn], layer_index=0, head_index=0, query_filter=all_queries, key_relation=previous, token_rows=[[1, 2, 3]], key_reduction="mean")
    assert value == pytest.approx((0.2 + 0.4) / 2)
    keys_before = _predicate("pos(k) < pos(q)")
    mean = attention_relation_score([attn], layer_index=0, head_index=0, query_filter=all_queries, key_relation=keys_before, token_rows=[[1, 2, 3]], key_reduction="mean")
    maximum = attention_relation_score([attn], layer_index=0, head_index=0, query_filter=all_queries, key_relation=keys_before, token_rows=[[1, 2, 3]], key_reduction="max")
    total = attention_relation_score([attn], layer_index=0, head_index=0, query_filter=all_queries, key_relation=keys_before, token_rows=[[1, 2, 3]], key_reduction="sum")
    assert mean == pytest.approx((0.2 + 0.4) / 2)
    assert maximum == pytest.approx((0.2 + 0.4) / 2)
    assert total == pytest.approx((0.2 + 0.8) / 2)


def test_attention_relation_score_nan_and_runtime_errors() -> None:
    attn = torch.ones((1, 1, 3, 3)) / 3
    none = attention_relation_score([attn], layer_index=0, head_index=0, query_filter=_predicate("true"), key_relation=_predicate("false"), token_rows=[[1, 2, 3]], key_reduction="mean")
    assert math.isnan(none)
    with pytest.raises(HTTPException, match="layer 1"):
        attention_relation_score([attn], layer_index=1, head_index=0, query_filter=_predicate("true"), key_relation=_predicate("true"), token_rows=[[1, 2, 3]], key_reduction="mean")
    with pytest.raises(HTTPException, match="rank-2"):
        attention_relation_score([attn], layer_index=0, head_index=0, query_filter=_predicate("tok(q) == 1"), key_relation=_predicate("true"), token_rows=[None], key_reduction="mean")


def test_attention_relation_layer_head_choices_are_paired_and_broadcast() -> None:
    assert attention_relation_pairs({"layerIndex": [1, 0], "headIndex": [0, 1]}) == [(1, 0), (0, 1)]
    assert attention_relation_pairs({"layerIndex": 1, "headIndex": [0, 1]}) == [(1, 0), (1, 1)]
    assert attention_relation_pair_label(1, 0) == "layer 1, head 0"
    with pytest.raises(HTTPException, match="equal lengths"):
        attention_relation_pairs({"layerIndex": [0, 1], "headIndex": [0, 1, 2]})
    with pytest.raises(HTTPException, match="at least one"):
        attention_relation_pairs({"layerIndex": [], "headIndex": [0]})


def test_attention_relation_curve_selectors_do_not_become_trainer_sweep_axes() -> None:
    from comfy_research.nodes.definitions.observables.attention_relation_score import ATTENTION_RELATION_SCORE

    fields = {field.key: field for field in ATTENTION_RELATION_SCORE.fields}
    assert fields["layerIndex"].sweepable is False
    assert fields["headIndex"].sweepable is False
