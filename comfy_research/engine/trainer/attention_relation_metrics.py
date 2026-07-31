"""Full attention-map metric for ``observable_attention_relation_score``."""
from __future__ import annotations

import math
from typing import Any

import torch
from fastapi import HTTPException

from comfy_research.engine.trainer.attention_relation_dsl import AttentionRelationPredicate

MAX_SELECTED_PAIRS = 4096


def attention_relation_score(
    attention_layers: list[torch.Tensor] | None,
    *,
    layer_index: int,
    head_index: int,
    query_filter: AttentionRelationPredicate,
    key_relation: AttentionRelationPredicate,
    token_rows: list[list[int] | None],
    key_reduction: str,
) -> float:
    if key_reduction not in {"mean", "max", "sum"}:
        raise HTTPException(400, "Attention relation score: key reduction must be mean, max, or sum.")
    if attention_layers is None:
        return float("nan")
    if not 0 <= layer_index < len(attention_layers):
        raise HTTPException(400, f"Attention relation score: layer {layer_index} is out of range [0, {len(attention_layers) - 1}].")
    attention = attention_layers[layer_index]
    if attention.dim() != 4:
        raise HTTPException(400, "Attention relation score: selected model returned an invalid attention-map shape.")
    batch_count, head_count, query_length, key_length = map(int, attention.shape)
    if not 0 <= head_index < head_count:
        raise HTTPException(400, f"Attention relation score: head {head_index} is out of range [0, {head_count - 1}].")
    if query_length != key_length:
        raise HTTPException(400, "Attention relation score requires square self-attention maps.")
    if (query_filter.uses_tokens or key_relation.uses_tokens) and any(row is None for row in token_rows[:batch_count]):
        raise HTTPException(400, "Attention relation score: tok() predicates are supported only for rank-2 single-token model inputs.")
    batch_scores: list[float] = []
    for batch_index in range(batch_count):
        token_ids = token_rows[batch_index] if batch_index < len(token_rows) else None
        query_scores: list[float] = []
        selected_count = 0
        for q in range(query_length):
            if not query_filter.evaluate(q=q, k=0, token_ids=token_ids, seq_len=query_length):
                continue
            keys = [k for k in range(key_length) if key_relation.evaluate(q=q, k=k, token_ids=token_ids, seq_len=query_length)]
            selected_count += len(keys)
            if selected_count > MAX_SELECTED_PAIRS:
                raise HTTPException(400, f"Attention relation score: batch {batch_index} selects {selected_count} pairs; limit is {MAX_SELECTED_PAIRS}.")
            if not keys:
                continue
            values = attention[batch_index, head_index, q, keys]
            if key_reduction == "max": query_scores.append(float(values.max().item()))
            elif key_reduction == "sum": query_scores.append(float(values.sum().item()))
            else: query_scores.append(float(values.mean().item()))
        if query_scores:
            batch_scores.append(sum(query_scores) / len(query_scores))
    return sum(batch_scores) / len(batch_scores) if batch_scores else math.nan


def attention_relation_pairs(data: dict[str, Any]) -> list[tuple[int, int]]:
    """Parse layer/head choices, broadcasting a scalar side when necessary."""
    def choices(value: object, field: str) -> list[int]:
        raw = value if isinstance(value, list) else [value]
        out: list[int] = []
        for item in raw:
            if isinstance(item, bool):
                raise HTTPException(400, f"Attention relation score: {field} must contain non-negative integers.")
            try:
                number = int(item)
                numeric = float(item)
            except (TypeError, ValueError) as exc:
                raise HTTPException(400, f"Attention relation score: {field} must contain non-negative integers.") from exc
            if number < 0 or numeric != number:
                raise HTTPException(400, f"Attention relation score: {field} must contain non-negative integers.")
            out.append(number)
        if not out:
            raise HTTPException(400, f"Attention relation score: {field} must contain at least one index.")
        return out

    layers = choices(data.get("layerIndex", 0), "Layer")
    heads = choices(data.get("headIndex", 0), "Head")
    if len(layers) == len(heads):
        return list(zip(layers, heads, strict=True))
    if len(layers) == 1:
        return [(layers[0], head) for head in heads]
    if len(heads) == 1:
        return [(layer, heads[0]) for layer in layers]
    raise HTTPException(400, "Attention relation score: Layer and Head lists must have equal lengths, unless one side has one value to broadcast.")


def attention_relation_pair_key(node_id: str, index: int) -> str:
    return f"{node_id}::attention_pair::{index}"


def attention_relation_pair_label(layer: int, head: int) -> str:
    return f"layer {layer}, head {head}"
