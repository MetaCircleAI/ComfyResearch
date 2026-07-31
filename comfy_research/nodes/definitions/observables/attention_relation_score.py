"""User-defined scalar attention relation score."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from comfy_research.nodes.registry import observable_def, recorder_for
from comfy_research.nodes.schema import EnumField, FrontendSpec, IntListField, ObservableDef, SpawnSpec, StringField, VizSpec

if TYPE_CHECKING:  # pragma: no cover
    from comfy_research.engine.trainer.recorder import ObservableRecorder
    from comfy_research.schemas.graph import Node


ATTENTION_RELATION_SCORE = observable_def(ObservableDef(
    type="observable_attention_relation_score", label="Attention relation score",
    hint="Scores selected full-map attention cells using safe query/key position and token predicates.",
    viz=VizSpec(variant="user", title="Attention relation score", info_markdown="**Attention relation score** — mean attention to keys selected by a restricted relation predicate.", spawns=True, user_whitelisted=True, spawn=SpawnSpec(kind="user_scalar", unit="attention score")),
    fields=(
        StringField(key="keyRelation", label="Key relation", default="pos(k) == pos(q) - 1"),
        StringField(key="queryFilter", label="Query filter", default=""),
        EnumField(key="keyReduction", label="Key reduction", default="mean", options=("mean", "max", "sum")),
        # These lists select curves inside one training run; they must not expand
        # into the trainer's Cartesian hyperparameter sweep.
        IntListField(key="layerIndex", label="Layer", default=0, min=0, sweepable=False),
        IntListField(key="headIndex", label="Head", default=0, min=0, sweepable=False),
    ),
    frontend=FrontendSpec(component_key="AttentionRelationScoreObservableNode"),
))


@recorder_for(ATTENTION_RELATION_SCORE)
def record(rec: "ObservableRecorder", on: "Node") -> None:
    from comfy_research.engine.trainer.attention_relation_dsl import compile_attention_relation_predicate
    from comfy_research.engine.trainer.attention_relation_metrics import (
        attention_relation_pair_key,
        attention_relation_pairs,
        attention_relation_score,
    )
    from comfy_research.engine.trainer.observable_metrics import _attention_token_ids_or_none

    data: dict[str, Any] = on.data or {}
    query_filter = compile_attention_relation_predicate(data.get("queryFilter"), field="Query filter", required=False)
    key_relation = compile_attention_relation_predicate(data.get("keyRelation"), field="Key relation", required=True)
    token_rows = [_attention_token_ids_or_none(rec.model, rec._xr, batch) for batch in range(int(rec._xr.shape[0]))]
    for index, (layer_index, head_index) in enumerate(attention_relation_pairs(data)):
        value = attention_relation_score(rec._attention_layers_for_log(), layer_index=layer_index, head_index=head_index, query_filter=query_filter, key_relation=key_relation, token_rows=token_rows, key_reduction=str(data.get("keyReduction") or "mean"))
        rec.observable_metric_histories.setdefault(attention_relation_pair_key(on.id, index), []).append(value)
        if index == 0:
            rec.observable_metric_histories[on.id].append(value)
