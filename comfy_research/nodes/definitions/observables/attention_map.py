"""observable_attention_map - recorded self-attention maps for tensor visualization.

Each log step stores the selected detached ``[query, key]`` maps in the
bounded ``observable_attention_slice_histories`` transport.  Frames remain
separate from scalar and embedding histories because one log tick can contain
multiple layer/batch/head slices.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from comfy_research.nodes.registry import observable_def, recorder_for
from comfy_research.nodes.schema import FrontendSpec, IntListField, ObservableDef, SpawnSpec, VizSpec

if TYPE_CHECKING:  # pragma: no cover
    from comfy_research.engine.trainer.recorder import ObservableRecorder
    from comfy_research.schemas.graph import Node


ATTENTION_MAP = observable_def(
    ObservableDef(
        type="observable_attention_map",
        label="Attention map",
        hint=(
            "Records every selected layer, batch, and head tuple at each logging step "
            "(at most 20 slices). Retains the newest 50 frames; maps larger than 25 by 25 "
            "are stored as their final 25 by 25 window."
        ),
        viz=VizSpec(
            variant="attention_map",
            title="Attention map",
            info_markdown=(
                "**Attention map** — records the Cartesian product of selected layer, batch, "
                "and head indices at every logging step. At most **20 slices** are recorded "
                "per step, the newest **50 frames** are retained, and maps larger than "
                "**25 x 25** retain their final query/key window."
            ),
            spawns=True,
            spawn=SpawnSpec(kind="attention_map", title="Attention map"),
        ),
        fields=(
            IntListField(
                key="attentionLayerIndices",
                label="Layer indices",
                default=0,
                min=0,
                sweepable=False,
            ),
            IntListField(
                key="attentionBatchIndices",
                label="Batch indices",
                default=0,
                min=0,
                sweepable=False,
            ),
            IntListField(
                key="attentionHeadIndices",
                label="Head indices",
                default=0,
                min=0,
                sweepable=False,
            ),
        ),
        frontend=FrontendSpec(component_key="AttentionMapObservableNode"),
    )
)


@recorder_for(ATTENTION_MAP)
def record(rec: "ObservableRecorder", on: "Node") -> None:
    """Store every configured attention slice in one bounded, self-contained frame."""
    from comfy_research.engine.trainer.attention_map_history import append_frame, crop_map, selected_tuples, validate_tuple
    metric_history = rec.observable_metric_histories[on.id]
    layers = rec._attention_layers_for_log() or []
    if not layers:
        from fastapi import HTTPException
        raise HTTPException(400, f"Attention map node {on.id}: selected model does not expose attention maps.")
    from comfy_research.engine.trainer.observable_metrics import _attention_token_ids_or_none
    slices: list[dict[str, Any]] = []
    means: list[float] = []
    for layer, batch, head in selected_tuples(on.id, on.data or {}):
        attention_map = validate_tuple(on.id, layer, batch, head, layers)
        matrix, source_shape, row_start, col_start = crop_map(attention_map)
        labels = _attention_token_ids_or_none(rec.model, rec._xr, batch)
        token_ids = labels[col_start:] if labels is not None and len(labels) == source_shape[1] == source_shape[0] else None
        slices.append({"layer": layer, "batch": batch, "head": head, "map": matrix, "token_ids": token_ids, "source_shape": source_shape, "row_start": row_start, "col_start": col_start})
        means.append(float(attention_map.mean().detach().cpu()))
    append_frame(rec.observable_attention_slice_histories[on.id], {"step": rec._step, "slices": slices})
    metric_history.append(sum(means) / len(means))
