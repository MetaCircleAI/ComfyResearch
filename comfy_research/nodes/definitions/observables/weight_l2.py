"""observable_weight_l2 — NodeDef-channel definition + recorder.

The custom ``WeightL2ObservableNode`` component provides the aggregation
multiselect and perTopLevel mirror through GENERATED_COMPONENT_ADAPTERS.
CODEGEN_ADAPTERS owns the code-generation body; this definition carries its key.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from comfy_research.nodes.registry import observable_def, recorder_for
from comfy_research.nodes.schema import EnumField, FrontendSpec, ObservableDef, SpawnSpec, VizSpec

if TYPE_CHECKING:  # pragma: no cover
    from comfy_research.engine.trainer.recorder import ObservableRecorder
    from comfy_research.schemas.graph import Node

WEIGHT_L2 = observable_def(
    ObservableDef(
        type="observable_weight_l2",
        label="Weight L2",
        hint="Log L2 norm of weights each step: global, per top-level name segment, or per-parameter tensor (capped).",
        viz=VizSpec(
            variant="weight_l2",
            title="Weight L2",
            info_markdown=(
                "**Weight L2** — Euclidean norm of weights each log step: **global** (entire model), "
                "**top-level module** (first segment of each parameter name), or **per-parameter tensor** "
                "(one curve per weight tensor, capped)."
            ),
            spawns=True,
            spawn=SpawnSpec(kind="hessian_topk", fixed_top_k=1),
        ),
        fields=(EnumField(key="normAggregation", label="Norm Aggregation", default="global"),),
        frontend=FrontendSpec(component_key="WeightL2ObservableNode", codegen_key="observable_weight_l2"),
    )
)


@recorder_for(WEIGHT_L2)
def record(rec: "ObservableRecorder", on: "Node") -> None:
    """Body moved verbatim from ObservableRecorder._record_weight_l2."""
    from comfy_research.engine.trainer.observable_config import (
        _observable_l2_aggregation,
        _sanitize_observable_hist_segment,
    )
    from comfy_research.engine.trainer.tensor_norms import (
        MAX_OBSERVABLE_L2_TENSOR_SERIES,
        _parameter_l2_norms_named,
        _weight_l2_norm,
        _weight_l2_norm_per_top_level,
    )

    model = rec.model
    observable_metric_histories = rec.observable_metric_histories
    weight_l2_canonical_tensor = rec.weight_l2_canonical_tensor
    weight_l2_canonical_top = rec.weight_l2_canonical_top
    weight_l2_tensor_seg_to_raw = rec.weight_l2_tensor_seg_to_raw
    weight_l2_top_seg_to_raw = rec.weight_l2_top_seg_to_raw
    od_w2: dict[str, Any] = on.data or {}
    agg_w = _observable_l2_aggregation(od_w2)
    observable_metric_histories[on.id].append(_weight_l2_norm(model))
    if agg_w == "top_level_module":
        wmap = _weight_l2_norm_per_top_level(model)
        wseg = weight_l2_top_seg_to_raw.setdefault(on.id, {})
        for raw_top in wmap:
            wseg[_sanitize_observable_hist_segment(raw_top)] = raw_top
        wcanon = weight_l2_canonical_top.setdefault(on.id, [])
        if not wcanon:
            wcanon.extend(
                sorted({_sanitize_observable_hist_segment(r) for r in wmap.keys()})[:32]
            )
        wglen = len(observable_metric_histories[on.id])
        for seg in wcanon:
            raw_top = wseg.get(seg)
            wn = (
                float(wmap[raw_top])
                if raw_top is not None and raw_top in wmap
                else float("nan")
            )
            wrk = f"{on.id}::top::{seg}"
            observable_metric_histories.setdefault(wrk, [])
            wrow = observable_metric_histories[wrk]
            while len(wrow) < wglen - 1:
                wrow.append(float("nan"))
            wrow.append(wn)
    elif agg_w == "tensor":
        wmap_t = _parameter_l2_norms_named(model, use_grad=False, max_params=MAX_OBSERVABLE_L2_TENSOR_SERIES)
        wseg_t = weight_l2_tensor_seg_to_raw.setdefault(on.id, {})
        for raw_n in wmap_t:
            wseg_t[_sanitize_observable_hist_segment(raw_n)] = raw_n
        wcanon_t = weight_l2_canonical_tensor.setdefault(on.id, [])
        if not wcanon_t:
            wcanon_t.extend(
                sorted({_sanitize_observable_hist_segment(r) for r in wmap_t.keys()})[
                    :MAX_OBSERVABLE_L2_TENSOR_SERIES
                ]
            )
        wglen_t = len(observable_metric_histories[on.id])
        for seg in wcanon_t:
            raw_n = wseg_t.get(seg)
            wn = (
                float(wmap_t[raw_n])
                if raw_n is not None and raw_n in wmap_t
                else float("nan")
            )
            wrk = f"{on.id}::tensor::{seg}"
            observable_metric_histories.setdefault(wrk, [])
            wrow = observable_metric_histories[wrk]
            while len(wrow) < wglen_t - 1:
                wrow.append(float("nan"))
            wrow.append(wn)
