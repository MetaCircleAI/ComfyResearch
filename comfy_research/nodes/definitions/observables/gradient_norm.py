"""observable_gradient_norm — NodeDef-channel definition + recorder.

The custom component mirrors ``normAggregation`` to ``perTopLevel``, a side
effect that GenericObservableNode cannot provide. Canonical-segment state is
stored on the recorder.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from comfy_research.nodes.registry import observable_def, recorder_for
from comfy_research.nodes.schema import BoolField, EnumField, FrontendSpec, ObservableDef, SpawnSpec, VizSpec

if TYPE_CHECKING:  # pragma: no cover
    from comfy_research.engine.trainer.recorder import ObservableRecorder
    from comfy_research.schemas.graph import Node

GRADIENT_NORM = observable_def(
    ObservableDef(
        type="observable_gradient_norm",
        label="Gradient norm",
        hint=(
            "Log L2 norm of gradients each step: global, per top-level name segment, or per-parameter "
            "tensor (capped). Optional normalization by √(parameter count)."
        ),
        viz=VizSpec(
            variant="gradient_norm",
            title="Gradient norm",
            info_markdown=(
                "**Gradient norm** — L2 norm of parameter gradients after each logged step: "
                "**global**, **top-level module** (first name segment), or **per-parameter tensor** "
                "(one curve per parameter with gradients, capped). **Normalized** (default): divide "
                "by √(parameter count) in that scope."
            ),
            spawns=True,
            spawn=SpawnSpec(kind="hessian_topk", fixed_top_k=1),
        ),
        fields=(
            EnumField(key="normAggregation", label="Norm Aggregation", default="global"),
            BoolField(key="gradientNormNormalized", label="Gradient Norm Normalized", default=True),
        ),
        frontend=FrontendSpec(component_key="GradientNormObservableNode"),
    )
)


@recorder_for(GRADIENT_NORM)
def record(rec: "ObservableRecorder", on: "Node") -> None:
    """Body moved verbatim from ObservableRecorder._record_gradient_norm."""
    from comfy_research.engine.trainer.observable_config import (
        _observable_gradient_norm_normalized,
        _observable_l2_aggregation,
        _sanitize_observable_hist_segment,
    )
    from comfy_research.engine.trainer.tensor_norms import (
        MAX_OBSERVABLE_L2_TENSOR_SERIES,
        _gradient_l2_norm_global,
        _gradient_l2_norm_per_top_level,
        _parameter_l2_norms_named,
    )

    gradient_norm_canonical_segments = rec.gradient_norm_canonical_segments
    gradient_norm_canonical_tensor = rec.gradient_norm_canonical_tensor
    gradient_norm_seg_to_raw = rec.gradient_norm_seg_to_raw
    gradient_norm_tensor_seg_to_raw = rec.gradient_norm_tensor_seg_to_raw
    model = rec.model
    observable_metric_histories = rec.observable_metric_histories
    od_gn: dict[str, Any] = on.data or {}
    agg_gn = _observable_l2_aggregation(od_gn)
    gn_norm = _observable_gradient_norm_normalized(od_gn)
    observable_metric_histories[on.id].append(_gradient_l2_norm_global(model, normalize=gn_norm))
    if agg_gn == "top_level_module":
        norm_map = _gradient_l2_norm_per_top_level(model, normalize=gn_norm)
        seg_map = gradient_norm_seg_to_raw.setdefault(on.id, {})
        for raw_top in norm_map:
            seg_map[_sanitize_observable_hist_segment(raw_top)] = raw_top
        canon = gradient_norm_canonical_segments.setdefault(on.id, [])
        if not canon:
            canon.extend(
                sorted({_sanitize_observable_hist_segment(r) for r in norm_map.keys()})[:32]
            )
        # Keep each ``{id}::top::{seg}`` list the same length as the global series. The first
        # ``record`` may run under no_grad (no grads) so no per-top rows yet; later rows must
        # backfill NaNs so ``observable_viz_metric_updates`` does not drop multi-series payloads.
        glen = len(observable_metric_histories[on.id])
        for seg in canon:
            raw_top = seg_map.get(seg)
            gn = float(norm_map[raw_top]) if raw_top is not None and raw_top in norm_map else float("nan")
            rk = f"{on.id}::top::{seg}"
            observable_metric_histories.setdefault(rk, [])
            row = observable_metric_histories[rk]
            while len(row) < glen - 1:
                row.append(float("nan"))
            row.append(gn)
    elif agg_gn == "tensor":
        norm_map_t = _parameter_l2_norms_named(
            model,
            use_grad=True,
            max_params=MAX_OBSERVABLE_L2_TENSOR_SERIES,
            normalize_grad=gn_norm,
        )
        seg_map_t = gradient_norm_tensor_seg_to_raw.setdefault(on.id, {})
        for raw_n in norm_map_t:
            seg_map_t[_sanitize_observable_hist_segment(raw_n)] = raw_n
        canon_t = gradient_norm_canonical_tensor.setdefault(on.id, [])
        if not canon_t:
            canon_t.extend(
                sorted({_sanitize_observable_hist_segment(r) for r in norm_map_t.keys()})[
                    :MAX_OBSERVABLE_L2_TENSOR_SERIES
                ]
            )
        glen_t = len(observable_metric_histories[on.id])
        for seg in canon_t:
            raw_n = seg_map_t.get(seg)
            gn = (
                float(norm_map_t[raw_n])
                if raw_n is not None and raw_n in norm_map_t
                else float("nan")
            )
            rk = f"{on.id}::tensor::{seg}"
            observable_metric_histories.setdefault(rk, [])
            row = observable_metric_histories[rk]
            while len(row) < glen_t - 1:
                row.append(float("nan"))
            row.append(gn)
