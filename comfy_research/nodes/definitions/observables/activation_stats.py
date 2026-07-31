"""observable_activation_stats — NodeDef-channel definition + recorder.

Custom component adapter (activationStatsLayers toggle is UI-only, not a
manifest field). The spawned hessian-shaped visualization uses seriesLabels
["mean", "std"]. ``observable_viz.py`` shapes the ::std, ::layer_mean, and
::layer_std subseries.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from comfy_research.nodes.registry import observable_def, recorder_for
from comfy_research.nodes.schema import FrontendSpec, ObservableDef, SpawnSpec, VizSpec

if TYPE_CHECKING:  # pragma: no cover
    from comfy_research.engine.trainer.recorder import ObservableRecorder
    from comfy_research.schemas.graph import Node

ACTIVATION_STATS = observable_def(
    ObservableDef(
        type="observable_activation_stats",
        label="Activation mean/std",
        hint=(
            "Mean and std of Linear/Conv activations: global = equal average across layer buckets "
            "(.layers.0., …) vs other modules; or separate series per bucket."
        ),
        viz=VizSpec(
            variant="activation_stats",
            title="Activation mean/std",
            info_markdown=(
                "**Activation mean/std** — mean and standard deviation of activations at Linear/Conv "
                "outputs (extra forward). Hooks are grouped by encoder layer index in the module path "
                "(`layers.0`, `layers.1`, …) vs **other** modules. **Layers → Global** averages those "
                "bucket statistics equally; **All layers** logs separate mean/std series per bucket."
            ),
            spawns=True,
            spawn=SpawnSpec(kind="hessian_topk", fixed_top_k=2, series_labels=("mean", "std")),
        ),
        frontend=FrontendSpec(component_key="ActivationStatsObservableNode"),
    )
)


@recorder_for(ACTIVATION_STATS)
def record(rec: "ObservableRecorder", on: "Node") -> None:
    """Body moved verbatim from ObservableRecorder._record_activation_stats."""
    from comfy_research.engine.trainer.observable_config import (
        _activation_stats_layer_mode,
        _activation_stats_ordered_bucket_keys,
    )
    from comfy_research.engine.trainer.observable_metrics import _activation_mean_std_bucketed

    activation_stats_layer_canon = rec.activation_stats_layer_canon
    model = rec.model
    observable_metric_histories = rec.observable_metric_histories
    xr = rec._xr
    od_act: dict[str, Any] = on.data or {}
    layer_mode_act = _activation_stats_layer_mode(od_act)
    std_key_act = f"{on.id}::std"
    observable_metric_histories.setdefault(std_key_act, [])
    buckets_act = _activation_mean_std_bucketed(model, xr)
    if not buckets_act:
        observable_metric_histories[on.id].append(float("nan"))
        observable_metric_histories[std_key_act].append(float("nan"))
        if layer_mode_act == "all_layers":
            canon_act_nan = activation_stats_layer_canon.setdefault(on.id, [])
            glen_act_nan = len(observable_metric_histories[on.id])
            for seg_act in canon_act_nan:
                rk_am = f"{on.id}::layer_mean::{seg_act}"
                rk_as = f"{on.id}::layer_std::{seg_act}"
                observable_metric_histories.setdefault(rk_am, [])
                observable_metric_histories.setdefault(rk_as, [])
                row_am = observable_metric_histories[rk_am]
                row_as = observable_metric_histories[rk_as]
                while len(row_am) < glen_act_nan - 1:
                    row_am.append(float("nan"))
                while len(row_as) < glen_act_nan - 1:
                    row_as.append(float("nan"))
                row_am.append(float("nan"))
                row_as.append(float("nan"))
    else:
        mean_g_act = sum(t[0] for t in buckets_act.values()) / len(buckets_act)
        std_g_act = sum(t[1] for t in buckets_act.values()) / len(buckets_act)
        observable_metric_histories[on.id].append(mean_g_act)
        observable_metric_histories[std_key_act].append(std_g_act)
        if layer_mode_act == "all_layers":
            canon_act = activation_stats_layer_canon.setdefault(on.id, [])
            if not canon_act:
                canon_act.extend(_activation_stats_ordered_bucket_keys(buckets_act))
            glen_act_l = len(observable_metric_histories[on.id])
            for seg_act in canon_act:
                pr_act = buckets_act.get(seg_act)
                mv_act = float(pr_act[0]) if pr_act else float("nan")
                sv_act = float(pr_act[1]) if pr_act else float("nan")
                rk_am = f"{on.id}::layer_mean::{seg_act}"
                rk_as = f"{on.id}::layer_std::{seg_act}"
                observable_metric_histories.setdefault(rk_am, [])
                observable_metric_histories.setdefault(rk_as, [])
                row_am = observable_metric_histories[rk_am]
                row_as = observable_metric_histories[rk_as]
                while len(row_am) < glen_act_l - 1:
                    row_am.append(float("nan"))
                while len(row_as) < glen_act_l - 1:
                    row_as.append(float("nan"))
                row_am.append(mv_act)
                row_as.append(sv_act)
