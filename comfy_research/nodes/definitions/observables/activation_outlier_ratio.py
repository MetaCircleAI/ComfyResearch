"""observable_activation_outlier_ratio — NodeDef-channel definition + recorder.

Custom component adapter; spawn title equals vizTitle (no SpawnSpec.title).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from comfy_research.nodes.registry import observable_def, recorder_for
from comfy_research.nodes.schema import FrontendSpec, ObservableDef, SpawnSpec, VizSpec

if TYPE_CHECKING:  # pragma: no cover
    from comfy_research.engine.trainer.recorder import ObservableRecorder
    from comfy_research.schemas.graph import Node

ACTIVATION_OUTLIER_RATIO = observable_def(
    ObservableDef(
        type="observable_activation_outlier_ratio",
        label="Activation outlier ratio",
        hint="max|act|/mean|act| in the same hook pass; global or per-bucket (multi-curve).",
        viz=VizSpec(
            variant="user",
            title="Activation outlier ratio",
            info_markdown=(
                "**Activation outlier ratio** — max |activation| / mean |activation| in the same "
                "hook pass as norm mean. **Global**: pooled; **All layers**: one ratio per bucket, "
                "primary = mean of bucket ratios."
            ),
            spawns=True,
            user_whitelisted=True,
            spawn=SpawnSpec(kind="user_scalar", unit="ratio"),
        ),
        frontend=FrontendSpec(component_key="ActivationOutlierRatioObservableNode"),
    )
)


@recorder_for(ACTIVATION_OUTLIER_RATIO)
def record(rec: "ObservableRecorder", on: "Node") -> None:
    """Body moved verbatim from ObservableRecorder._record_activation_outlier_ratio."""
    from comfy_research.engine.trainer.observable_config import (
        _activation_stats_ordered_bucket_keys,
        _obs_encoder_layer_mode,
    )
    from comfy_research.engine.trainer.observable_metrics import (
        _activation_norm_mean_and_outlier_per_bucket,
        _activation_norm_mean_and_outlier_ratio,
    )

    encoder_obs_layer_canon = rec.encoder_obs_layer_canon
    model = rec.model
    observable_metric_histories = rec.observable_metric_histories
    xr = rec._xr
    od_ol: dict[str, Any] = on.data or {}
    if _obs_encoder_layer_mode(od_ol) == "all_layers":
        buck_ol = _activation_norm_mean_and_outlier_per_bucket(model, xr)
        if not buck_ol:
            observable_metric_histories[on.id].append(float("nan"))
            canon_ol = encoder_obs_layer_canon.setdefault(on.id, [])
            glen_ol = len(observable_metric_histories[on.id])
            for seg in canon_ol:
                rk_ol = f"{on.id}::layer::{seg}"
                observable_metric_histories.setdefault(rk_ol, [])
                row_ol = observable_metric_histories[rk_ol]
                while len(row_ol) < glen_ol - 1:
                    row_ol.append(float("nan"))
                row_ol.append(float("nan"))
        else:
            ordered_ol = _activation_stats_ordered_bucket_keys(
                {k: (0.0, 0.0) for k in buck_ol.keys()}
            )
            vals_ol = [float(buck_ol[k][1]) if k in buck_ol else float("nan") for k in ordered_ol]
            mean_ol = float(sum(vals_ol) / len(vals_ol)) if vals_ol else float("nan")
            observable_metric_histories[on.id].append(mean_ol)
            canon_ol = encoder_obs_layer_canon.setdefault(on.id, [])
            if not canon_ol:
                canon_ol.extend(ordered_ol)
            glen_ol = len(observable_metric_histories[on.id])
            for seg in canon_ol:
                pr_ol = buck_ol.get(seg)
                v_ol = float(pr_ol[1]) if pr_ol else float("nan")
                rk_ol = f"{on.id}::layer::{seg}"
                observable_metric_histories.setdefault(rk_ol, [])
                row_ol = observable_metric_histories[rk_ol]
                while len(row_ol) < glen_ol - 1:
                    row_ol.append(float("nan"))
                row_ol.append(v_ol)
    else:
        _n_m, r_o = _activation_norm_mean_and_outlier_ratio(model, xr)
        observable_metric_histories[on.id].append(r_o)
