"""observable_activation_norm_mean — NodeDef-channel definition + recorder.

Custom component adapter; spawn title "Activation norm mean" differs from vizTitle.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from comfy_research.nodes.registry import observable_def, recorder_for
from comfy_research.nodes.schema import FrontendSpec, ObservableDef, SpawnSpec, VizSpec

if TYPE_CHECKING:  # pragma: no cover
    from comfy_research.engine.trainer.recorder import ObservableRecorder
    from comfy_research.schemas.graph import Node

ACTIVATION_NORM_MEAN = observable_def(
    ObservableDef(
        type="observable_activation_norm_mean",
        label="Activation norm (mean)",
        hint="Mean L2 norm of hooked Linear/Conv outputs; global pooled or per activation-stats bucket (multi-curve).",
        viz=VizSpec(
            variant="user",
            title="Activation norm (mean)",
            info_markdown=(
                "**Activation norm (mean)** — mean L2 norm of Linear/Conv outputs (extra forward). "
                "**Global**: average over hooked modules. **All layers**: one curve per "
                "activation-stats bucket (`layers.0`, …, **other**), primary = mean of bucket means."
            ),
            spawns=True,
            user_whitelisted=True,
            spawn=SpawnSpec(kind="user_scalar", title="Activation norm mean", unit="‖h‖ mean"),
        ),
        frontend=FrontendSpec(component_key="ActivationNormMeanObservableNode"),
    )
)


@recorder_for(ACTIVATION_NORM_MEAN)
def record(rec: "ObservableRecorder", on: "Node") -> None:
    """Body moved verbatim from ObservableRecorder._record_activation_norm_mean."""
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
    od_nm: dict[str, Any] = on.data or {}
    if _obs_encoder_layer_mode(od_nm) == "all_layers":
        buck_nm = _activation_norm_mean_and_outlier_per_bucket(model, xr)
        if not buck_nm:
            observable_metric_histories[on.id].append(float("nan"))
            canon_nm = encoder_obs_layer_canon.setdefault(on.id, [])
            glen_nm = len(observable_metric_histories[on.id])
            for seg in canon_nm:
                rk_nm = f"{on.id}::layer::{seg}"
                observable_metric_histories.setdefault(rk_nm, [])
                row_nm = observable_metric_histories[rk_nm]
                while len(row_nm) < glen_nm - 1:
                    row_nm.append(float("nan"))
                row_nm.append(float("nan"))
        else:
            ordered_nm = _activation_stats_ordered_bucket_keys(
                {k: (0.0, 0.0) for k in buck_nm.keys()}
            )
            vals_nm = [float(buck_nm[k][0]) if k in buck_nm else float("nan") for k in ordered_nm]
            mean_nm = float(sum(vals_nm) / len(vals_nm)) if vals_nm else float("nan")
            observable_metric_histories[on.id].append(mean_nm)
            canon_nm = encoder_obs_layer_canon.setdefault(on.id, [])
            if not canon_nm:
                canon_nm.extend(ordered_nm)
            glen_nm = len(observable_metric_histories[on.id])
            for seg in canon_nm:
                pr_nm = buck_nm.get(seg)
                v_nm = float(pr_nm[0]) if pr_nm else float("nan")
                rk_nm = f"{on.id}::layer::{seg}"
                observable_metric_histories.setdefault(rk_nm, [])
                row_nm = observable_metric_histories[rk_nm]
                while len(row_nm) < glen_nm - 1:
                    row_nm.append(float("nan"))
                row_nm.append(v_nm)
    else:
        n_m, _r = _activation_norm_mean_and_outlier_ratio(model, xr)
        observable_metric_histories[on.id].append(n_m)
