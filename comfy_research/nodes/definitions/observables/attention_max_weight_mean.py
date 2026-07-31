"""observable_attention_max_weight_mean — NodeDef-channel definition + recorder.

Custom component adapter; spawn title "Attention max weight" differs from vizTitle.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from comfy_research.nodes.registry import observable_def, recorder_for
from comfy_research.nodes.schema import FrontendSpec, ObservableDef, SpawnSpec, VizSpec

if TYPE_CHECKING:  # pragma: no cover
    from comfy_research.engine.trainer.recorder import ObservableRecorder
    from comfy_research.schemas.graph import Node

ATTENTION_MAX_WEIGHT_MEAN = observable_def(
    ObservableDef(
        type="observable_attention_max_weight_mean",
        label="Attention max weight (mean)",
        hint="Mean max softmax mass per query; global = last layer or all layers = per-layer curves.",
        viz=VizSpec(
            variant="user",
            title="Attention max weight (mean)",
            info_markdown=(
                "**Attention max weight (mean)** — mean max softmax mass to any key. **Global** = "
                "last layer; **All layers** = per-layer curves + layer mean in the primary series."
            ),
            spawns=True,
            user_whitelisted=True,
            spawn=SpawnSpec(kind="user_scalar", title="Attention max weight", unit="mean max P"),
        ),
        frontend=FrontendSpec(component_key="AttentionMaxWeightMeanObservableNode"),
    )
)


@recorder_for(ATTENTION_MAX_WEIGHT_MEAN)
def record(rec: "ObservableRecorder", on: "Node") -> None:
    """Body moved verbatim from ObservableRecorder._record_attention_max_weight_mean."""
    from comfy_research.engine.trainer.observable_config import _obs_encoder_layer_mode
    from comfy_research.engine.trainer.observable_metrics import _attn_max_weight_mean

    encoder_obs_layer_canon = rec.encoder_obs_layer_canon
    observable_metric_histories = rec.observable_metric_histories
    od_mw: dict[str, Any] = on.data or {}
    enc_mw = _obs_encoder_layer_mode(od_mw)
    layers_mw = rec._attention_layers_for_log()
    if enc_mw == "all_layers":
        if layers_mw is None:
            observable_metric_histories[on.id].append(float("nan"))
            canon_mw = encoder_obs_layer_canon.setdefault(on.id, [])
            glen_mw = len(observable_metric_histories[on.id])
            for seg in canon_mw:
                rk_mw = f"{on.id}::layer::{seg}"
                observable_metric_histories.setdefault(rk_mw, [])
                row_mw = observable_metric_histories[rk_mw]
                while len(row_mw) < glen_mw - 1:
                    row_mw.append(float("nan"))
                row_mw.append(float("nan"))
        else:
            vals_mw = [_attn_max_weight_mean(t) for t in layers_mw]
            mean_mw = float(sum(vals_mw) / len(vals_mw)) if vals_mw else float("nan")
            observable_metric_histories[on.id].append(mean_mw)
            canon_mw = encoder_obs_layer_canon.setdefault(on.id, [])
            if not canon_mw:
                canon_mw.extend(str(i) for i in range(len(vals_mw)))
            glen_mw = len(observable_metric_histories[on.id])
            for seg in canon_mw:
                idx_mw = int(seg)
                v_mw = float(vals_mw[idx_mw]) if 0 <= idx_mw < len(vals_mw) else float("nan")
                rk_mw = f"{on.id}::layer::{seg}"
                observable_metric_histories.setdefault(rk_mw, [])
                row_mw = observable_metric_histories[rk_mw]
                while len(row_mw) < glen_mw - 1:
                    row_mw.append(float("nan"))
                row_mw.append(v_mw)
    else:
        attn_m = rec._attention_last_layer_for_log()
        if attn_m is None:
            observable_metric_histories[on.id].append(float("nan"))
        else:
            observable_metric_histories[on.id].append(_attn_max_weight_mean(attn_m))
