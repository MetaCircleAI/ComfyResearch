"""observable_attention_position_bias_ratio — NodeDef-channel definition + recorder.

Custom component adapter; spawn title "Position bias ratio" differs from vizTitle.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from comfy_research.nodes.registry import observable_def, recorder_for
from comfy_research.nodes.schema import FrontendSpec, ObservableDef, SpawnSpec, VizSpec

if TYPE_CHECKING:  # pragma: no cover
    from comfy_research.engine.trainer.recorder import ObservableRecorder
    from comfy_research.schemas.graph import Node

ATTENTION_POSITION_BIAS_RATIO = observable_def(
    ObservableDef(
        type="observable_attention_position_bias_ratio",
        label="Attention position bias ratio",
        hint="Early-key vs average-key mass ratio; global vs per-layer series.",
        viz=VizSpec(
            variant="user",
            title="Attention position bias ratio",
            info_markdown=(
                "**Attention position bias ratio** — early-key vs average-key mass ratio. "
                "**Global** = last layer; **All layers** = per-layer curves + mean."
            ),
            spawns=True,
            user_whitelisted=True,
            spawn=SpawnSpec(kind="user_scalar", title="Position bias ratio", unit="ratio"),
        ),
        frontend=FrontendSpec(component_key="AttentionPositionBiasRatioObservableNode"),
    )
)


@recorder_for(ATTENTION_POSITION_BIAS_RATIO)
def record(rec: "ObservableRecorder", on: "Node") -> None:
    """Body moved verbatim from ObservableRecorder._record_attention_position_bias_ratio."""
    from comfy_research.engine.trainer.observable_config import _obs_encoder_layer_mode
    from comfy_research.engine.trainer.observable_metrics import _attn_position_bias_ratio

    encoder_obs_layer_canon = rec.encoder_obs_layer_canon
    observable_metric_histories = rec.observable_metric_histories
    od_pb: dict[str, Any] = on.data or {}
    enc_pb = _obs_encoder_layer_mode(od_pb)
    layers_pb = rec._attention_layers_for_log()
    if enc_pb == "all_layers":
        if layers_pb is None:
            observable_metric_histories[on.id].append(float("nan"))
            canon_pb = encoder_obs_layer_canon.setdefault(on.id, [])
            glen_pb = len(observable_metric_histories[on.id])
            for seg in canon_pb:
                rk_pb = f"{on.id}::layer::{seg}"
                observable_metric_histories.setdefault(rk_pb, [])
                row_pb = observable_metric_histories[rk_pb]
                while len(row_pb) < glen_pb - 1:
                    row_pb.append(float("nan"))
                row_pb.append(float("nan"))
        else:
            vals_pb = [_attn_position_bias_ratio(t) for t in layers_pb]
            mean_pb = float(sum(vals_pb) / len(vals_pb)) if vals_pb else float("nan")
            observable_metric_histories[on.id].append(mean_pb)
            canon_pb = encoder_obs_layer_canon.setdefault(on.id, [])
            if not canon_pb:
                canon_pb.extend(str(i) for i in range(len(vals_pb)))
            glen_pb = len(observable_metric_histories[on.id])
            for seg in canon_pb:
                idx_pb = int(seg)
                v_pb = float(vals_pb[idx_pb]) if 0 <= idx_pb < len(vals_pb) else float("nan")
                rk_pb = f"{on.id}::layer::{seg}"
                observable_metric_histories.setdefault(rk_pb, [])
                row_pb = observable_metric_histories[rk_pb]
                while len(row_pb) < glen_pb - 1:
                    row_pb.append(float("nan"))
                row_pb.append(v_pb)
    else:
        attn_p = rec._attention_last_layer_for_log()
        if attn_p is None:
            observable_metric_histories[on.id].append(float("nan"))
        else:
            observable_metric_histories[on.id].append(_attn_position_bias_ratio(attn_p))
