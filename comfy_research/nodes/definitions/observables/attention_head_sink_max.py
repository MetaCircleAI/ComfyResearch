"""observable_attention_head_sink_max — NodeDef-channel definition + recorder.

Custom component adapter (encoder layer-mode multiselect is UI-only, not a
manifest field). SpawnSpec.title is "Head sink max", while vizTitle is
"Attention head sink max".
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from comfy_research.nodes.registry import observable_def, recorder_for
from comfy_research.nodes.schema import FrontendSpec, IntField, ObservableDef, SpawnSpec, VizSpec

if TYPE_CHECKING:  # pragma: no cover
    from comfy_research.engine.trainer.recorder import ObservableRecorder
    from comfy_research.schemas.graph import Node

ATTENTION_HEAD_SINK_MAX = observable_def(
    ObservableDef(
        type="observable_attention_head_sink_max",
        label="Attention head sink max",
        hint="Max over heads of mean attention to sink key; global vs per-layer series like sink mass.",
        viz=VizSpec(
            variant="user",
            title="Attention head sink max",
            info_markdown=(
                "**Attention head sink max** — max over heads of mean mass on the sink key. "
                "**Global** = last layer; **All layers** = per-layer curves + mean."
            ),
            spawns=True,
            user_whitelisted=True,
            spawn=SpawnSpec(kind="user_scalar", title="Head sink max", unit="mean P(sink)"),
        ),
        fields=(IntField(key="sinkTokenIndex", label="Sink Token Index", default=0),),
        frontend=FrontendSpec(component_key="AttentionHeadSinkMaxObservableNode"),
    )
)


@recorder_for(ATTENTION_HEAD_SINK_MAX)
def record(rec: "ObservableRecorder", on: "Node") -> None:
    """Body moved verbatim from ObservableRecorder._record_attention_head_sink_max."""
    from comfy_research.engine.trainer.observable_config import _obs_encoder_layer_mode
    from comfy_research.engine.trainer.observable_metrics import _attn_head_sink_max
    from comfy_research.engine.trainer.scalar import _scalar_int

    encoder_obs_layer_canon = rec.encoder_obs_layer_canon
    observable_metric_histories = rec.observable_metric_histories
    od_hk: dict[str, Any] = on.data or {}
    sink_h = _scalar_int(od_hk.get("sinkTokenIndex"), 0)
    enc_hk = _obs_encoder_layer_mode(od_hk)
    layers_hk = rec._attention_layers_for_log()
    if enc_hk == "all_layers":
        if layers_hk is None:
            observable_metric_histories[on.id].append(float("nan"))
            canon_hk = encoder_obs_layer_canon.setdefault(on.id, [])
            glen_hk = len(observable_metric_histories[on.id])
            for seg in canon_hk:
                rk_hk = f"{on.id}::layer::{seg}"
                observable_metric_histories.setdefault(rk_hk, [])
                row_hk = observable_metric_histories[rk_hk]
                while len(row_hk) < glen_hk - 1:
                    row_hk.append(float("nan"))
                row_hk.append(float("nan"))
        else:
            vals_hk = [_attn_head_sink_max(t, sink_h) for t in layers_hk]
            mean_hk = float(sum(vals_hk) / len(vals_hk)) if vals_hk else float("nan")
            observable_metric_histories[on.id].append(mean_hk)
            canon_hk = encoder_obs_layer_canon.setdefault(on.id, [])
            if not canon_hk:
                canon_hk.extend(str(i) for i in range(len(vals_hk)))
            glen_hk = len(observable_metric_histories[on.id])
            for seg in canon_hk:
                idx_hk = int(seg)
                v_hk = float(vals_hk[idx_hk]) if 0 <= idx_hk < len(vals_hk) else float("nan")
                rk_hk = f"{on.id}::layer::{seg}"
                observable_metric_histories.setdefault(rk_hk, [])
                row_hk = observable_metric_histories[rk_hk]
                while len(row_hk) < glen_hk - 1:
                    row_hk.append(float("nan"))
                row_hk.append(v_hk)
    else:
        attn_h = rec._attention_last_layer_for_log()
        if attn_h is None:
            observable_metric_histories[on.id].append(float("nan"))
        else:
            observable_metric_histories[on.id].append(_attn_head_sink_max(attn_h, sink_h))
