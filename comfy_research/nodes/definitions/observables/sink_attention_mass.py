"""observable_sink_attention_mass — NodeDef-channel definition + recorder.

Custom component adapter (SinkAttentionMassObservableNode keeps the layer-mode
multiselect whose key is UI-only, not a manifest field). First hint-less def:
the OVERRIDES-era manifest entry has no hint key, so none is declared here.
Layer-canon state stays on the recorder (rec.sink_attn_mass_layer_canon).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from comfy_research.nodes.registry import observable_def, recorder_for
from comfy_research.nodes.schema import FrontendSpec, IntField, ObservableDef, SpawnSpec, VizSpec

if TYPE_CHECKING:  # pragma: no cover
    from comfy_research.engine.trainer.recorder import ObservableRecorder
    from comfy_research.schemas.graph import Node

SINK_ATTENTION_MASS = observable_def(
    ObservableDef(
        type="observable_sink_attention_mass",
        label="Sink attention mass",
        hint="Mean softmax attention to a chosen key index (sink); global = averaged over encoder layers, or one series per layer. Attention-only + transformer / numeric token paths.",
        viz=VizSpec(
            variant="user",
            title="Sink attention mass",
            info_markdown=(
                "**Sink attention mass** — mass on a chosen key index (often BOS / first token) in "
                "softmax self-attention; high values indicate attention sink behavior. **Layers → "
                "Global** averages mean sink mass over all encoder layers; **All layers** records one "
                "value per layer (multi-curve viz). Models: **attention_only_model** (one layer), "
                "**transformer_token_model**, **transformer_multi_token_model**, "
                "**numeric_transformer_model**."
            ),
            spawns=True,
            user_whitelisted=True,
            spawn=SpawnSpec(kind="user_scalar", unit="mean P(sink)"),
        ),
        fields=(IntField(key="sinkTokenIndex", label="Sink Token Index", default=0),),
        frontend=FrontendSpec(component_key="SinkAttentionMassObservableNode"),
    )
)


@recorder_for(SINK_ATTENTION_MASS)
def record(rec: "ObservableRecorder", on: "Node") -> None:
    """Body moved verbatim from ObservableRecorder._record_sink_attention_mass."""
    from comfy_research.engine.trainer.observable_config import _sink_attention_mass_layer_mode
    from comfy_research.engine.trainer.observable_metrics import _attn_sink_mass
    from comfy_research.engine.trainer.scalar import _scalar_int

    observable_metric_histories = rec.observable_metric_histories
    sink_attn_mass_layer_canon = rec.sink_attn_mass_layer_canon
    od_sk: dict[str, Any] = on.data or {}
    sink_i = _scalar_int(od_sk.get("sinkTokenIndex"), 0)
    layer_mode_sk = _sink_attention_mass_layer_mode(od_sk)
    layers_sk = rec._attention_layers_for_log()
    if layers_sk is None:
        observable_metric_histories[on.id].append(float("nan"))
        if layer_mode_sk == "all_layers":
            canon_sink = sink_attn_mass_layer_canon.setdefault(on.id, [])
            glen_nan = len(observable_metric_histories[on.id])
            for seg in canon_sink:
                rk_sink = f"{on.id}::layer::{seg}"
                observable_metric_histories.setdefault(rk_sink, [])
                row_sink = observable_metric_histories[rk_sink]
                while len(row_sink) < glen_nan - 1:
                    row_sink.append(float("nan"))
                row_sink.append(float("nan"))
    else:
        masses_sk = [_attn_sink_mass(p, sink_i) for p in layers_sk]
        mean_sk = (
            float(sum(masses_sk) / len(masses_sk))
            if masses_sk
            else float("nan")
        )
        observable_metric_histories[on.id].append(mean_sk)
        if layer_mode_sk == "all_layers":
            canon_sink = sink_attn_mass_layer_canon.setdefault(on.id, [])
            if not canon_sink:
                canon_sink.extend(str(i) for i in range(len(masses_sk)))
            glen_sink_l = len(observable_metric_histories[on.id])
            for seg in canon_sink:
                idx_sink = int(seg)
                val_sink = (
                    float(masses_sk[idx_sink])
                    if 0 <= idx_sink < len(masses_sk)
                    else float("nan")
                )
                rk_sink = f"{on.id}::layer::{seg}"
                observable_metric_histories.setdefault(rk_sink, [])
                row_sink = observable_metric_histories[rk_sink]
                while len(row_sink) < glen_sink_l - 1:
                    row_sink.append(float("nan"))
                row_sink.append(val_sink)
