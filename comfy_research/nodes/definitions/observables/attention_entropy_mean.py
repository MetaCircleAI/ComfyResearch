"""observable_attention_entropy_mean — NodeDef-channel definition + recorder.

Custom component adapter (encoder layer-mode multiselect is UI-only, not a
manifest field). Spawn title "Attention entropy" differs from vizTitle.
Shared layer-canon state stays on the recorder (rec.encoder_obs_layer_canon).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from comfy_research.nodes.registry import observable_def, recorder_for
from comfy_research.nodes.schema import FrontendSpec, ObservableDef, SpawnSpec, VizSpec

if TYPE_CHECKING:  # pragma: no cover
    from comfy_research.engine.trainer.recorder import ObservableRecorder
    from comfy_research.schemas.graph import Node

ATTENTION_ENTROPY_MEAN = observable_def(
    ObservableDef(
        type="observable_attention_entropy_mean",
        label="Attention entropy (mean)",
        hint="Mean entropy of attention rows; global = last layer, all layers = one curve per encoder layer (same cores as sink mass).",
        viz=VizSpec(
            variant="user",
            title="Attention entropy (mean)",
            info_markdown=(
                "**Attention entropy (mean)** — mean Shannon entropy over attention keys. "
                "**Global**: last encoder layer (legacy default). **All layers**: average over "
                "layers in the primary series plus one curve per layer. Same supported cores as "
                "sink mass."
            ),
            spawns=True,
            user_whitelisted=True,
            spawn=SpawnSpec(kind="user_scalar", title="Attention entropy", unit="nat"),
        ),
        frontend=FrontendSpec(component_key="AttentionEntropyMeanObservableNode"),
    )
)


@recorder_for(ATTENTION_ENTROPY_MEAN)
def record(rec: "ObservableRecorder", on: "Node") -> None:
    """Body moved verbatim from ObservableRecorder._record_attention_entropy_mean."""
    from comfy_research.engine.trainer.observable_config import _obs_encoder_layer_mode
    from comfy_research.engine.trainer.observable_metrics import _attn_entropy_mean

    encoder_obs_layer_canon = rec.encoder_obs_layer_canon
    observable_metric_histories = rec.observable_metric_histories
    od_ent: dict[str, Any] = on.data or {}
    enc_ent = _obs_encoder_layer_mode(od_ent)
    layers_ent = rec._attention_layers_for_log()
    if enc_ent == "all_layers":
        if layers_ent is None:
            observable_metric_histories[on.id].append(float("nan"))
            canon_ent = encoder_obs_layer_canon.setdefault(on.id, [])
            glen_ent = len(observable_metric_histories[on.id])
            for seg in canon_ent:
                rk_ent = f"{on.id}::layer::{seg}"
                observable_metric_histories.setdefault(rk_ent, [])
                row_ent = observable_metric_histories[rk_ent]
                while len(row_ent) < glen_ent - 1:
                    row_ent.append(float("nan"))
                row_ent.append(float("nan"))
        else:
            vals_ent = [_attn_entropy_mean(t) for t in layers_ent]
            mean_ent = float(sum(vals_ent) / len(vals_ent)) if vals_ent else float("nan")
            observable_metric_histories[on.id].append(mean_ent)
            canon_ent = encoder_obs_layer_canon.setdefault(on.id, [])
            if not canon_ent:
                canon_ent.extend(str(i) for i in range(len(vals_ent)))
            glen_ent = len(observable_metric_histories[on.id])
            for seg in canon_ent:
                idx_ent = int(seg)
                v_ent = float(vals_ent[idx_ent]) if 0 <= idx_ent < len(vals_ent) else float("nan")
                rk_ent = f"{on.id}::layer::{seg}"
                observable_metric_histories.setdefault(rk_ent, [])
                row_ent = observable_metric_histories[rk_ent]
                while len(row_ent) < glen_ent - 1:
                    row_ent.append(float("nan"))
                row_ent.append(v_ent)
    else:
        attn_e = rec._attention_last_layer_for_log()
        if attn_e is None:
            observable_metric_histories[on.id].append(float("nan"))
        else:
            observable_metric_histories[on.id].append(_attn_entropy_mean(attn_e))
