"""observable_embedding_effective_rank — NodeDef-channel definition + recorder.

Custom component adapter; spawn title equals vizTitle.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from comfy_research.nodes.registry import observable_def, recorder_for
from comfy_research.nodes.schema import FrontendSpec, ObservableDef, SpawnSpec, VizSpec

if TYPE_CHECKING:  # pragma: no cover
    from comfy_research.engine.trainer.recorder import ObservableRecorder
    from comfy_research.schemas.graph import Node

EMBEDDING_EFFECTIVE_RANK = observable_def(
    ObservableDef(
        type="observable_embedding_effective_rank",
        label="Embedding effective rank",
        hint="Effective rank of the token embedding (global) or max rank per encoder block .layers.i (all layers, multi-curve).",
        viz=VizSpec(
            variant="user",
            title="Embedding effective rank",
            info_markdown=(
                "**Embedding effective rank** — **Global**: entropy-based effective rank of the "
                "token embedding matrix. **All layers**: max effective rank among 2D weights in "
                "each encoder block (`layers.i`), one curve per block."
            ),
            spawns=True,
            user_whitelisted=True,
            spawn=SpawnSpec(kind="user_scalar", unit="rank"),
        ),
        frontend=FrontendSpec(component_key="EmbeddingEffectiveRankObservableNode"),
    )
)


@recorder_for(EMBEDDING_EFFECTIVE_RANK)
def record(rec: "ObservableRecorder", on: "Node") -> None:
    """Body moved verbatim from ObservableRecorder._record_embedding_effective_rank."""
    import math

    from comfy_research.engine.analysis.tensor_metrics import effective_rank_from_matrix
    from comfy_research.engine.trainer.observable_config import (
        _EMBEDDING_OBSERVABLE_MODEL_TYPES,
        _obs_encoder_layer_mode,
    )
    from comfy_research.engine.trainer.observable_metrics import _effective_rank_max_per_encoder_layer

    encoder_obs_layer_canon = rec.encoder_obs_layer_canon
    model = rec.model
    observable_metric_histories = rec.observable_metric_histories
    od_er: dict[str, Any] = on.data or {}
    if _obs_encoder_layer_mode(od_er) == "all_layers":
        lay_er = _effective_rank_max_per_encoder_layer(model)
        if not lay_er:
            observable_metric_histories[on.id].append(float("nan"))
            canon_er = encoder_obs_layer_canon.setdefault(on.id, [])
            glen_er = len(observable_metric_histories[on.id])
            for seg in canon_er:
                rk_er = f"{on.id}::layer::{seg}"
                observable_metric_histories.setdefault(rk_er, [])
                row_er = observable_metric_histories[rk_er]
                while len(row_er) < glen_er - 1:
                    row_er.append(float("nan"))
                row_er.append(float("nan"))
        else:
            keys_er = sorted(lay_er.keys(), key=lambda s: int(s) if s.isdigit() else 0)
            finite_er = [float(lay_er[k]) for k in keys_er if math.isfinite(float(lay_er[k]))]
            mean_er = float(sum(finite_er) / len(finite_er)) if finite_er else float("nan")
            observable_metric_histories[on.id].append(mean_er)
            canon_er = encoder_obs_layer_canon.setdefault(on.id, [])
            if not canon_er:
                canon_er.extend(keys_er)
            glen_er = len(observable_metric_histories[on.id])
            for seg in canon_er:
                v_er = float(lay_er[seg]) if seg in lay_er and math.isfinite(float(lay_er[seg])) else float(
                    "nan"
                )
                rk_er = f"{on.id}::layer::{seg}"
                observable_metric_histories.setdefault(rk_er, [])
                row_er = observable_metric_histories[rk_er]
                while len(row_er) < glen_er - 1:
                    row_er.append(float("nan"))
                row_er.append(v_er)
    else:
        if not isinstance(model, _EMBEDDING_OBSERVABLE_MODEL_TYPES):
            observable_metric_histories[on.id].append(float("nan"))
            return
        arrays_er = model.observable_numpy_arrays()
        emb_er = arrays_er.get("embedding")
        if emb_er is None:
            observable_metric_histories[on.id].append(float("nan"))
            return
        observable_metric_histories[on.id].append(float(effective_rank_from_matrix(emb_er)))
