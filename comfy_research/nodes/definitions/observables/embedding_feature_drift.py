"""observable_embedding_feature_drift — NodeDef-channel definition + recorder.

Custom component adapter; drift snapshot state stays on the recorder
(rec.embedding_prev_for_drift / rec.embedding_prev_layer_flat_for_drift).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from comfy_research.nodes.registry import observable_def, recorder_for
from comfy_research.nodes.schema import FrontendSpec, ObservableDef, SpawnSpec, VizSpec

if TYPE_CHECKING:  # pragma: no cover
    from comfy_research.engine.trainer.recorder import ObservableRecorder
    from comfy_research.schemas.graph import Node

EMBEDDING_FEATURE_DRIFT = observable_def(
    ObservableDef(
        type="observable_embedding_feature_drift",
        label="Embedding feature drift",
        hint="1 − cosine between successive embeddings (global) or between flattened weights per .layers.i (all layers).",
        viz=VizSpec(
            variant="user",
            title="Embedding feature drift",
            info_markdown=(
                "**Embedding feature drift** — **Global**: 1 − cosine between consecutive "
                "flattened embedding matrices. **All layers**: same metric on flattened weights "
                "concatenated per encoder block."
            ),
            spawns=True,
            user_whitelisted=True,
            spawn=SpawnSpec(kind="user_scalar", unit="1 − cos"),
        ),
        frontend=FrontendSpec(component_key="EmbeddingFeatureDriftObservableNode"),
    )
)


@recorder_for(EMBEDDING_FEATURE_DRIFT)
def record(rec: "ObservableRecorder", on: "Node") -> None:
    """Body moved verbatim from ObservableRecorder._record_embedding_feature_drift."""
    import math

    import numpy as np

    from comfy_research.engine.trainer.observable_config import (
        _EMBEDDING_OBSERVABLE_MODEL_TYPES,
        _obs_encoder_layer_mode,
    )
    from comfy_research.engine.trainer.observable_metrics import _layer_bucket_flat_concat_vectors

    embedding_prev_for_drift = rec.embedding_prev_for_drift
    embedding_prev_layer_flat_for_drift = rec.embedding_prev_layer_flat_for_drift
    encoder_obs_layer_canon = rec.encoder_obs_layer_canon
    model = rec.model
    observable_metric_histories = rec.observable_metric_histories
    od_fd: dict[str, Any] = on.data or {}
    if _obs_encoder_layer_mode(od_fd) == "all_layers":
        flats_fd = _layer_bucket_flat_concat_vectors(model)
        drift_fd: dict[str, float] = {}
        for seg, vec in flats_fd.items():
            keyp = (on.id, seg)
            prev_l = embedding_prev_layer_flat_for_drift.get(keyp)
            if prev_l is None or prev_l.shape != vec.shape:
                drift_fd[seg] = float("nan")
            else:
                pn = float(np.linalg.norm(prev_l))
                cn = float(np.linalg.norm(vec))
                if pn < 1e-12 or cn < 1e-12:
                    drift_fd[seg] = float("nan")
                else:
                    cos = float(np.dot(prev_l, vec) / (pn * cn))
                    cos = max(-1.0, min(1.0, cos))
                    drift_fd[seg] = 1.0 - cos
            embedding_prev_layer_flat_for_drift[keyp] = vec.copy()
        if not drift_fd:
            observable_metric_histories[on.id].append(float("nan"))
            canon_fd = encoder_obs_layer_canon.setdefault(on.id, [])
            glen_fd0 = len(observable_metric_histories[on.id])
            for seg in canon_fd:
                rk_fd = f"{on.id}::layer::{seg}"
                observable_metric_histories.setdefault(rk_fd, [])
                row_fd = observable_metric_histories[rk_fd]
                while len(row_fd) < glen_fd0 - 1:
                    row_fd.append(float("nan"))
                row_fd.append(float("nan"))
        else:
            keys_fd = sorted(drift_fd.keys(), key=lambda s: int(s) if s.isdigit() else 0)
            finite_fd = [float(drift_fd[k]) for k in keys_fd if math.isfinite(float(drift_fd[k]))]
            mean_fd = float(sum(finite_fd) / len(finite_fd)) if finite_fd else float("nan")
            observable_metric_histories[on.id].append(mean_fd)
            canon_fd = encoder_obs_layer_canon.setdefault(on.id, [])
            if not canon_fd:
                canon_fd.extend(keys_fd)
            glen_fd = len(observable_metric_histories[on.id])
            for seg in canon_fd:
                v_fd = float(drift_fd[seg]) if seg in drift_fd else float("nan")
                rk_fd = f"{on.id}::layer::{seg}"
                observable_metric_histories.setdefault(rk_fd, [])
                row_fd = observable_metric_histories[rk_fd]
                while len(row_fd) < glen_fd - 1:
                    row_fd.append(float("nan"))
                row_fd.append(v_fd)
    else:
        if not isinstance(model, _EMBEDDING_OBSERVABLE_MODEL_TYPES):
            observable_metric_histories[on.id].append(float("nan"))
            return
        arrays_fd = model.observable_numpy_arrays()
        emb_fd = arrays_fd.get("embedding")
        if emb_fd is None:
            observable_metric_histories[on.id].append(float("nan"))
            return
        flat_fd = emb_fd.reshape(-1).astype(np.float64, copy=False)
        prev_fd = embedding_prev_for_drift.get(on.id)
        if prev_fd is None or prev_fd.shape != flat_fd.shape:
            observable_metric_histories[on.id].append(float("nan"))
        else:
            pn = float(np.linalg.norm(prev_fd))
            cn = float(np.linalg.norm(flat_fd))
            if pn < 1e-12 or cn < 1e-12:
                observable_metric_histories[on.id].append(float("nan"))
            else:
                cos = float(np.dot(prev_fd, flat_fd) / (pn * cn))
                cos = max(-1.0, min(1.0, cos))
                observable_metric_histories[on.id].append(1.0 - cos)
        embedding_prev_for_drift[on.id] = flat_fd.copy()
