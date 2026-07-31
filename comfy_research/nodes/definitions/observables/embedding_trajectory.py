"""observable_embedding_trajectory — NodeDef-channel definition + recorder.

The generic component skips enum fields without options. The dedicated
embedding_trajectory spawn resolves observableName via title_from_field and
sets showTrails/showPoints defaults. ``observable_viz.py`` shapes the
embedding_history stream by node type.
Records into rec.observable_embedding_histories.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from comfy_research.nodes.registry import observable_def, recorder_for
from comfy_research.nodes.schema import EnumField, FrontendSpec, ObservableDef, SpawnSpec, VizSpec

if TYPE_CHECKING:  # pragma: no cover
    from comfy_research.engine.trainer.recorder import ObservableRecorder
    from comfy_research.schemas.graph import Node

EMBEDDING_TRAJECTORY = observable_def(
    ObservableDef(
        type="observable_embedding_trajectory",
        label="Embedding trajectory",
        hint="Logs per-token embedding vectors over training for trajectory plotting.",
        viz=VizSpec(
            variant="embedding_trajectory",
            title="Embedding trajectory",
            info_markdown=(
                "**Embedding trajectory** — low-dimensional projection / path of embedding "
                "statistics over training (paired viz shows curves)."
            ),
            spawns=True,
            user_whitelisted=True,
            spawn=SpawnSpec(kind="embedding_trajectory", title_from_field="label"),
        ),
        fields=(EnumField(key="label", label="Label", default="Embedding trajectory"),),
        frontend=FrontendSpec(),
    )
)


@recorder_for(EMBEDDING_TRAJECTORY)
def record(rec: "ObservableRecorder", on: "Node") -> None:
    """Body moved verbatim from ObservableRecorder._record_embedding_trajectory."""
    from comfy_research.engine.analysis.token_lm_observable_support import TOKEN_LM_EMBEDDING_OBSERVABLE_MODULES

    model = rec.model
    observable_embedding_histories = rec.observable_embedding_histories
    if not isinstance(model, TOKEN_LM_EMBEDDING_OBSERVABLE_MODULES):
        return
    arrays = model.observable_numpy_arrays()
    emb = arrays.get("embedding")
    if emb is None:
        return
    observable_embedding_histories[on.id].append(emb.tolist())
