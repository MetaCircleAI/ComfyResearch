"""observable_embedding_evolution — NodeDef-channel definition + recorder.

The generic component skips the defaults-inferred ``label`` enum because it
has no options. The user_scalar spawn derives its title from ``data.label``.
``ObservableRecorder.__init__`` eagerly computes ``attention_arrays_init`` so
the provider can read the t=0 baseline.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from comfy_research.nodes.registry import observable_def, recorder_for
from comfy_research.nodes.schema import EnumField, FrontendSpec, ObservableDef, SpawnSpec, VizSpec

if TYPE_CHECKING:  # pragma: no cover
    from comfy_research.engine.trainer.recorder import ObservableRecorder
    from comfy_research.schemas.graph import Node

EMBEDDING_EVOLUTION = observable_def(
    ObservableDef(
        type="observable_embedding_evolution",
        label="Embedding evolution",
        hint="Relative embedding drift over training (||E_t-E_0|| / ||E_0||).",
        viz=VizSpec(
            variant="user",
            title="Embedding evolution",
            info_markdown=(
                "**Embedding evolution** — relative drift \\(\\|E_t - E_0\\| / \\|E_0\\|\\) of the "
                "token embedding matrix across training steps."
            ),
            spawns=True,
            user_whitelisted=True,
            spawn=SpawnSpec(kind="user_scalar", title_from_field="label", unit="value"),
        ),
        fields=(EnumField(key="label", label="Label", default="Embedding evolution"),),
        frontend=FrontendSpec(),
    )
)


@recorder_for(EMBEDDING_EVOLUTION)
def record(rec: "ObservableRecorder", on: "Node") -> None:
    """Body moved verbatim from ObservableRecorder._record_embedding_evolution."""
    import numpy as np

    from comfy_research.engine.analysis.token_lm_observable_support import TOKEN_LM_EMBEDDING_OBSERVABLE_MODULES

    attention_arrays_init = rec.attention_arrays_init
    model = rec.model
    observable_metric_histories = rec.observable_metric_histories
    if not isinstance(model, TOKEN_LM_EMBEDDING_OBSERVABLE_MODULES):
        observable_metric_histories[on.id].append(float("nan"))
        return
    arrays = model.observable_numpy_arrays()
    emb = arrays.get("embedding")
    emb0 = (attention_arrays_init or {}).get("embedding")
    if emb is None or emb0 is None or emb.shape != emb0.shape:
        observable_metric_histories[on.id].append(float("nan"))
        return
    num = float(np.linalg.norm(emb - emb0))
    den = float(np.linalg.norm(emb0)) + 1e-12
    observable_metric_histories[on.id].append(num / den)
