"""observable_weight_product_sv — NodeDef-channel definition + recorder.

The canvas renders this node through ``GenericObservableNode`` using the
generated spec fields.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from comfy_research.nodes.registry import observable_def, recorder_for
from comfy_research.nodes.schema import FrontendSpec, IntField, ObservableDef, SpawnSpec, VizSpec

if TYPE_CHECKING:  # pragma: no cover
    from comfy_research.engine.trainer.recorder import ObservableRecorder
    from comfy_research.schemas.graph import Node

WEIGHT_PRODUCT_SV = observable_def(
    ObservableDef(
        type="observable_weight_product_sv",
        label="Weight product SV",
        hint="Logs top-k singular values of the effective weight product W_eff = W_n … W_1 at each log step.",
        viz=VizSpec(
            variant="weight_product_sv",
            title="Weight product SV",
            info_markdown=(
                "**Weight product SV** — top-k singular values of the effective weight product "
                "W_eff = W_n ⋯ W_1 (all .weight tensors multiplied right-to-left). Use with deep linear "
                "networks to observe staggered singular-value learning (Saxe et al. 2013)."
            ),
            spawns=True,
            spawn=SpawnSpec(kind="hessian_topk", top_k_from_field="topK"),
        ),
        fields=(IntField(key="topK", label="Top K", default=3, min=1, step=1),),
        frontend=FrontendSpec(),
    )
)


@recorder_for(WEIGHT_PRODUCT_SV)
def record(rec: "ObservableRecorder", on: "Node") -> None:
    """Return the top-k singular values of W_eff = W_n ⋯ W_1."""
    import torch

    from comfy_research.engine.trainer.scalar import _scalar_int

    model = rec.model
    observable_metric_histories = rec.observable_metric_histories
    od_sv: dict[str, Any] = on.data or {}
    top_k_sv = max(1, _scalar_int(od_sv.get("topK"), 3))
    weight_matrices: list[torch.Tensor] = [
        p.detach().cpu().float()
        for name, p in model.named_parameters()
        if name.endswith(".weight")
    ]
    if weight_matrices:
        try:
            W_eff = weight_matrices[0]
            for W in weight_matrices[1:]:
                W_eff = W @ W_eff
            sv_sorted = torch.linalg.svdvals(W_eff).sort(descending=True).values
        except Exception:
            sv_sorted = torch.zeros(0)
    else:
        sv_sorted = torch.zeros(0)
    primary_sv = float(sv_sorted[0].item()) if sv_sorted.numel() > 0 else float("nan")
    observable_metric_histories[on.id].append(primary_sv)
    glen_sv = len(observable_metric_histories[on.id])
    for i in range(top_k_sv):
        sv_val = float(sv_sorted[i].item()) if i < sv_sorted.numel() else float("nan")
        rank_key = f"{on.id}::{i}"
        observable_metric_histories.setdefault(rank_key, [])
        sv_row = observable_metric_histories[rank_key]
        while len(sv_row) < glen_sv - 1:
            sv_row.append(float("nan"))
        sv_row.append(sv_val)
