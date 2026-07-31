"""observable_user — NodeDef-channel definition + recorder.

Custom component adapter (ObservableUserNode: async /api/user-observables
describe-path fetch, anchor-wait loop). The generic add path excludes this type
because Observables-panel drag-and-drop passes options into its defaults.

The recorder receives ``get_user_observable_record`` through dependency
injection to preserve the package boundary with ``api.user_observables``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from comfy_research.nodes.registry import observable_def, recorder_for
from comfy_research.nodes.schema import EnumField, FrontendSpec, InPort, ObservableDef, PortAccept, SpawnSpec, VizSpec

if TYPE_CHECKING:  # pragma: no cover
    from comfy_research.engine.trainer.recorder import ObservableRecorder
    from comfy_research.schemas.graph import Node

OBSERVABLE_USER = observable_def(
    ObservableDef(
        type="observable_user",
        label="User observable",
        hint="User-defined scalar from tensor path (saved definitions).",
        viz=VizSpec(
            variant="user",
            title="User observable",
            info_markdown=(
                "**User observable** — runs your registered training-eval code over a tensor path "
                "from the graph; the scalar returned is logged each step when wired to the Trainer."
            ),
            spawns=True,
            user_whitelisted=True,
            spawn=SpawnSpec(kind="user_scalar", title_from_field="label", unit="Value"),
        ),
        fields=(
            EnumField(key="userObservableId", label="User Observable Id", default=""),
            EnumField(key="label", label="Label", default="User observable"),
            EnumField(key="tensorVizNodeId", label="Tensor Viz Node Id", default=""),
            EnumField(key="tensorSelectorNodeId", label="Tensor Selector Node Id", default=""),
        ),
        # 原 cascade 分支:observable_in ← tensor_viz_0d.observable。
        ports=(InPort(id="observable_in", accepts=(PortAccept(handles=("observable",), source_type="tensor_viz_0d"),)),),
        frontend=FrontendSpec(component_key="ObservableUserNode"),
    )
)


@recorder_for(OBSERVABLE_USER)
def record(rec: "ObservableRecorder", on: "Node") -> None:
    """Body moved verbatim from ObservableRecorder._record_observable_user."""
    import torch

    from comfy_research.engine.analysis.observable_user_eval import (
        eval_algebra_observable_user,
        eval_observable_user_mean_abs,
    )
    from comfy_research.engine.trainer.user_observable_helpers import (
        _user_observable_algebra_rec,
        _user_observable_definition_code,
        _user_observable_path_anchor,
    )

    algebra_tensor_member_canon = rec.algebra_tensor_member_canon
    depth = rec.depth
    edges = rec.edges
    get_user_observable_record = rec.get_user_observable_record
    loss_history = rec.loss_history
    model = rec.model
    nodes = rec.nodes
    observable_metric_histories = rec.observable_metric_histories
    test_loss_history = rec.test_loss_history
    xr = rec._xr
    od_u: dict[str, Any] = on.data or {}
    algebra_rec = _user_observable_algebra_rec(od_u, get_user_observable_record=get_user_observable_record)
    def_code = _user_observable_definition_code(od_u, get_user_observable_record=get_user_observable_record)
    anchor = _user_observable_path_anchor(od_u, get_user_observable_record=get_user_observable_record)
    if algebra_rec is not None and (
        str(algebra_rec.tensor_scope or "single") == "all_matching"
        or str(algebra_rec.observable_source or "weight") == "representation"
    ):
        with torch.no_grad():
            rep_tensors_arg = (
                rec._representation_tensors_for_log()
                if str(algebra_rec.observable_source or "weight").strip().lower() == "representation"
                else None
            )
            primary, members = eval_algebra_observable_user(
                tensor_name=str(algebra_rec.tensor_name or ""),
                tensor_scope=str(algebra_rec.tensor_scope or "single"),
                reductions_raw=list(algebra_rec.reductions or []),
                definition_code=def_code,
                model=model,
                x=xr,
                depth=depth,
                flatten_mode=str(algebra_rec.flatten_mode or "none"),
                observable_source=str(algebra_rec.observable_source or "weight"),
                representation_id=str(algebra_rec.representation_id or ""),
                loss_history=loss_history,
                test_loss_history=test_loss_history,
                observable_metric_histories=observable_metric_histories,
                representation_tensors=rep_tensors_arg,
            )
        observable_metric_histories[on.id].append(primary)
        if members:
            from comfy_research.engine.analysis.observable_algebra import member_storage_key

            canon_alg = algebra_tensor_member_canon.setdefault(on.id, [])
            if not canon_alg:
                canon_alg.extend(sorted(members.keys()))
            glen_alg = len(observable_metric_histories[on.id])
            for tname in canon_alg:
                rk_alg = f"{on.id}::member::{member_storage_key(tname)}"
                observable_metric_histories.setdefault(rk_alg, [])
                row_alg = observable_metric_histories[rk_alg]
                while len(row_alg) < glen_alg - 1:
                    row_alg.append(float("nan"))
                row_alg.append(float(members.get(tname, float("nan"))))
    elif not def_code and not anchor:
        observable_metric_histories[on.id].append(float("nan"))
    else:
        with torch.no_grad():
            v_u = eval_observable_user_mean_abs(
                nodes,
                edges,
                anchor,
                model,
                xr,
                depth,
                loss_history=loss_history,
                test_loss_history=test_loss_history,
                observable_metric_histories=observable_metric_histories,
                definition_code=def_code if def_code else None,
                representation_tensors=rec._rep_tensors_cache,
            )
        observable_metric_histories[on.id].append(v_u)
