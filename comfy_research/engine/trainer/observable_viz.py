"""Frontend-facing observable visualization endpoints (extracted from trainer_run)."""

from typing import Any, Literal

from comfy_research.engine.trainer.graph import _node_map
from comfy_research.engine.trainer.observable_config import (
    OBS_ENCODER_LAYER_SERIES_NODEKINDS,
    _activation_stats_layer_mode,
    _obs_encoder_layer_mode,
    _observable_l2_aggregation,
    _sink_attention_mass_layer_mode,
)
from comfy_research.engine.trainer.observable_metrics import (
    _activation_stats_layer_series_payload_from_hist,
    _observable_multi_series_l2_payload_from_hist,
    _paired_layer_series_payload_from_hist,
    _paired_member_series_payload_from_hist,
)
from comfy_research.engine.trainer.scalar import _scalar_int
from comfy_research.nodes.registry import all_observable_defs
from comfy_research.schemas.graph import Edge, Node, NodeKind


def find_loss_visualization_targets(
    edges: list[Edge], nodes: list[Node], trainer_node_id: str
) -> list[str]:
    nmap = _node_map(nodes)
    out: list[str] = []
    for e in edges:
        if e.source != trainer_node_id:
            continue
        if e.sourceHandle != "loss_results":
            continue
        if e.targetHandle not in ("tensor_list", "tensor"):
            continue
        tgt = nmap.get(e.target)
        if tgt and tgt.type == "training_visualization":
            out.append(tgt.id)
    return out

def _trainer_observable_input_handle_ok(th: str | None) -> bool:
    """Accept legacy graphs that omit handles or used singular ``observable``."""
    if th is None or th == "":
        return True
    return th in ("observables", "observable")


def _observable_connected_to_trainer(
    edges: list[Edge], trainer_id: str, observable_id: str
) -> bool:
    for e in edges:
        if e.target == trainer_id and _trainer_observable_input_handle_ok(e.targetHandle) and e.source == observable_id:
            return True
    return False


_LEGACY_VIZ_TO_OBS: dict[str, str] = {
    "observable_viz_weight_l2": "observable_weight_l2",
    "observable_viz_weight_l1": "observable_weight_l1",
    "observable_viz_relu_nonlinear": "observable_relu_nonlinear_count",
    "observable_viz_embedding_trajectory": "observable_embedding_trajectory",
    "observable_viz_neuron_trajectory_2d": "observable_neuron_trajectory_2d",
}


_VARIANT_TO_OBS: dict[str, str] = {}


def _coerce_node_type_value(tp: Any) -> str:
    if isinstance(tp, str):
        return tp
    v = getattr(tp, "value", None)
    return str(v) if v is not None else str(tp)


def _default_observable_viz_variant_for_obs(obs_type: Any) -> str | None:
    """Infer ``vizVariant`` when missing (older graphs); keep in sync with frontend spawn / ``ObservableVizNode``."""
    key = _coerce_node_type_value(obs_type)
    return _VIZ_VARIANT_FOR_OBS_WITH_DEFS.get(key)


# Unified from the two inline user-variant whitelists in
# observable_viz_metric_updates; they are now equal by construction.
_USER_VARIANT_OBSERVABLE_KINDS: frozenset[str] = frozenset(
    {
    }
)


# NodeDef 通道并集；kan_reg 接入后手写侧仅保留空的 _VARIANT_TO_OBS shim。
_DEFS_VARIANT_FOR_OBS: dict[str, str] = {d.type: d.viz.variant for d in all_observable_defs()}
_VIZ_VARIANT_FOR_OBS_WITH_DEFS: dict[str, str] = dict(_DEFS_VARIANT_FOR_OBS)
_VARIANT_TO_OBS_WITH_DEFS: dict[str, str] = {**_VARIANT_TO_OBS, **{v: k for k, v in _DEFS_VARIANT_FOR_OBS.items()}}
_USER_KINDS_WITH_DEFS: frozenset[str] = _USER_VARIANT_OBSERVABLE_KINDS | frozenset(
    d.type for d in all_observable_defs() if d.viz.user_whitelisted
)


def observable_user_variant_kinds() -> frozenset[str]:
    """Runtime source of truth for sync guards, replacing AST literal scans."""
    return _USER_KINDS_WITH_DEFS


def observable_viz_variant_map() -> dict[str, str]:
    """Runtime source of truth for sync guards, replacing AST literal scans."""
    return dict(_VIZ_VARIANT_FOR_OBS_WITH_DEFS)


def _obs_viz_stream_row(row: dict[str, Any], paired_observable_id: str) -> dict[str, Any]:
    """Client can match updates by viz id or by ``paired_observable_id`` (see ``applyTrainerVizPayload``)."""
    out_row = dict(row)
    out_row["paired_observable_id"] = paired_observable_id
    return out_row


def observable_viz_metric_updates(
    edges: list[Edge],
    nmap: dict[str, Node],
    trainer_node_id: str,
    metric_histories: dict[str, list[float]],
    embedding_histories: dict[str, list[list[list[float]]]],
    attention_slice_histories: dict[str, list[dict[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
    """Build per-viz payloads for weight norm dynamics (same step_ticks as loss)."""
    has_viz_target = any(
        e.source == trainer_node_id
        and nmap.get(e.target) is not None
        and (
            nmap[e.target].type in (NodeKind.observable_viz, "observable_viz_user")
            or nmap[e.target].type in _LEGACY_VIZ_TO_OBS
        )
        for e in edges
    )
    if not has_viz_target:
        result_edges = list(edges)
        result_nmap = dict(nmap)
        for e in edges:
            if e.target != trainer_node_id or not _trainer_observable_input_handle_ok(e.targetHandle):
                continue
            obs = nmap.get(e.source)
            if obs is None:
                continue
            variant = _default_observable_viz_variant_for_obs(obs.type)
            if variant is None:
                continue
            result_id = f"{trainer_node_id}::__observable_result__{obs.id}"
            result_nmap[result_id] = Node(
                id=result_id,
                type=NodeKind.observable_viz,
                data={"pairedObservableId": obs.id, "vizVariant": variant},
            )
            result_edges.append(
                Edge(
                    id=f"{result_id}::__edge",
                    source=trainer_node_id,
                    target=result_id,
                    sourceHandle="observable_results",
                    targetHandle="tensor",
                )
            )
        if len(result_edges) != len(edges):
            return observable_viz_metric_updates(
                result_edges,
                result_nmap,
                trainer_node_id,
                metric_histories,
                embedding_histories,
                attention_slice_histories,
            )

    out: list[dict[str, Any]] = []
    attention_slice_histories = attention_slice_histories or {}
    for e in edges:
        if e.source != trainer_node_id:
            continue
        sh_e = e.sourceHandle
        if sh_e is not None and sh_e != "" and sh_e != "observable_results":
            continue
        th_e = e.targetHandle
        if th_e is not None and th_e != "" and th_e != "tensor":
            continue
        viz = nmap.get(e.target)
        if viz is None:
            continue
        vd: dict[str, Any] = viz.data or {}
        paired = vd.get("pairedObservableId")
        if not isinstance(paired, str) or not paired:
            continue
        if not _observable_connected_to_trainer(edges, trainer_node_id, paired):
            continue
        obs = nmap.get(paired)
        if obs is None:
            continue

        if viz.type == "observable_viz":
            variant_raw = vd.get("vizVariant")
            if isinstance(variant_raw, str) and variant_raw.strip():
                variant = variant_raw.strip()
            else:
                inferred = _default_observable_viz_variant_for_obs(obs.type)
                if inferred is None:
                    continue
                variant = inferred
            if variant == "user":
                if obs.type not in _USER_KINDS_WITH_DEFS:
                    continue
            elif variant in ("embedding_trajectory", "neuron_trajectory_2d"):
                if variant == "embedding_trajectory" and obs.type != "observable_embedding_trajectory":
                    continue
                if variant == "neuron_trajectory_2d" and obs.type != "observable_neuron_trajectory_2d":
                    continue
            else:
                expected = _VARIANT_TO_OBS_WITH_DEFS.get(variant)
                if not expected or obs.type != expected:
                    continue
        elif viz.type == "observable_viz_user":
            if obs.type not in _USER_KINDS_WITH_DEFS:
                continue
        else:
            obs_type = _LEGACY_VIZ_TO_OBS.get(viz.type)
            if obs_type is None:
                continue
            if obs.type != obs_type:
                continue

        if obs.type == "observable_attention_map":
            attention_frames = attention_slice_histories.get(paired)
            if not attention_frames:
                continue
            out.append(
                _obs_viz_stream_row(
                    {
                        "node_id": viz.id,
                        "attention_map_frames": attention_frames,
                    },
                    paired,
                )
            )
        elif obs.type in (
            "observable_embedding_trajectory",
            "observable_neuron_trajectory_2d",
            "observable_information_plane",
        ):
            emb_hist = embedding_histories.get(paired)
            if not emb_hist:
                continue
            out.append(_obs_viz_stream_row({"node_id": viz.id, "embedding_history": emb_hist}, paired))
        elif obs.type == NodeKind.observable_hessian_eigenvalues:
            od_h: dict[str, Any] = obs.data or {}
            top_k = max(1, _scalar_int(od_h.get("topK"), 5))
            value_histories: list[list[float]] = []
            for i in range(top_k):
                rank_key = f"{paired}::{i}"
                series = metric_histories.get(rank_key)
                if not series:
                    break
                value_histories.append([float(x) for x in series])
            if value_histories:
                out.append(_obs_viz_stream_row({"node_id": viz.id, "value_histories": value_histories}, paired))
        elif obs.type == "observable_weight_product_sv":
            od_sv: dict[str, Any] = obs.data or {}
            top_k_sv = max(1, _scalar_int(od_sv.get("topK"), 3))
            sv_histories: list[list[float]] = []
            for i in range(top_k_sv):
                rank_key = f"{paired}::{i}"
                series = metric_histories.get(rank_key)
                if not series:
                    break
                sv_histories.append([float(x) for x in series])
            if sv_histories:
                out.append(_obs_viz_stream_row({"node_id": viz.id, "value_histories": sv_histories}, paired))
        elif obs.type == "observable_layer_spectral_norm":
            histories = [series for key, series in sorted(metric_histories.items()) if key.startswith(f"{paired}::layer::") and series]
            if histories:
                out.append(_obs_viz_stream_row({"node_id": viz.id, "value_histories": [[float(x) for x in series] for series in histories], "series_labels": [f"Linear layer {index}" for index in range(1, len(histories) + 1)]}, paired))
        elif obs.type == NodeKind.observable_gradient_norm:
            od_gn: dict[str, Any] = obs.data or {}
            hist_g = metric_histories.get(paired)
            if not hist_g:
                continue
            agg_v = _observable_l2_aggregation(od_gn)
            if agg_v == "global":
                out.append(
                    _obs_viz_stream_row(
                        {
                            "node_id": viz.id,
                            "value_histories": [[float(x) for x in hist_g]],
                            "series_labels": ["global"],
                        },
                        paired,
                    )
                )
            else:
                br: Literal["top_level_module", "tensor"] = (
                    "top_level_module" if agg_v == "top_level_module" else "tensor"
                )
                payload = _observable_multi_series_l2_payload_from_hist(paired, metric_histories, breakdown=br)
                if payload:
                    out.append(_obs_viz_stream_row({"node_id": viz.id, **payload}, paired))
                else:
                    out.append(
                        _obs_viz_stream_row(
                            {
                                "node_id": viz.id,
                                "value_histories": [[float(x) for x in hist_g]],
                                "series_labels": ["global"],
                            },
                            paired,
                        )
                    )
        elif obs.type == NodeKind.observable_weight_l2:
            od_w: dict[str, Any] = obs.data or {}
            hist_w = metric_histories.get(paired)
            if not hist_w:
                continue
            agg_w = _observable_l2_aggregation(od_w)
            if agg_w == "global":
                out.append(
                    _obs_viz_stream_row(
                        {
                            "node_id": viz.id,
                            "value_histories": [[float(x) for x in hist_w]],
                            "series_labels": ["global"],
                        },
                        paired,
                    )
                )
            else:
                brw: Literal["top_level_module", "tensor"] = (
                    "top_level_module" if agg_w == "top_level_module" else "tensor"
                )
                payload_w = _observable_multi_series_l2_payload_from_hist(paired, metric_histories, breakdown=brw)
                if payload_w:
                    out.append(_obs_viz_stream_row({"node_id": viz.id, **payload_w}, paired))
                else:
                    out.append(
                        _obs_viz_stream_row(
                            {
                                "node_id": viz.id,
                                "value_histories": [[float(x) for x in hist_w]],
                                "series_labels": ["global"],
                            },
                            paired,
                        )
                    )
        elif obs.type in OBS_ENCODER_LAYER_SERIES_NODEKINDS:
            od_enc: dict[str, Any] = obs.data or {}
            hist_enc = metric_histories.get(paired)
            if not hist_enc:
                continue
            if _obs_encoder_layer_mode(od_enc) == "global":
                out.append(
                    _obs_viz_stream_row({"node_id": viz.id, "value_history": [float(x) for x in hist_enc]}, paired)
                )
            else:
                pay_enc = _paired_layer_series_payload_from_hist(paired, metric_histories)
                if pay_enc:
                    out.append(_obs_viz_stream_row({"node_id": viz.id, **pay_enc}, paired))
                else:
                    out.append(
                        _obs_viz_stream_row(
                            {"node_id": viz.id, "value_history": [float(x) for x in hist_enc]},
                            paired,
                        )
                    )
        elif obs.type == NodeKind.observable_activation_stats:
            od_ast: dict[str, Any] = obs.data or {}
            mean_hist = metric_histories.get(paired)
            std_hist = metric_histories.get(f"{paired}::std")
            if (
                not mean_hist
                or not std_hist
                or len(mean_hist) != len(std_hist)
                or len(mean_hist) <= 0
            ):
                continue
            if _activation_stats_layer_mode(od_ast) == "global":
                out.append(
                    _obs_viz_stream_row(
                        {
                            "node_id": viz.id,
                            "value_histories": [
                                [float(x) for x in mean_hist],
                                [float(x) for x in std_hist],
                            ],
                            "series_labels": ["mean", "std"],
                        },
                        paired,
                    )
                )
            else:
                payload_ast = _activation_stats_layer_series_payload_from_hist(paired, metric_histories)
                if payload_ast:
                    out.append(_obs_viz_stream_row({"node_id": viz.id, **payload_ast}, paired))
                else:
                    out.append(
                        _obs_viz_stream_row(
                            {
                                "node_id": viz.id,
                                "value_histories": [
                                    [float(x) for x in mean_hist],
                                    [float(x) for x in std_hist],
                                ],
                                "series_labels": ["mean", "std"],
                            },
                            paired,
                        )
                    )
        elif obs.type == "observable_accuracy":
            hist = metric_histories.get(paired)
            if not hist:
                continue
            test_key = f"{paired}::test"
            tseries = metric_histories.get(test_key) or []
            if len(tseries) == len(hist) and tseries:
                out.append(
                    _obs_viz_stream_row(
                        {
                            "node_id": viz.id,
                            "value_history": [float(x) for x in hist],
                            "test_value_history": [float(x) for x in tseries],
                        },
                        paired,
                    )
                )
            else:
                out.append(
                    _obs_viz_stream_row({"node_id": viz.id, "value_history": [float(x) for x in hist]}, paired)
                )
        elif obs.type == "observable_attention_relation_score":
            from comfy_research.engine.trainer.attention_relation_metrics import (
                attention_relation_pair_key,
                attention_relation_pair_label,
                attention_relation_pairs,
            )
            pairs = attention_relation_pairs(obs.data or {})
            series = [metric_histories.get(attention_relation_pair_key(paired, index), []) for index in range(len(pairs))]
            if not series or any(not history for history in series):
                continue
            out.append(_obs_viz_stream_row({
                "node_id": viz.id,
                "value_histories": [[float(x) for x in history] for history in series],
                "series_labels": [attention_relation_pair_label(layer, head) for layer, head in pairs],
            }, paired))
        elif obs.type == "observable_user":
            hist = metric_histories.get(paired)
            if not hist:
                continue
            payload_u = _paired_member_series_payload_from_hist(paired, metric_histories)
            if payload_u:
                out.append(_obs_viz_stream_row({"node_id": viz.id, **payload_u}, paired))
            else:
                out.append(_obs_viz_stream_row({"node_id": viz.id, "value_history": [float(x) for x in hist]}, paired))
        elif obs.type == NodeKind.observable_sink_attention_mass:
            od_sm: dict[str, Any] = obs.data or {}
            hist_sm = metric_histories.get(paired)
            if not hist_sm:
                continue
            if _sink_attention_mass_layer_mode(od_sm) == "global":
                out.append(
                    _obs_viz_stream_row(
                        {"node_id": viz.id, "value_history": [float(x) for x in hist_sm]},
                        paired,
                    )
                )
            else:
                payload_sm = _paired_layer_series_payload_from_hist(paired, metric_histories)
                if payload_sm:
                    out.append(_obs_viz_stream_row({"node_id": viz.id, **payload_sm}, paired))
                else:
                    out.append(
                        _obs_viz_stream_row(
                            {"node_id": viz.id, "value_history": [float(x) for x in hist_sm]},
                            paired,
                        )
                    )
        else:
            hist = metric_histories.get(paired)
            if not hist:
                continue
            out.append(_obs_viz_stream_row({"node_id": viz.id, "value_history": hist}, paired))
    return out
