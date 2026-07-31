"""Coordinate-descent hyperparameter tuning against a target training curve."""

from __future__ import annotations

import json
import math
from typing import Any, Iterator, Literal

from fastapi import HTTPException
from pydantic import BaseModel, Field, model_validator

from comfy_research.engine.runs.sweep_control import (
    is_sweep_abort_requested,
    register_sweep_session,
    unregister_sweep_session,
)
from comfy_research.engine.runs.train_sweep import (
    SweepAxis,
    _apply_axes_to_nodes,
    _deep_copy_nodes,
    _parse_data_path,
    _trainer_training_steps_override,
)
from comfy_research.engine.runs.trainer_run import iter_trainer_events_from_context, prepare_trainer_run
from comfy_research.schemas.graph import Edge, Node

MAX_COORDINATE_DESCENT_ROUNDS = 24
MAX_AXIS_VALUES = 128
MAX_CURVE_POINTS_RESPONSE = 1024

ObjectiveKind = Literal["closest_to_target"]


class CurveSimilarityConfig(BaseModel):
    objective: ObjectiveKind = "closest_to_target"
    end_weight: float = Field(
        default=1.0,
        ge=1.0,
        le=10.0,
        description="Optional linear emphasis toward later training steps.",
    )
    smoothness_weight: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Low-weight penalty for jagged candidate curves.",
    )


class TrainCoordinateDescentRequest(BaseModel):
    trainer_node_id: str = Field(min_length=1)
    nodes: list[Node] = Field(default_factory=list)
    edges: list[Edge] = Field(default_factory=list)
    axes: list[SweepAxis] = Field(min_length=1)
    target_step_ticks: list[float] = Field(min_length=2)
    target_loss_history: list[float] = Field(min_length=2)
    session_id: str = Field(min_length=1, description="Client-generated id for abort between runs.")
    max_rounds: int = Field(default=3, ge=1, le=MAX_COORDINATE_DESCENT_ROUNDS)
    training_steps_override: int | None = Field(default=None, ge=1)
    score: CurveSimilarityConfig = Field(default_factory=CurveSimilarityConfig)
    min_improvement: float = Field(default=1e-9, ge=0.0)

    @model_validator(mode="after")
    def _target_shapes_match(self) -> TrainCoordinateDescentRequest:
        if len(self.target_step_ticks) != len(self.target_loss_history):
            raise ValueError("target_step_ticks and target_loss_history must have the same length.")
        if len(self.target_step_ticks) > 4096:
            raise ValueError("target curve has too many points (max 4096).")
        return self


def _safe_float(v: Any) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x):
        return None
    return x


def _prepare_curve(step_ticks: list[Any], values: list[Any]) -> tuple[list[float], list[float]]:
    xs: list[float] = []
    ys: list[float] = []
    for raw_x, raw_y in zip(step_ticks, values):
        x = _safe_float(raw_x)
        y = _safe_float(raw_y)
        if x is None or y is None:
            continue
        xs.append(x)
        ys.append(y)
    if len(xs) < 2:
        raise ValueError("curve must contain at least 2 finite points.")
    pairs = sorted(zip(xs, ys), key=lambda p: p[0])
    out_x: list[float] = []
    out_y: list[float] = []
    for x, y in pairs:
        if out_x and abs(x - out_x[-1]) < 1e-12:
            out_y[-1] = y
            continue
        out_x.append(x)
        out_y.append(y)
    if len(out_x) < 2:
        raise ValueError("curve must span at least 2 distinct x positions.")
    return out_x, out_y


def _interp_linear(xs: list[float], ys: list[float], xq: float) -> float:
    if xq <= xs[0]:
        return ys[0]
    if xq >= xs[-1]:
        return ys[-1]
    lo = 0
    hi = len(xs) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if xs[mid] <= xq:
            lo = mid
        else:
            hi = mid
    x0 = xs[lo]
    x1 = xs[hi]
    if abs(x1 - x0) < 1e-12:
        return ys[lo]
    t = (xq - x0) / (x1 - x0)
    return ys[lo] + (ys[hi] - ys[lo]) * t


def _curve_roughness(vals: list[float]) -> float:
    if len(vals) < 3:
        return 0.0
    acc = 0.0
    n = 0
    for i in range(1, len(vals) - 1):
        second = vals[i + 1] - 2 * vals[i] + vals[i - 1]
        acc += abs(second)
        n += 1
    return acc / max(1, n)


def _score_curve(
    source_steps: list[Any],
    source_values: list[Any],
    target_steps: list[float],
    target_values: list[float],
    cfg: CurveSimilarityConfig,
) -> tuple[float, float, float]:
    sx, sy = _prepare_curve(source_steps, source_values)
    interp = [_interp_linear(sx, sy, xt) for xt in target_steps]
    n = len(interp)
    if n < 1:
        return float("nan"), float("nan"), float("nan")
    mse_acc = 0.0
    wsum = 0.0
    for i, (sv, tv) in enumerate(zip(interp, target_values)):
        ramp = i / max(1, n - 1)
        w = 1.0 + (cfg.end_weight - 1.0) * ramp
        diff = sv - tv
        mse_acc += w * diff * diff
        wsum += w
    mse = mse_acc / max(1e-12, wsum)
    final_abs = abs(interp[-1] - target_values[-1])
    rough = _curve_roughness(interp)
    score = mse + (cfg.smoothness_weight * rough)
    return score, final_abs, rough


def _read_current_axis_value(nodes: list[Node], axis: SweepAxis) -> Any:
    hit = next((n for n in nodes if n.id == axis.node_id), None)
    if hit is None:
        return axis.values[0]
    keys = _parse_data_path(axis.path)
    cur: Any = hit.data if isinstance(hit.data, dict) else {}
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return axis.values[0]
        cur = cur[k]
    return cur


def _build_params_map(axes: list[SweepAxis], vals: list[Any]) -> dict[str, Any]:
    return {f"{axis.node_id}.{axis.path}": v for axis, v in zip(axes, vals)}


def _params_fingerprint(params: dict[str, Any]) -> str:
    return json.dumps(params, sort_keys=True, default=str)


def _trim_curve_pair(step_ticks: list[Any], loss_history: list[Any], max_points: int) -> dict[str, list[float]]:
    st_out: list[float] = []
    lh_out: list[float] = []
    for x, y in zip(step_ticks or [], loss_history or []):
        xf = _safe_float(x)
        yf = _safe_float(y)
        if xf is None or yf is None:
            continue
        st_out.append(xf)
        lh_out.append(yf)
    n = len(st_out)
    if n < 2:
        return {"step_ticks": [], "loss_history": []}
    if n <= max_points:
        return {"step_ticks": st_out, "loss_history": lh_out}
    idxs = [round(i * (n - 1) / (max_points - 1)) for i in range(max_points)]
    return {"step_ticks": [st_out[i] for i in idxs], "loss_history": [lh_out[i] for i in idxs]}


def _run_train_once(
    nodes: list[Node],
    edges: list[Edge],
    trainer_node_id: str,
    training_steps_override: int | None,
) -> tuple[dict[str, Any] | None, str | None]:
    nodes_p = _trainer_training_steps_override(nodes, trainer_node_id, training_steps_override)
    try:
        ctx = prepare_trainer_run(
            nodes_p,
            edges,
            trainer_node_id,
            resume=None,
            hessian_oversized_policy="skip",
        )
    except HTTPException as e:
        d = e.detail
        return None, d if isinstance(d, str) else str(d)
    except Exception as e:
        return None, str(e)
    for ev in iter_trainer_events_from_context(ctx):
        if ev.get("type") == "complete":
            return ev, None
        if ev.get("type") == "aborted":
            return None, "training_aborted"
        if ev.get("type") == "paused":
            return None, "training_paused_not_supported_in_tuning"
    return None, "no_complete_event"


def _is_better(a: tuple[float, float, float], b: tuple[float, float, float] | None) -> bool:
    if b is None:
        return True
    if math.isnan(a[0]):
        return False
    if math.isnan(b[0]):
        return True
    if a[0] < b[0] - 1e-12:
        return True
    if abs(a[0] - b[0]) <= 1e-12 and a[1] < b[1] - 1e-12:
        return True
    if abs(a[0] - b[0]) <= 1e-12 and abs(a[1] - b[1]) <= 1e-12 and a[2] < b[2] - 1e-12:
        return True
    return False


def validate_coordinate_descent_request(body: TrainCoordinateDescentRequest) -> None:
    nmap = {n.id: n for n in body.nodes}
    if body.trainer_node_id not in nmap:
        raise ValueError("trainer_node_id not found in nodes.")
    if nmap[body.trainer_node_id].type != "trainer":
        raise ValueError("trainer_node_id must refer to a trainer node.")
    for ax in body.axes:
        if len(ax.values) > MAX_AXIS_VALUES:
            raise ValueError(f"Axis {ax.node_id}.{ax.path} has too many values (max {MAX_AXIS_VALUES}).")
        if ax.node_id not in nmap:
            raise ValueError(f"Axis references unknown node_id: {ax.node_id}")
        _parse_data_path(ax.path)
    _prepare_curve(body.target_step_ticks, body.target_loss_history)
    try:
        prepare_trainer_run(
            body.nodes,
            body.edges,
            body.trainer_node_id,
            resume=None,
            hessian_oversized_policy="skip",
        )
    except HTTPException as e:
        d = e.detail
        raise ValueError(d if isinstance(d, str) else str(d)) from e


def iter_coordinate_descent_events(body: TrainCoordinateDescentRequest) -> Iterator[dict[str, Any]]:
    target_x, target_y = _prepare_curve(body.target_step_ticks, body.target_loss_history)
    session_id = body.session_id.strip()
    register_sweep_session(session_id)
    current_vals = [_read_current_axis_value(body.nodes, ax) for ax in body.axes]
    eval_index = 0
    best_tuple: tuple[float, float, float] | None = None
    best_params: dict[str, Any] = _build_params_map(body.axes, current_vals)
    best_curve: dict[str, Any] | None = None
    best_complete: dict[str, Any] | None = None
    best_by_fingerprint: dict[str, dict[str, Any]] = {}
    try:
        yield {
            "type": "tuning_started",
            "session_id": session_id,
            "axis_count": len(body.axes),
            "max_rounds": body.max_rounds,
            "target_points": len(target_x),
            "objective": body.score.objective,
        }
        baseline_nodes = _deep_copy_nodes(body.nodes)
        baseline_complete, baseline_err = _run_train_once(
            baseline_nodes,
            body.edges,
            body.trainer_node_id,
            body.training_steps_override,
        )
        eval_index += 1
        yield {
            "type": "baseline_evaluated",
            "session_id": session_id,
            "evaluation_index": eval_index,
            "error": baseline_err,
            "curve": _trim_curve_pair(
                (baseline_complete or {}).get("step_ticks") or [],
                (baseline_complete or {}).get("loss_history") or [],
                MAX_CURVE_POINTS_RESPONSE,
            )
            if baseline_complete
            else {"step_ticks": [], "loss_history": []},
        }
        for round_idx in range(body.max_rounds):
            if is_sweep_abort_requested(session_id):
                yield {"type": "tuning_aborted", "session_id": session_id, "evaluations": eval_index}
                return
            yield {
                "type": "round_started",
                "session_id": session_id,
                "round_index": round_idx,
                "current_params": _build_params_map(body.axes, current_vals),
            }
            improved_this_round = False
            for axis_idx, axis in enumerate(body.axes):
                axis_best_vals = list(current_vals)
                axis_best_tuple: tuple[float, float, float] | None = None
                axis_best_complete: dict[str, Any] | None = None
                axis_err_count = 0
                for candidate in axis.values:
                    if is_sweep_abort_requested(session_id):
                        yield {"type": "tuning_aborted", "session_id": session_id, "evaluations": eval_index}
                        return
                    cand_vals = list(current_vals)
                    cand_vals[axis_idx] = candidate
                    err: str | None = None
                    score = float("nan")
                    final_abs = float("nan")
                    smooth = float("nan")
                    complete: dict[str, Any] | None = None
                    eval_index += 1
                    try:
                        nodes_p = _apply_axes_to_nodes(body.nodes, body.axes, tuple(cand_vals))
                        nodes_p = _trainer_training_steps_override(
                            nodes_p, body.trainer_node_id, body.training_steps_override
                        )
                        ctx = prepare_trainer_run(
                            nodes_p,
                            body.edges,
                            body.trainer_node_id,
                            resume=None,
                            hessian_oversized_policy="skip",
                        )
                        for ev in iter_trainer_events_from_context(ctx):
                            if ev.get("type") == "complete":
                                complete = ev
                                break
                            if ev.get("type") == "aborted":
                                err = "training_aborted"
                                break
                            if ev.get("type") == "paused":
                                err = "training_paused_not_supported_in_tuning"
                                break
                        if err is None and complete is None:
                            err = "no_complete_event"
                        if err is None and complete is not None:
                            score, final_abs, smooth = _score_curve(
                                complete.get("step_ticks") or [],
                                complete.get("loss_history") or [],
                                target_x,
                                target_y,
                                body.score,
                            )
                            params_m = _build_params_map(body.axes, cand_vals)
                            fp = _params_fingerprint(params_m)
                            prev_rec = best_by_fingerprint.get(fp)
                            st_rec = (score, final_abs, smooth)
                            if prev_rec is None or _is_better(
                                st_rec,
                                (prev_rec["score"], prev_rec["final_abs"], prev_rec["smooth"]),
                            ):
                                best_by_fingerprint[fp] = {
                                    "score": score,
                                    "final_abs": final_abs,
                                    "smooth": smooth,
                                    "params": params_m,
                                    "curve": _trim_curve_pair(
                                        complete.get("step_ticks") or [],
                                        complete.get("loss_history") or [],
                                        MAX_CURVE_POINTS_RESPONSE,
                                    ),
                                }
                    except Exception as e:
                        err = str(e)
                    if err:
                        axis_err_count += 1
                    score_tuple = (score, final_abs, smooth)
                    if err is None and _is_better(score_tuple, axis_best_tuple):
                        axis_best_vals = cand_vals
                        axis_best_tuple = score_tuple
                        axis_best_complete = complete
                    if err is None and _is_better(score_tuple, best_tuple):
                        improved_this_round = best_tuple is None or (
                            score_tuple[0] + body.min_improvement < (best_tuple[0] if best_tuple else float("inf"))
                        )
                        best_tuple = score_tuple
                        best_params = _build_params_map(body.axes, cand_vals)
                        best_complete = complete
                        best_curve = {
                            "step_ticks": list((complete or {}).get("step_ticks") or []),
                            "loss_history": list((complete or {}).get("loss_history") or []),
                            "test_loss_history": list((complete or {}).get("test_loss_history") or []),
                        }
                    yield {
                        "type": "candidate_evaluated",
                        "session_id": session_id,
                        "evaluation_index": eval_index,
                        "round_index": round_idx,
                        "axis_index": axis_idx,
                        "axis_key": f"{axis.node_id}.{axis.path}",
                        "candidate_value": candidate,
                        "params": _build_params_map(body.axes, cand_vals),
                        "score": score,
                        "final_abs_error": final_abs,
                        "smoothness_penalty": smooth,
                        "error": err,
                    }
                if axis_best_tuple is not None:
                    current_vals = axis_best_vals
                yield {
                    "type": "axis_best_selected",
                    "session_id": session_id,
                    "round_index": round_idx,
                    "axis_index": axis_idx,
                    "axis_key": f"{axis.node_id}.{axis.path}",
                    "axis_error_count": axis_err_count,
                    "params": _build_params_map(body.axes, current_vals),
                    "score": axis_best_tuple[0] if axis_best_tuple else float("nan"),
                }
                if axis_best_tuple is not None and _is_better(axis_best_tuple, best_tuple):
                    best_tuple = axis_best_tuple
                    best_complete = axis_best_complete
                    best_params = _build_params_map(body.axes, current_vals)
                    best_curve = {
                        "step_ticks": list((axis_best_complete or {}).get("step_ticks") or []),
                        "loss_history": list((axis_best_complete or {}).get("loss_history") or []),
                        "test_loss_history": list((axis_best_complete or {}).get("test_loss_history") or []),
                    }
            if not improved_this_round:
                break
        ranked_rows = [
            r
            for r in best_by_fingerprint.values()
            if isinstance(r.get("score"), (int, float)) and math.isfinite(float(r["score"]))
        ]
        ranked_rows.sort(key=lambda r: (float(r["score"]), float(r["final_abs"]), float(r["smooth"])))
        ranked_curves: list[dict[str, Any]] = []
        for i, r in enumerate(ranked_rows[:4]):
            ranked_curves.append(
                {
                    "rank": i + 1,
                    "score": float(r["score"]),
                    "final_abs_error": float(r["final_abs"]),
                    "smoothness_penalty": float(r["smooth"]),
                    "params": r["params"],
                    "curve": r["curve"],
                }
            )
        baseline_trim = (
            _trim_curve_pair(
                (baseline_complete or {}).get("step_ticks") or [],
                (baseline_complete or {}).get("loss_history") or [],
                MAX_CURVE_POINTS_RESPONSE,
            )
            if baseline_complete
            else {"step_ticks": [], "loss_history": []}
        )
        target_trim = _trim_curve_pair(
            list(body.target_step_ticks),
            list(body.target_loss_history),
            MAX_CURVE_POINTS_RESPONSE,
        )
        best_src_ticks = (best_complete or {}).get("step_ticks") or (best_curve or {}).get("step_ticks") or []
        best_src_loss = (best_complete or {}).get("loss_history") or (best_curve or {}).get("loss_history") or []
        best_trim = _trim_curve_pair(best_src_ticks, best_src_loss, MAX_CURVE_POINTS_RESPONSE)
        best_test = (best_complete or {}).get("test_loss_history") or (best_curve or {}).get("test_loss_history") or []
        best_test_trim: list[float] = []
        if isinstance(best_test, list) and len(best_src_ticks) == len(best_test):
            paired = _trim_curve_pair(best_src_ticks, best_test, MAX_CURVE_POINTS_RESPONSE)
            best_test_trim = [float(x) for x in paired["loss_history"]]
        out: dict[str, Any] = {
            "type": "tuning_complete",
            "session_id": session_id,
            "evaluations": eval_index,
            "best_params": best_params,
            "best_score": (best_tuple[0] if best_tuple else float("nan")),
            "best_final_abs_error": (best_tuple[1] if best_tuple else float("nan")),
            "best_smoothness_penalty": (best_tuple[2] if best_tuple else float("nan")),
            "best_curve": {
                **best_trim,
                "test_loss_history": best_test_trim,
            },
            "baseline_curve": baseline_trim,
            "target_curve": target_trim,
            "ranked_curves": ranked_curves,
        }
        if isinstance(best_complete, dict):
            out["best_complete"] = {
                "loss_history": list(best_complete.get("loss_history") or []),
                "test_loss_history": list(best_complete.get("test_loss_history") or []),
                "step_ticks": list(best_complete.get("step_ticks") or []),
                "checkpoint_b64": best_complete.get("checkpoint_b64"),
            }
        yield out
    finally:
        unregister_sweep_session(session_id)
