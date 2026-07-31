"""Parameter sweep: Cartesian grid over node ``data`` fields, one full train per point."""

from __future__ import annotations

import copy
import itertools
import math
from typing import Any, Iterator, Literal

from fastapi import HTTPException
from pydantic import BaseModel, Field, model_validator

from comfy_research.generated.node_capabilities import has_capability
from comfy_research.engine.runs.sweep_control import (
    is_sweep_abort_requested,
    register_sweep_session,
    unregister_sweep_session,
)
from comfy_research.engine.crl.crl_run import iter_crl_events_from_context, prepare_crl_run
from comfy_research.engine.runs.trainer_run import iter_trainer_events_from_context, prepare_trainer_run
from comfy_research.schemas.graph import Edge, Node, NodeKind

MAX_SWEEP_POINTS = 256


class SweepAxis(BaseModel):
    """One tunable: patch ``node.data`` at a dot-separated path (e.g. ``learningRate``, ``extras.myParam``)."""

    node_id: str = Field(min_length=1)
    path: str = Field(
        min_length=1,
        description="Keys under node.data, dot-separated (e.g. learningRate, extras.myParam).",
    )
    values: list[Any] = Field(min_length=1)

    @model_validator(mode="after")
    def _non_empty_values(self) -> SweepAxis:
        if len(self.values) > 512:
            raise ValueError("Each axis may have at most 512 values.")
        return self


MetricReduce = Literal["last", "min", "max", "mean"]
MetricKind = Literal["observable_user", "final_train_loss", "final_test_loss"]


class SweepMetric(BaseModel):
    kind: MetricKind
    observable_node_id: str = ""
    reduce: MetricReduce = "last"

    @model_validator(mode="after")
    def _obs_required(self) -> SweepMetric:
        if self.kind == "observable_user" and not str(self.observable_node_id or "").strip():
            raise ValueError("observable_node_id is required when kind is observable_user.")
        return self


class TrainSweepRequest(BaseModel):
    trainer_node_id: str = Field(min_length=1)
    nodes: list[Node] = Field(default_factory=list)
    edges: list[Edge] = Field(default_factory=list)
    axes: list[SweepAxis] = Field(min_length=1)
    metric: SweepMetric
    sweep_session_id: str = Field(min_length=1, description="Client-generated id for abort between runs.")
    training_steps_override: int | None = Field(
        default=None,
        ge=1,
        description="If set, overrides trainer node trainingSteps for every sweep run.",
    )


def _parse_data_path(path: str) -> list[str]:
    parts = [p for p in path.strip().split(".") if p]
    if not parts:
        raise ValueError(f"Invalid data path: {path!r}")
    return parts


def _set_in_dict(root: dict[str, Any], keys: list[str], value: Any) -> None:
    cur: dict[str, Any] = root
    for k in keys[:-1]:
        nxt = cur.get(k)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[k] = nxt
        cur = nxt
    cur[keys[-1]] = value


def _deep_copy_nodes(nodes: list[Node]) -> list[Node]:
    raw = [n.model_dump(mode="json") for n in nodes]
    return [Node.model_validate(x) for x in raw]


def _apply_axes_to_nodes(nodes: list[Node], axes: list[SweepAxis], value_tuple: tuple[Any, ...]) -> list[Node]:
    out = _deep_copy_nodes(nodes)
    by_id = {n.id: i for i, n in enumerate(out)}
    for axis, val in zip(axes, value_tuple):
        i = by_id.get(axis.node_id)
        if i is None:
            raise ValueError(f"Sweep axis references unknown node_id: {axis.node_id}")
        n = out[i]
        keys = _parse_data_path(axis.path)
        data = dict(copy.deepcopy(n.data) if n.data else {})
        _set_in_dict(data, keys, val)
        out[i] = n.model_copy(update={"data": data})
    return out


def _trainer_training_steps_override(nodes: list[Node], trainer_id: str, steps: int | None) -> list[Node]:
    if steps is None:
        return nodes
    out = _deep_copy_nodes(nodes)
    for i, n in enumerate(out):
        if n.id != trainer_id:
            continue
        if not has_capability(n.type, "trainer_runner"):
            continue
        data = dict(copy.deepcopy(n.data) if n.data else {})
        data["trainingSteps"] = int(steps)
        out[i] = n.model_copy(update={"data": data})
        break
    return out


def _expand_cartesian(axes: list[SweepAxis]) -> list[tuple[Any, ...]]:
    lens = math.prod(len(a.values) for a in axes)
    if lens > MAX_SWEEP_POINTS:
        raise ValueError(
            f"Sweep grid has {lens} points; maximum is {MAX_SWEEP_POINTS}. "
            "Reduce the number of values per axis.",
        )
    return list(itertools.product(*(tuple(a.values) for a in axes)))


def _column_key(axis: SweepAxis) -> str:
    return f"{axis.node_id}.{axis.path}"


def _reduce_series(vals: list[float] | None, how: MetricReduce) -> float:
    if not vals:
        return float("nan")
    xs = [float(x) for x in vals if isinstance(x, (int, float)) and x == x]
    if not xs:
        return float("nan")
    if how == "last":
        return xs[-1]
    if how == "min":
        return min(xs)
    if how == "max":
        return max(xs)
    if how == "mean":
        return sum(xs) / len(xs)
    return float("nan")


def _metric_from_complete(ev: dict[str, Any], metric: SweepMetric) -> float:
    if metric.kind == "final_train_loss":
        lh = ev.get("loss_history") or []
        if not lh:
            return float("nan")
        return float(lh[-1])
    if metric.kind == "final_test_loss":
        th = ev.get("test_loss_history") or []
        if not th:
            return float("nan")
        return float(th[-1])
    oid = str(metric.observable_node_id).strip()
    omh = ev.get("observable_metric_histories") or {}
    hist = omh.get(oid) if isinstance(omh, dict) else None
    if not isinstance(hist, list):
        return float("nan")
    nums: list[float] = []
    for x in hist:
        try:
            nums.append(float(x))
        except (TypeError, ValueError):
            continue
    return _reduce_series(nums, metric.reduce)


def _observable_connected_to_trainer(edges: list[Edge], trainer_id: str, observable_id: str) -> bool:
    for e in edges:
        if e.target == trainer_id and e.targetHandle == "observables" and e.source == observable_id:
            return True
    return False


def validate_sweep_request(body: TrainSweepRequest) -> None:
    nmap = {n.id: n for n in body.nodes}
    if body.trainer_node_id not in nmap:
        raise ValueError("trainer_node_id not found in nodes.")
    tk = str(nmap[body.trainer_node_id].type)
    if not has_capability(tk, "trainer_runner"):
        raise ValueError("trainer_node_id must refer to a trainer or crl_trainer node.")

    for ax in body.axes:
        if ax.node_id not in nmap:
            raise ValueError(f"Sweep axis unknown node_id: {ax.node_id}")
        try:
            _parse_data_path(ax.path)
        except ValueError as e:
            raise ValueError(str(e)) from e

    if body.metric.kind == "observable_user":
        oid = body.metric.observable_node_id.strip()
        if oid not in nmap:
            raise ValueError("observable_node_id not found in nodes.")
        if nmap[oid].type != "observable_user":
            raise ValueError("metric observable_node_id must be an observable_user node.")
        if not _observable_connected_to_trainer(body.edges, body.trainer_node_id, oid):
            raise ValueError("observable_user must be connected to this trainer's observables input.")

    _expand_cartesian(body.axes)

    # Dry-run prepare on base graph (catches wiring errors before streaming).
    try:
        if str(nmap[body.trainer_node_id].type) == str(NodeKind.crl_trainer):
            prepare_crl_run(body.nodes, body.edges, body.trainer_node_id, resume=None)
        else:
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


def iter_sweep_events(body: TrainSweepRequest) -> Iterator[dict[str, Any]]:
    """Yield NDJSON-friendly dicts: sweep_started, sweep_progress, sweep_row, sweep_complete | sweep_aborted."""
    combos = _expand_cartesian(body.axes)
    metric_col = (
        f"obs:{body.metric.observable_node_id}:{body.metric.reduce}"
        if body.metric.kind == "observable_user"
        else ("final_train_loss" if body.metric.kind == "final_train_loss" else "final_test_loss")
    )
    columns = [*[_column_key(ax) for ax in body.axes], metric_col, "error"]

    session_id = body.sweep_session_id.strip()
    register_sweep_session(session_id)
    rows_out: list[dict[str, Any]] = []
    nmap = {n.id: n for n in body.nodes}
    try:
        yield {
            "type": "sweep_started",
            "session_id": session_id,
            "total": len(combos),
            "columns": columns,
        }
        for i, tup in enumerate(combos):
            if is_sweep_abort_requested(session_id):
                yield {"type": "sweep_aborted", "session_id": session_id, "completed_rows": len(rows_out)}
                return
            yield {"type": "sweep_progress", "index": i, "total": len(combos)}
            err: str | None = None
            metric_val = float("nan")
            try:
                nodes_p = _apply_axes_to_nodes(body.nodes, body.axes, tup)
                nodes_p = _trainer_training_steps_override(
                    nodes_p, body.trainer_node_id, body.training_steps_override
                )
                is_crl = str(nmap[body.trainer_node_id].type) == str(NodeKind.crl_trainer)
                if is_crl:
                    ctx_crl = prepare_crl_run(nodes_p, body.edges, body.trainer_node_id, resume=None)
                    event_iter = iter_crl_events_from_context(ctx_crl)
                else:
                    ctx_sup = prepare_trainer_run(
                        nodes_p,
                        body.edges,
                        body.trainer_node_id,
                        resume=None,
                        hessian_oversized_policy="skip",
                    )
                    event_iter = iter_trainer_events_from_context(ctx_sup)
                complete: dict[str, Any] | None = None
                for ev in event_iter:
                    if ev.get("type") == "complete":
                        complete = ev
                        break
                    if ev.get("type") == "aborted":
                        err = "training_aborted"
                        break
                    if ev.get("type") == "paused":
                        err = "training_paused_not_supported_in_sweep"
                        break
                if err:
                    pass
                elif complete is None:
                    err = "no_complete_event"
                else:
                    metric_val = _metric_from_complete(complete, body.metric)
            except Exception as e:
                err = str(e)

            row: dict[str, Any] = {}
            for ax, v in zip(body.axes, tup):
                row[_column_key(ax)] = v
            row[metric_col] = metric_val
            row["error"] = err
            rows_out.append(row)
            yield {
                "type": "sweep_row",
                "index": i,
                "total": len(combos),
                "params": {_column_key(ax): v for ax, v in zip(body.axes, tup)},
                "metric_value": metric_val,
                "error": err,
            }

        yield {
            "type": "sweep_complete",
            "session_id": session_id,
            "columns": columns,
            "rows": rows_out,
        }
    finally:
        unregister_sweep_session(session_id)
