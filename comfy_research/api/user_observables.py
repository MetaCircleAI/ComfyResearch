"""Persisted user-defined observables (from 0D tensor viz)."""

from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, TypeVar

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from comfy_research.schemas.graph import Edge, Node

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DATA_PATH = _REPO_ROOT / "data" / "user_observables.json"
_GROKING_BUNDLE_PATH = _REPO_ROOT / "data" / "bundled" / "grokking_demo_user_observables.json"
_GROKING_OBS_PREFIX = "grokking-demo-obs-"
_LOCK = threading.Lock()
_T = TypeVar("_T")


class UserObservableRecord(BaseModel):
    id: str
    label: str
    tensor_viz_node_id: str = ""
    tensor_selector_node_id: str = ""
    definition_code: str = ""
    human_chain: str = ""
    created_at: str = ""
    definition_kind: str = "graph_path"
    source_model_node_id: str = ""
    tensor_name: str = ""
    tensor_shape: list[int] = Field(default_factory=list)
    reductions: list[dict[str, Any]] = Field(default_factory=list)
    tensor_scope: str = "single"
    tensor_pattern: str = ""
    flatten_mode: str = "none"
    observable_source: str = "weight"
    representation_id: str = ""
    layer_index: int = 0
    layer_io: str = ""


class UserObservablesFile(BaseModel):
    version: int = 1
    items: list[UserObservableRecord] = Field(default_factory=list)


class CreateUserObservableBody(BaseModel):
    tensor_viz_node_id: str = Field(min_length=1)
    label: str | None = None
    nodes: list[Node] = Field(default_factory=list)
    edges: list[Edge] = Field(default_factory=list)


class PatchUserObservableBody(BaseModel):
    label: str = Field(min_length=1)


class AxisReductionBody(BaseModel):
    axis_index: int
    axis_label: str = ""
    op: str = Field(min_length=1)


class CreateAlgebraObservableBody(BaseModel):
    label: str | None = None
    source_model_node_id: str = Field(min_length=1)
    tensor_name: str = Field(min_length=1)
    tensor_shape: list[int] = Field(default_factory=list)
    reductions: list[AxisReductionBody] = Field(default_factory=list)
    tensor_scope: str = "single"
    flatten_mode: str = "none"
    observable_source: str = "weight"
    representation_id: str = ""
    layer_index: int = 0
    layer_io: str = ""


class DescribeObservablePathBody(BaseModel):
    nodes: list[Node] = Field(default_factory=list)
    edges: list[Edge] = Field(default_factory=list)
    tensor_viz_node_id: str = Field(min_length=1)


def _ensure_parent() -> None:
    _DATA_PATH.parent.mkdir(parents=True, exist_ok=True)


def _parse_file_text(text: str) -> dict[str, Any]:
    stripped = text.lstrip()
    try:
        raw = json.loads(stripped)
    except json.JSONDecodeError as e:
        try:
            raw, end = json.JSONDecoder().raw_decode(stripped)
        except json.JSONDecodeError as e2:
            raise e2 from e
        if end < len(stripped):
            salvaged = UserObservablesFile.model_validate(raw)
            _write_file_unlocked(salvaged)
    if not isinstance(raw, dict):
        raise ValueError("User observables file must be a JSON object.")
    return raw


def _read_file_unlocked() -> UserObservablesFile:
    if not _DATA_PATH.is_file():
        return UserObservablesFile()
    try:
        text = _DATA_PATH.read_text(encoding="utf-8")
        raw = _parse_file_text(text)
        return UserObservablesFile.model_validate(raw)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Invalid user observables file: {e}") from e


def _write_file_unlocked(data: UserObservablesFile) -> None:
    _ensure_parent()
    text = json.dumps(data.model_dump(), indent=2, ensure_ascii=False)
    tmp = _DATA_PATH.with_suffix(_DATA_PATH.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(_DATA_PATH)


def _load() -> UserObservablesFile:
    with _LOCK:
        return _read_file_unlocked()


def _save(data: UserObservablesFile) -> None:
    with _LOCK:
        _write_file_unlocked(data)


def _update(mutator: Callable[[UserObservablesFile], _T]) -> _T:
    with _LOCK:
        data = _read_file_unlocked()
        result = mutator(data)
        _write_file_unlocked(data)
        return result


def get_user_observable_record(item_id: str) -> UserObservableRecord | None:
    """Used by the trainer to resolve ``tensor_selector_node_id`` without an HTTP round trip."""
    data = _load()
    for it in data.items:
        if it.id == item_id:
            return it
    return None


def ensure_grokking_demo_user_observables() -> int:
    """Merge bundled grokking demo observables when the template ids are missing locally."""
    if not _GROKING_BUNDLE_PATH.is_file():
        return 0
    try:
        raw = json.loads(_GROKING_BUNDLE_PATH.read_text(encoding="utf-8"))
        bundled = UserObservablesFile.model_validate(raw)
    except Exception:
        return 0

    def _merge(data: UserObservablesFile) -> int:
        existing = {it.id for it in data.items}
        added = 0
        for rec in bundled.items:
            if not str(rec.id).startswith(_GROKING_OBS_PREFIX):
                continue
            if rec.id in existing:
                continue
            data.items.append(rec)
            existing.add(rec.id)
            added += 1
        return added

    return _update(_merge)


router = APIRouter(prefix="/api", tags=["user-observables"])


@router.post("/user-observables/describe-path")
def post_describe_observable_path(body: DescribeObservablePathBody) -> dict[str, Any]:
    """Return pseudo-Python and a short human-readable pipeline line.

    ``tensor_viz_node_id`` may be a tensor viz id or, equivalently, a tensor selector id (viz not required).
    """
    from comfy_research.engine.analysis.observable_user_eval import describe_observable_training_path

    tid = body.tensor_viz_node_id.strip()
    return describe_observable_training_path(body.nodes, body.edges, tid)


@router.get("/user-observables")
def list_user_observables() -> dict[str, Any]:
    data = _load()
    return {"version": 1, "items": [i.model_dump() for i in data.items]}


@router.get("/user-observables/{item_id}")
def get_user_observable(item_id: str) -> dict[str, Any]:
    rec = get_user_observable_record(item_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="User observable not found.")
    return {"item": rec.model_dump()}


@router.post("/user-observables")
def create_user_observable(body: CreateUserObservableBody) -> dict[str, Any]:
    tv = body.tensor_viz_node_id.strip()
    if not tv:
        raise HTTPException(status_code=400, detail="tensor_viz_node_id is required.")
    label = (body.label or "").strip() or f"0D viz · {tv[:10]}…"
    uid = str(uuid.uuid4())

    tensor_selector_node_id = ""
    definition_code = ""
    human_chain = ""
    if body.nodes and body.edges:
        from comfy_research.engine.analysis.observable_user_eval import (
            describe_observable_training_path,
            parse_observable_user_path,
        )

        try:
            path = parse_observable_user_path(body.nodes, body.edges, tv)
            ts_id = path.chain_rev[-1].id if path.chain_rev else ""
            tensor_selector_node_id = ts_id
            desc = describe_observable_training_path(body.nodes, body.edges, ts_id)
            definition_code = desc["definition"]
            human_chain = desc["human_chain"]
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Could not snapshot observable definition from the graph: {e}",
            ) from e

    rec = UserObservableRecord(
        id=uid,
        label=label,
        tensor_viz_node_id=tv,
        tensor_selector_node_id=tensor_selector_node_id,
        definition_code=definition_code,
        human_chain=human_chain,
        created_at=datetime.now(timezone.utc).isoformat(),
        definition_kind="graph_path",
    )

    def _append(data: UserObservablesFile) -> UserObservableRecord:
        data.items.append(rec)
        return rec

    saved = _update(_append)
    return {"item": saved.model_dump()}


@router.post("/user-observables/algebra")
def create_algebra_observable(body: CreateAlgebraObservableBody) -> dict[str, Any]:
    from comfy_research.engine.analysis.observable_algebra import (
        auto_algebra_label,
        family_pattern_from_representation_id,
        family_pattern_from_tensor_name,
        format_algebra_definition_code,
        format_algebra_human_chain,
        global_flatten_label_base,
        global_flatten_representation_kind,
        normalize_flatten_mode,
        parse_axis_reductions,
    )

    model_id = body.source_model_node_id.strip()
    tensor_name = body.tensor_name.strip()
    if not model_id or not tensor_name:
        raise HTTPException(status_code=400, detail="source_model_node_id and tensor_name are required.")
    observable_source = (body.observable_source or "weight").strip().lower()
    if observable_source not in ("weight", "representation"):
        raise HTTPException(status_code=400, detail="observable_source must be 'weight' or 'representation'.")
    representation_id = (body.representation_id or tensor_name).strip() if observable_source == "representation" else ""
    try:
        flatten_mode = normalize_flatten_mode(body.flatten_mode)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    try:
        specs = parse_axis_reductions(
            [r.model_dump() for r in body.reductions],
            flatten_mode=flatten_mode,
            tensor_shape=[int(x) for x in body.tensor_shape],
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    scope = (body.tensor_scope or "single").strip()
    if scope not in ("single", "all_matching"):
        raise HTTPException(status_code=400, detail="tensor_scope must be 'single' or 'all_matching'.")
    subject_id = representation_id if observable_source == "representation" else tensor_name
    if flatten_mode == "global":
        if observable_source == "representation":
            pattern = global_flatten_representation_kind(representation_id)
        else:
            pattern = global_flatten_label_base(tensor_name)
    elif scope == "all_matching":
        if observable_source == "representation":
            pattern = family_pattern_from_representation_id(representation_id)
        else:
            pattern = family_pattern_from_tensor_name(tensor_name)
    else:
        pattern = subject_id

    label = (body.label or "").strip()
    if not label:
        label = auto_algebra_label(
            tensor_name=subject_id,
            reductions=specs,
            tensor_scope=scope,
            flatten_mode=flatten_mode,
            observable_source=observable_source,
            representation_id=representation_id,
            layer_index=int(body.layer_index),
            layer_io=str(body.layer_io or ""),
        )

    uid = str(uuid.uuid4())
    human_chain = format_algebra_human_chain(
        tensor_name=subject_id,
        tensor_shape=[int(x) for x in body.tensor_shape],
        reductions=specs,
        source_model_node_id=model_id,
        tensor_scope=scope,
        flatten_mode=flatten_mode,
        observable_source=observable_source,
        representation_id=representation_id,
        layer_index=int(body.layer_index),
        layer_io=str(body.layer_io or ""),
    )
    definition_code = format_algebra_definition_code(
        tensor_name=subject_id,
        reductions=specs,
        flatten_mode=flatten_mode,
        observable_source=observable_source,
    )
    rec = UserObservableRecord(
        id=uid,
        label=label,
        definition_code=definition_code,
        human_chain=human_chain,
        created_at=datetime.now(timezone.utc).isoformat(),
        definition_kind="algebra",
        source_model_node_id=model_id,
        tensor_name=subject_id,
        tensor_shape=[int(x) for x in body.tensor_shape],
        reductions=[r.model_dump() for r in body.reductions],
        tensor_scope=scope,
        tensor_pattern=pattern,
        flatten_mode=flatten_mode,
        observable_source=observable_source,
        representation_id=representation_id,
        layer_index=int(body.layer_index),
        layer_io=str(body.layer_io or ""),
    )

    def _append(data: UserObservablesFile) -> UserObservableRecord:
        data.items.append(rec)
        return rec

    saved = _update(_append)
    return {"item": saved.model_dump()}


@router.patch("/user-observables/{item_id}")
def patch_user_observable(item_id: str, body: PatchUserObservableBody) -> dict[str, Any]:
    label = body.label.strip()
    if not label:
        raise HTTPException(status_code=400, detail="label is required.")
    def _patch(data: UserObservablesFile) -> UserObservableRecord:
        for i, it in enumerate(data.items):
            if it.id == item_id:
                data.items[i] = UserObservableRecord(
                    id=it.id,
                    label=label,
                    tensor_viz_node_id=it.tensor_viz_node_id,
                    tensor_selector_node_id=it.tensor_selector_node_id,
                    definition_code=it.definition_code,
                    human_chain=it.human_chain,
                    created_at=it.created_at,
                    definition_kind=it.definition_kind,
                    source_model_node_id=it.source_model_node_id,
                    tensor_name=it.tensor_name,
                    tensor_shape=it.tensor_shape,
                    reductions=it.reductions,
                    tensor_scope=it.tensor_scope,
                    tensor_pattern=it.tensor_pattern,
                    flatten_mode=it.flatten_mode,
                    observable_source=it.observable_source,
                    representation_id=it.representation_id,
                    layer_index=it.layer_index,
                    layer_io=it.layer_io,
                )
                return data.items[i]
        raise HTTPException(status_code=404, detail="User observable not found.")

    saved = _update(_patch)
    return {"item": saved.model_dump()}


@router.delete("/user-observables/{item_id}")
def delete_user_observable(item_id: str) -> dict[str, str]:
    def _delete(data: UserObservablesFile) -> None:
        before = len(data.items)
        data.items = [x for x in data.items if x.id != item_id]
        if len(data.items) == before:
            raise HTTPException(status_code=404, detail="User observable not found.")

    _update(_delete)
    return {"status": "ok"}


def catalog_children_for_observables() -> list[dict[str, Any]]:
    """Entries merged into the observables category in node-categories."""
    data = _load()
    out: list[dict[str, Any]] = []
    for it in data.items:
        out.append(
            {
                "id": "observable_user",
                "label": it.label,
                "user_observable_id": it.id,
                "tensor_viz_node_id": it.tensor_viz_node_id,
                "tensor_selector_node_id": it.tensor_selector_node_id,
                "definition_kind": it.definition_kind,
                "source_model_node_id": it.source_model_node_id,
                "deletable": True,
            }
        )
    return out
