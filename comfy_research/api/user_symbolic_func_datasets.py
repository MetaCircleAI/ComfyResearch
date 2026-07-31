"""Persisted user-defined symbolic func dataset blueprints (Save new from spec editor)."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DATA_PATH = _REPO_ROOT / "data" / "user_symbolic_func_datasets.json"


class UserSymbolicFuncDatasetRecord(BaseModel):
    id: str
    label: str
    node_data: dict[str, Any] = Field(default_factory=dict)
    created_at: str = ""


class UserSymbolicFuncDatasetsFile(BaseModel):
    version: int = 1
    items: list[UserSymbolicFuncDatasetRecord] = Field(default_factory=list)


class CreateUserSymbolicFuncDatasetBody(BaseModel):
    label: str = Field(min_length=1)
    node_data: dict[str, Any] = Field(default_factory=dict)


class PatchUserSymbolicFuncDatasetBody(BaseModel):
    label: str = Field(min_length=1)


def _ensure_parent() -> None:
    _DATA_PATH.parent.mkdir(parents=True, exist_ok=True)


def _load() -> UserSymbolicFuncDatasetsFile:
    if not _DATA_PATH.is_file():
        return UserSymbolicFuncDatasetsFile()
    try:
        raw = json.loads(_DATA_PATH.read_text(encoding="utf-8"))
        return UserSymbolicFuncDatasetsFile.model_validate(raw)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Invalid user symbolic func datasets file: {e}",
        ) from e


def _save(data: UserSymbolicFuncDatasetsFile) -> None:
    _ensure_parent()
    _DATA_PATH.write_text(
        json.dumps(data.model_dump(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


router = APIRouter(prefix="/api", tags=["user-symbolic-func-datasets"])


@router.get("/user-symbolic-func-datasets")
def list_user_symbolic_func_datasets() -> dict[str, Any]:
    data = _load()
    return {"version": 1, "items": [i.model_dump() for i in data.items]}


@router.get("/user-symbolic-func-datasets/{item_id}")
def get_user_symbolic_func_dataset(item_id: str) -> dict[str, Any]:
    data = _load()
    for it in data.items:
        if it.id == item_id:
            return {"item": it.model_dump()}
    raise HTTPException(status_code=404, detail="User symbolic func dataset not found.")


@router.post("/user-symbolic-func-datasets")
def create_user_symbolic_func_dataset(body: CreateUserSymbolicFuncDatasetBody) -> dict[str, Any]:
    label = body.label.strip()
    if not label:
        raise HTTPException(status_code=400, detail="label is required.")
    data = _load()
    uid = str(uuid.uuid4())
    rec = UserSymbolicFuncDatasetRecord(
        id=uid,
        label=label,
        node_data=dict(body.node_data),
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    data.items.append(rec)
    _save(data)
    return {"item": rec.model_dump()}


@router.patch("/user-symbolic-func-datasets/{item_id}")
def patch_user_symbolic_func_dataset(item_id: str, body: PatchUserSymbolicFuncDatasetBody) -> dict[str, Any]:
    label = body.label.strip()
    if not label:
        raise HTTPException(status_code=400, detail="label is required.")
    data = _load()
    for i, it in enumerate(data.items):
        if it.id == item_id:
            data.items[i] = UserSymbolicFuncDatasetRecord(
                id=it.id,
                label=label,
                node_data=it.node_data,
                created_at=it.created_at,
            )
            _save(data)
            return {"item": data.items[i].model_dump()}
    raise HTTPException(status_code=404, detail="User symbolic func dataset not found.")


@router.delete("/user-symbolic-func-datasets/{item_id}")
def delete_user_symbolic_func_dataset(item_id: str) -> dict[str, str]:
    data = _load()
    before = len(data.items)
    data.items = [x for x in data.items if x.id != item_id]
    if len(data.items) == before:
        raise HTTPException(status_code=404, detail="User symbolic func dataset not found.")
    _save(data)
    return {"status": "ok"}


def load_user_symbolic_node_data(item_id: str) -> dict[str, Any] | None:
    """Return persisted ``node_data`` for a user symbolic dataset blueprint, or ``None`` if missing."""
    uid = str(item_id or "").strip()
    if not uid:
        return None
    data = _load()
    for it in data.items:
        if it.id == uid:
            return dict(it.node_data)
    return None


def catalog_children_for_symbolic_func_datasets() -> list[dict[str, Any]]:
    """Entries merged into the dataset category in node-categories."""
    data = _load()
    out: list[dict[str, Any]] = []
    for it in data.items:
        out.append(
            {
                "id": "symbolic_func_dataset",
                "label": it.label,
                "user_symbolic_func_dataset_id": it.id,
                "deletable": True,
            }
        )
    return out
