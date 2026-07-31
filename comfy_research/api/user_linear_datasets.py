"""Persisted user-defined linear dataset blueprints (Save new from spec editor)."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DATA_PATH = _REPO_ROOT / "data" / "user_linear_datasets.json"


class UserLinearDatasetRecord(BaseModel):
    id: str
    label: str
    node_data: dict[str, Any] = Field(default_factory=dict)
    created_at: str = ""


class UserLinearDatasetsFile(BaseModel):
    version: int = 1
    items: list[UserLinearDatasetRecord] = Field(default_factory=list)


class CreateUserLinearDatasetBody(BaseModel):
    label: str = Field(min_length=1)
    node_data: dict[str, Any] = Field(default_factory=dict)


class PatchUserLinearDatasetBody(BaseModel):
    label: str = Field(min_length=1)


def _ensure_parent() -> None:
    _DATA_PATH.parent.mkdir(parents=True, exist_ok=True)


def _load() -> UserLinearDatasetsFile:
    if not _DATA_PATH.is_file():
        return UserLinearDatasetsFile()
    try:
        raw = json.loads(_DATA_PATH.read_text(encoding="utf-8"))
        return UserLinearDatasetsFile.model_validate(raw)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Invalid user linear datasets file: {e}") from e


def _save(data: UserLinearDatasetsFile) -> None:
    _ensure_parent()
    _DATA_PATH.write_text(
        json.dumps(data.model_dump(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


router = APIRouter(prefix="/api", tags=["user-linear-datasets"])


@router.get("/user-linear-datasets")
def list_user_linear_datasets() -> dict[str, Any]:
    data = _load()
    return {"version": 1, "items": [i.model_dump() for i in data.items]}


@router.get("/user-linear-datasets/{item_id}")
def get_user_linear_dataset(item_id: str) -> dict[str, Any]:
    data = _load()
    for it in data.items:
        if it.id == item_id:
            return {"item": it.model_dump()}
    raise HTTPException(status_code=404, detail="User linear dataset not found.")


@router.post("/user-linear-datasets")
def create_user_linear_dataset(body: CreateUserLinearDatasetBody) -> dict[str, Any]:
    label = body.label.strip()
    if not label:
        raise HTTPException(status_code=400, detail="label is required.")
    data = _load()
    uid = str(uuid.uuid4())
    rec = UserLinearDatasetRecord(
        id=uid,
        label=label,
        node_data=dict(body.node_data),
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    data.items.append(rec)
    _save(data)
    return {"item": rec.model_dump()}


@router.patch("/user-linear-datasets/{item_id}")
def patch_user_linear_dataset(item_id: str, body: PatchUserLinearDatasetBody) -> dict[str, Any]:
    label = body.label.strip()
    if not label:
        raise HTTPException(status_code=400, detail="label is required.")
    data = _load()
    for i, it in enumerate(data.items):
        if it.id == item_id:
            data.items[i] = UserLinearDatasetRecord(
                id=it.id,
                label=label,
                node_data=it.node_data,
                created_at=it.created_at,
            )
            _save(data)
            return {"item": data.items[i].model_dump()}
    raise HTTPException(status_code=404, detail="User linear dataset not found.")


@router.delete("/user-linear-datasets/{item_id}")
def delete_user_linear_dataset(item_id: str) -> dict[str, str]:
    data = _load()
    before = len(data.items)
    data.items = [x for x in data.items if x.id != item_id]
    if len(data.items) == before:
        raise HTTPException(status_code=404, detail="User linear dataset not found.")
    _save(data)
    return {"status": "ok"}


def catalog_children_for_linear_datasets() -> list[dict[str, Any]]:
    """Entries merged into the dataset category in node-categories."""
    data = _load()
    out: list[dict[str, Any]] = []
    for it in data.items:
        out.append(
            {
                "id": "linear_dataset",
                "label": it.label,
                "user_linear_dataset_id": it.id,
                "deletable": True,
            }
        )
    return out
