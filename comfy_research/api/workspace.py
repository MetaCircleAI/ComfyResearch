"""Persisted workspace: multiple projects, each with one canvas."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, ValidationError, model_validator

from comfy_research.schemas.workspace import (
    WorkspaceCanvas,
    WorkspaceProject,
    WorkspaceSnapshot,
    empty_workspace,
)

router = APIRouter(prefix="/api", tags=["workspace"])

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WORKSPACE_PATH = _REPO_ROOT / "data" / "workspace.json"


class _WorkspaceV2Project(BaseModel):
    id: str
    title: str = "Project"
    active_canvas_id: str
    canvases: list[WorkspaceCanvas] = Field(default_factory=list)

    @model_validator(mode="after")
    def _active_canvas_exists(self) -> _WorkspaceV2Project:
        if self.active_canvas_id not in {canvas.id for canvas in self.canvases}:
            raise ValueError("active_canvas_id must match a canvas id")
        return self


class _WorkspaceV2Snapshot(BaseModel):
    version: Literal[2]
    active_project_id: str
    projects: list[_WorkspaceV2Project] = Field(default_factory=list)

    @model_validator(mode="after")
    def _active_project_exists(self) -> _WorkspaceV2Snapshot:
        if self.active_project_id not in {project.id for project in self.projects}:
            raise ValueError("active_project_id must match a project id")
        return self


def _ensure_data_dir() -> None:
    _WORKSPACE_PATH.parent.mkdir(parents=True, exist_ok=True)


def _migrate_v2_workspace(raw: object) -> WorkspaceSnapshot:
    if isinstance(raw, dict):
        projects = raw.get("projects")
        if isinstance(projects, list) and all(
            isinstance(project, dict) and "canvas" in project for project in projects
        ):
            return WorkspaceSnapshot.model_validate({**raw, "version": 3})

    legacy = _WorkspaceV2Snapshot.model_validate(raw)
    projects: list[WorkspaceProject] = []

    for legacy_project in legacy.projects:
        active_canvas = next(
            canvas
            for canvas in legacy_project.canvases
            if canvas.id == legacy_project.active_canvas_id
        )
        projects.append(
            WorkspaceProject(
                id=legacy_project.id,
                title=legacy_project.title,
                canvas=active_canvas,
            )
        )
        projects.extend(
            WorkspaceProject(
                id=str(uuid.uuid4()),
                title=f"{legacy_project.title} / {canvas.title}",
                canvas=canvas,
            )
            for canvas in legacy_project.canvases
            if canvas.id != legacy_project.active_canvas_id
        )

    return WorkspaceSnapshot(
        active_project_id=legacy.active_project_id,
        projects=projects,
    )


def _load_or_create_workspace() -> WorkspaceSnapshot:
    """Load workspace.json or create a fresh single-canvas workspace."""
    if _WORKSPACE_PATH.is_file():
        try:
            workspace_text = _WORKSPACE_PATH.read_text(encoding="utf-8")
            raw = json.loads(workspace_text)
            if isinstance(raw, dict) and raw.get("version") == 2:
                snap = _migrate_v2_workspace(raw)
                _WORKSPACE_PATH.with_suffix(".v2.json").write_text(workspace_text, encoding="utf-8")
                _WORKSPACE_PATH.write_text(snap.model_dump_json(indent=2), encoding="utf-8")
                return snap
            return WorkspaceSnapshot.model_validate(raw)
        except (json.JSONDecodeError, ValidationError) as e:
            raise HTTPException(status_code=500, detail=f"Invalid workspace.json: {e}") from e
    snap = empty_workspace()
    _ensure_data_dir()
    _WORKSPACE_PATH.write_text(snap.model_dump_json(indent=2), encoding="utf-8")
    return snap


@router.get("/workspace")
def get_workspace() -> WorkspaceSnapshot:
    return _load_or_create_workspace()


@router.post("/workspace")
def post_workspace(body: WorkspaceSnapshot) -> WorkspaceSnapshot:
    _ensure_data_dir()
    try:
        _WORKSPACE_PATH.write_text(body.model_dump_json(indent=2), encoding="utf-8")
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"Could not write workspace: {e}") from e
    return body
