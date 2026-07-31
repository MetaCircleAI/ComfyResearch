"""Multi-project workspace snapshot (persisted as workspace.json)."""

from __future__ import annotations

import uuid
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from comfy_research.schemas.graph import GraphDocument, empty_graph


class WorkspaceCanvas(BaseModel):
    """A project's graph (nodes, edges, viewport)."""

    id: str
    title: str = "Canvas"
    document: GraphDocument = Field(default_factory=empty_graph)


class WorkspaceProject(BaseModel):
    """A project owns exactly one canvas; use a separate project for a controlled variant."""

    id: str
    title: str = "Project"
    canvas: WorkspaceCanvas


class WorkspaceSnapshot(BaseModel):
    """Root document for GET/POST /api/workspace."""

    version: Literal[3] = 3
    active_project_id: str
    projects: list[WorkspaceProject] = Field(default_factory=list)

    @model_validator(mode="after")
    def _active_project_exists(self) -> WorkspaceSnapshot:
        ids = {p.id for p in self.projects}
        if not self.projects:
            raise ValueError("workspace must have at least one project")
        if self.active_project_id not in ids:
            raise ValueError("active_project_id must match a project id")
        return self


def new_workspace_from_graph_document(doc: GraphDocument) -> WorkspaceSnapshot:
    """Build a v3 workspace around one graph document."""
    pid = str(uuid.uuid4())
    cid = str(uuid.uuid4())
    return WorkspaceSnapshot(
        active_project_id=pid,
        projects=[
            WorkspaceProject(
                id=pid,
                title="Project",
                canvas=WorkspaceCanvas(id=cid, title="Main", document=doc),
            )
        ],
    )


def empty_workspace() -> WorkspaceSnapshot:
    """One empty project with one empty canvas."""
    return new_workspace_from_graph_document(empty_graph())
