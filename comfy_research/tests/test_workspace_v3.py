import json

from comfy_research.api import workspace as workspace_api
from comfy_research.schemas.workspace import empty_workspace


def test_empty_workspace_uses_v3_single_canvas_contract() -> None:
    snapshot = empty_workspace()

    assert snapshot.version == 3
    assert len(snapshot.projects) == 1
    assert snapshot.projects[0].canvas.document.version == 1


def test_fresh_workspace_file_uses_v3_contract(tmp_path, monkeypatch) -> None:
    workspace_path = tmp_path / "workspace.json"
    monkeypatch.setattr(workspace_api, "_WORKSPACE_PATH", workspace_path)

    snapshot = workspace_api.get_workspace()
    persisted = json.loads(workspace_path.read_text(encoding="utf-8"))

    assert snapshot.version == 3
    assert persisted["version"] == 3
    assert persisted["projects"][0]["canvas"]["document"]["version"] == 1


def test_existing_v2_workspace_migrates_all_canvases_to_v3_projects(tmp_path, monkeypatch) -> None:
    workspace_path = tmp_path / "workspace.json"
    workspace_path.write_text(
        json.dumps(
            {
                "version": 2,
                "active_project_id": "project-1",
                "projects": [
                    {
                        "id": "project-1",
                        "title": "Experiment",
                        "active_canvas_id": "canvas-2",
                        "canvases": [
                            {
                                "id": "canvas-1",
                                "title": "Baseline",
                                "document": {
                                    "version": 1,
                                    "nodes": [],
                                    "edges": [],
                                    "viewport": {"x": 10, "y": 20, "zoom": 0.5},
                                },
                            },
                            {
                                "id": "canvas-2",
                                "title": "Active",
                                "document": {
                                    "version": 1,
                                    "nodes": [],
                                    "edges": [],
                                    "viewport": {"x": 30, "y": 40, "zoom": 1.5},
                                },
                            },
                        ],
                        "workflow_hierarchy": None,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(workspace_api, "_WORKSPACE_PATH", workspace_path)

    snapshot = workspace_api.get_workspace()
    persisted = json.loads(workspace_path.read_text(encoding="utf-8"))

    assert snapshot.version == 3
    assert snapshot.active_project_id == "project-1"
    assert snapshot.projects[0].id == "project-1"
    assert snapshot.projects[0].canvas.id == "canvas-2"
    assert {project.canvas.id for project in snapshot.projects} == {"canvas-1", "canvas-2"}
    assert persisted == snapshot.model_dump(mode="json")


def test_existing_v2_workspace_is_backed_up_before_hierarchy_is_dropped(tmp_path, monkeypatch) -> None:
    workspace_path = tmp_path / "workspace.json"
    legacy = {
        "version": 2,
        "active_project_id": "project-1",
        "projects": [
            {
                "id": "project-1",
                "title": "Experiment",
                "active_canvas_id": "canvas-1",
                "canvases": [
                    {
                        "id": "canvas-1",
                        "title": "Baseline",
                        "document": {
                            "version": 1,
                            "nodes": [],
                            "edges": [],
                            "viewport": None,
                        },
                    }
                ],
                "workflow_hierarchy": {
                    "base_canvas_id": "canvas-1",
                    "sweeps": [],
                },
            }
        ],
    }
    workspace_path.write_text(json.dumps(legacy), encoding="utf-8")
    monkeypatch.setattr(workspace_api, "_WORKSPACE_PATH", workspace_path)

    snapshot = workspace_api.get_workspace()
    backup = json.loads(workspace_path.with_suffix(".v2.json").read_text(encoding="utf-8"))

    assert snapshot.version == 3
    assert backup == legacy


def test_flat_v2_workspace_is_upgraded_without_changing_its_canvas(tmp_path, monkeypatch) -> None:
    workspace_path = tmp_path / "workspace.json"
    legacy = {
        "version": 2,
        "active_project_id": "project-1",
        "projects": [
            {
                "id": "project-1",
                "title": "Experiment",
                "canvas": {
                    "id": "canvas-1",
                    "title": "Baseline",
                    "document": {
                        "version": 1,
                        "nodes": [],
                        "edges": [],
                        "viewport": {"x": 10, "y": 20, "zoom": 0.5},
                    },
                },
            }
        ],
    }
    workspace_path.write_text(json.dumps(legacy), encoding="utf-8")
    monkeypatch.setattr(workspace_api, "_WORKSPACE_PATH", workspace_path)

    snapshot = workspace_api.get_workspace()
    persisted = json.loads(workspace_path.read_text(encoding="utf-8"))

    assert snapshot.version == 3
    assert snapshot.active_project_id == "project-1"
    assert snapshot.projects[0].canvas.id == "canvas-1"
    assert persisted["projects"][0]["canvas"]["document"]["viewport"] == {
        "x": 10.0,
        "y": 20.0,
        "zoom": 0.5,
    }


def test_fresh_workspace_does_not_import_legacy_graph_file(tmp_path, monkeypatch) -> None:
    workspace_path = tmp_path / "workspace.json"
    legacy_graph_path = tmp_path / "graph.json"
    legacy_graph_path.write_text(
        json.dumps(
            {
                "version": 1,
                "nodes": [],
                "edges": [],
                "viewport": {"x": 42, "y": 0, "zoom": 1},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(workspace_api, "_WORKSPACE_PATH", workspace_path)

    assert not hasattr(workspace_api, "_LEGACY_GRAPH_PATH")
    snapshot = workspace_api.get_workspace()

    assert snapshot.projects[0].canvas.document.viewport is None
