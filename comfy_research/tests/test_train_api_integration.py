from __future__ import annotations

import json
import math

from fastapi.testclient import TestClient

from comfy_research.main import app
from comfy_research.tests.train_test_fixtures import minimal_cpu_train_request


def _ndjson_events(response_text: str) -> list[dict[str, object]]:
    return [json.loads(line) for line in response_text.splitlines() if line.strip()]


def test_post_train_streams_real_cpu_training_result() -> None:
    response = TestClient(app).post("/api/train", json=minimal_cpu_train_request())

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ndjson")

    events = _ndjson_events(response.text)
    progress = [event for event in events if event.get("type") == "progress"]
    complete = [event for event in events if event.get("type") == "complete"]

    assert [event["step"] for event in progress] == [0, 1, 2, 3]
    assert len(complete) == 1

    loss_history = complete[0]["loss_history"]
    assert isinstance(loss_history, list)
    assert len(loss_history) == 4
    assert all(
        isinstance(value, int | float) and math.isfinite(value)
        for value in loss_history
    )


def test_post_train_rejects_invalid_graph_before_streaming() -> None:
    response = TestClient(app).post(
        "/api/train",
        json={
            "trainer_node_id": "trainer",
            "nodes": [
                {
                    "id": "trainer",
                    "type": "trainer",
                    "data": {"trainingSteps": 1, "computeDevice": "cpu"},
                }
            ],
            "edges": [],
        },
    )

    assert response.status_code == 400
    assert response.headers["content-type"].startswith("application/json")
    assert response.json()["detail"]
