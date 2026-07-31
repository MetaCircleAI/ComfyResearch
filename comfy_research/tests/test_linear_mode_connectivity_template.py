"""Structural contract for the small same-init LMC template."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_lmc_template_has_shared_data_model_and_two_train_seeds() -> None:
    doc = json.loads((ROOT / "data/graph_library/templates/repro-linear-mode-connectivity-cifar10.json").read_text())
    document = doc["document"]
    nodes = {node["id"]: node for node in document["nodes"]}
    edges = {(edge["source"], edge["target"], edge["targetHandle"]) for edge in document["edges"]}

    assert doc["tier"] == "small"
    assert doc["savedAt"] > 0
    assert nodes["lmc-model"]["data"]["seed"] == 0
    assert nodes["lmc-dataset"]["data"]["subsetSeed"] == 0
    assert nodes["lmc-dataset"]["data"]["trainSize"] == 50_000
    assert nodes["lmc-trainer-a"]["data"]["trainingSteps"] == 50_000
    assert nodes["lmc-trainer-a"]["data"]["batchSize"] == 128
    assert nodes["lmc-trainer-a"]["data"]["trainSeed"] == 0
    assert nodes["lmc-trainer-b"]["data"]["trainSeed"] == 1
    assert nodes["lmc-barrier"]["data"]["bnCalibrationBatches"] == 100
    assert ("lmc-model", "lmc-trainer-a", "model") in edges
    assert ("lmc-model", "lmc-trainer-b", "model") in edges
    assert ("lmc-checkpoint-a", "lmc-barrier", "checkpoint_a") in edges
    assert ("lmc-checkpoint-b", "lmc-barrier", "checkpoint_b") in edges
    assert ("lmc-checkpoint-a", "lmc-bezier", "checkpoint_a") in edges
    assert ("lmc-checkpoint-b", "lmc-bezier", "checkpoint_b") in edges
