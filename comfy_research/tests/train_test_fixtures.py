from __future__ import annotations

from typing import Any


def minimal_cpu_train_request(
    *,
    optimizer_type: str = "adam_optimizer",
    training_steps: int = 3,
) -> dict[str, Any]:
    return {
        "trainer_node_id": "trainer",
        "nodes": [
            {
                "id": "dataset",
                "type": "linear_dataset",
                "data": {
                    "inputDim": 2,
                    "outputDim": 1,
                    "trainSize": 8,
                    "testSize": 4,
                    "noiseLevel": 0,
                    "seed": 0,
                    "samplingMode": "fixed",
                },
            },
            {
                "id": "model",
                "type": "mlp_model",
                "data": {
                    "inputDim": 2,
                    "outputDim": 1,
                    "depth": 1,
                    "width": 4,
                    "activation": "relu",
                    "seed": 0,
                },
            },
            {
                "id": "optimizer",
                "type": optimizer_type,
                "data": {"learningRate": 0.01},
            },
            {"id": "loss", "type": "mse_loss", "data": {}},
            {
                "id": "trainer",
                "type": "trainer",
                "data": {
                    "trainingSteps": training_steps,
                    "logFrequency": 1,
                    "batchSize": -1,
                    "computeDevice": "cpu",
                },
            },
        ],
        "edges": [
            {
                "id": "dataset-trainer",
                "source": "dataset",
                "target": "trainer",
                "sourceHandle": "dataset",
                "targetHandle": "dataset",
            },
            {
                "id": "model-trainer",
                "source": "model",
                "target": "trainer",
                "sourceHandle": "model",
                "targetHandle": "model",
            },
            {
                "id": "optimizer-trainer",
                "source": "optimizer",
                "target": "trainer",
                "sourceHandle": "optimizer",
                "targetHandle": "optimizer",
            },
            {
                "id": "loss-trainer",
                "source": "loss",
                "target": "trainer",
                "sourceHandle": "loss",
                "targetHandle": "loss",
            },
        ],
    }
