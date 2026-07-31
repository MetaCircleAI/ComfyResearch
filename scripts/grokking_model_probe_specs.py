#!/usr/bin/env python3
"""Print weight + representation probe specs for grokking_physics_demo MLP_token model."""
from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from comfy_research.engine.analysis.model_weight_materialize import run_model_weight_specs
from comfy_research.engine.analysis.representation_specs import run_model_representation_specs
from comfy_research.schemas.graph import Node, NodeKind, Position

MODEL_ID = "grok-model-0"


def main() -> None:
    data = {
        "vocabSize": 59,
        "embedDim": 32,
        "tokensPerInput": 2,
        "depth": 2,
        "width": 64,
        "activation": "relu",
        "tieWeights": "yes",
        "seed": 9196,
    }
    nodes = [
        Node(
            id=MODEL_ID,
            type=NodeKind.mlp_token_model,
            position=Position(x=0, y=0),
            data=data,
        )
    ]
    weights = run_model_weight_specs(nodes, [], MODEL_ID)["specs"]
    reps = run_model_representation_specs(nodes, [], MODEL_ID)["entries"]
    json.dump({"weights": weights, "representations": reps}, sys.stdout)


if __name__ == "__main__":
    main()
