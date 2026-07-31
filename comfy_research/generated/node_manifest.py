from __future__ import annotations

import json
from functools import lru_cache
from importlib.resources import files
from typing import Any


@lru_cache(maxsize=1)
def load_node_manifest() -> list[dict[str, Any]]:
    """Load the frontend-generated node manifest.

    The manifest is generated from ``frontend/src/graph/nodeRegistry.ts`` and is
    the handoff point for backend enum/catalog generation in the node-registry
    refactor.
    """
    text = files("comfy_research.generated").joinpath("node_manifest.json").read_text(encoding="utf-8")
    raw = json.loads(text)
    if not isinstance(raw, list):
        raise ValueError("node_manifest.json must contain a list")
    return [entry for entry in raw if isinstance(entry, dict)]

