import json
from pathlib import Path

import pytest

from comfy_research.engine.models.attention_only_model import AttentionTokenPredictBundle


TEMPLATES = Path("data/graph_library/templates")
SPARSE_ATTENTION_TEMPLATES = (
    "569dffac-0c43-450f-b02b-8a4e732a8ff8.json",
    "f47ae3d5-49ba-430c-95da-7608db1b3dfc.json",
)


@pytest.mark.parametrize("template_name", SPARSE_ATTENTION_TEMPLATES)
def test_sparse_attention_template_weight_selection_matches_attention_only_bundle(template_name: str) -> None:
    raw = json.loads((TEMPLATES / template_name).read_text(encoding="utf-8"))
    nodes = raw["document"]["nodes"]
    selector = next(node for node in nodes if node["id"] == "wex_er1")
    weight_node = next(node for node in nodes if node["type"] == "model_weight_tensors")
    model_parameter_names = {name for name, _ in AttentionTokenPredictBundle(4, 2, 2, 1).named_parameters()}
    selected_keys = selector["data"].get("selectedTensorKeys", [selector["data"]["selectedTensorKey"]])
    cached_keys = set(weight_node["data"]["weightTensorPayloads"])

    assert selector["data"]["selectedTensorKey"] == "block.w_q.weight"
    assert set(selected_keys) <= model_parameter_names
    assert cached_keys <= model_parameter_names
