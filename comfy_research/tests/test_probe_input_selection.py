from comfy_research.engine.runs.probe_input_selection import uses_token_probe_input
from comfy_research.generated.node_capabilities import node_types_with_capability
from comfy_research.schemas.graph import NodeKind


def test_token_probe_selection_uses_token_model_capability() -> None:
    for node_type in node_types_with_capability("token_model"):
        assert uses_token_probe_input(node_type)


def test_token_probe_selection_preserves_numeric_and_model_attribute_fallbacks() -> None:
    assert uses_token_probe_input(NodeKind.numeric_transformer_model)
    assert uses_token_probe_input(NodeKind.numeric_hyena_model)
    assert uses_token_probe_input(NodeKind.mlp_model, has_tokens_per_input=True)
    assert not uses_token_probe_input(NodeKind.mlp_model)
