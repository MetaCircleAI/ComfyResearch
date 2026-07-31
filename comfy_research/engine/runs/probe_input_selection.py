from __future__ import annotations

from comfy_research.generated.node_capabilities import has_capability
from comfy_research.schemas.graph import NodeKind

NUMERIC_TOKEN_PROBE_TYPES = frozenset(
    {
        NodeKind.numeric_transformer_model.value,
        NodeKind.numeric_hyena_model.value,
    }
)


def uses_token_probe_input(model_type: object, *, has_tokens_per_input: bool = False) -> bool:
    return (
        has_capability(model_type, "token_model")
        or str(getattr(model_type, "value", model_type)) in NUMERIC_TOKEN_PROBE_TYPES
        or has_tokens_per_input
    )
