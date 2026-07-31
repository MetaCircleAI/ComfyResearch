from comfy_research.generated.node_kind import GeneratedNodeKind
from comfy_research.generated.node_manifest import load_node_manifest
from comfy_research.schemas.graph import NodeKind, TRANSITIONAL_SCHEMA_ONLY_NODE_KIND_VALUES


TRANSITIONAL_SCHEMA_ONLY_KINDS = {
    "dataset",
    "loss",
    "model",
    "observable",
    "observable_viz_embedding_trajectory",
    "observable_viz_relu_nonlinear",
    "observable_viz_user",
    "observable_viz_weight_l1",
    "observable_viz_weight_l2",
    "optimizer",
}


def test_generated_node_kind_matches_manifest() -> None:
    manifest_types = {str(entry["type"]) for entry in load_node_manifest()}
    generated_types = {kind.value for kind in GeneratedNodeKind}
    assert generated_types == manifest_types


def test_legacy_schema_node_kind_is_manifest_plus_known_transition_extras() -> None:
    manifest_types = {str(entry["type"]) for entry in load_node_manifest()}
    schema_types = {kind.value for kind in NodeKind}
    assert manifest_types <= schema_types
    assert schema_types - manifest_types == TRANSITIONAL_SCHEMA_ONLY_KINDS
    assert set(TRANSITIONAL_SCHEMA_ONLY_NODE_KIND_VALUES) == TRANSITIONAL_SCHEMA_ONLY_KINDS
