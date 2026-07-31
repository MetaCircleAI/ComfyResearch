from __future__ import annotations

from comfy_research.schemas.graph import Node, NodeKind

_TYPE_ALIAS: dict[NodeKind, NodeKind] = {
    NodeKind.pairwise_rbf_layer: NodeKind.linear_layer,
    NodeKind.equivariant_message_layer: NodeKind.linear_layer,
    NodeKind.energy_readout_layer: NodeKind.linear_layer,
    NodeKind.relative_pose_encoder_layer: NodeKind.linear_layer,
    NodeKind.distance_contact_layer: NodeKind.linear_layer,
    NodeKind.hypothesis: NodeKind.comment,
    NodeKind.interatomic_eval_viz: NodeKind.tensor_viz_scatter,
    NodeKind.docking_pose_viz: NodeKind.tensor_viz_scatter,
    NodeKind.agent_trace_viz: NodeKind.table_viz,
}


def remap_ai4science_node_types(nodes: list[Node]) -> list[Node]:
    """Map AI4Science toy node kinds to existing executable node kinds."""
    remapped: list[Node] = []
    for node in nodes:
        mapped = _TYPE_ALIAS.get(node.type, node.type)
        if mapped == node.type:
            remapped.append(node)
            continue
        remapped.append(node.model_copy(update={"type": mapped}))
    return remapped
