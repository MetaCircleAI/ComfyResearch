"""observable_viz_neuron_trajectory_2d — FrontendNodeDef-channel definition."""
from __future__ import annotations

from comfy_research.nodes.registry import frontend_node_def
from comfy_research.nodes.schema import FrontendNodeDef, FrontendSpec

DEF = frontend_node_def(
    FrontendNodeDef(
        type="observable_viz_neuron_trajectory_2d",
        label="Neuron trajectory 2D viz",
        category="visualization",
        defaults=None,  # hasDefaults:false
        frontend=FrontendSpec(component_key="NeuronTrajectory2dVizNode"),
    )
)
