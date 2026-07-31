"""Released-IDNNs fan-in initialization node."""

from __future__ import annotations

from comfy_research.nodes.registry import initialization_def
from comfy_research.nodes.schema import FrontendSpec, InitializationDef, IntField


DEF = initialization_def(
    InitializationDef(
        type="idnns_initialization",
        label="IDNNs initialization",
        hint=(
            "Re-initialize every Linear layer with the released IDNNs convention: "
            "fan-in truncated-normal weights (two standard deviations) and zero biases."
        ),
        fields=(IntField(key="seed", label="Seed", default=0, min=0, sweepable=False),),
        frontend=FrontendSpec(component_key="IdnnsInitializationNode"),
    )
)
