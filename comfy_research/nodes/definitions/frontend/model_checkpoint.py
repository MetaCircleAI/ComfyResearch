"""model_checkpoint — FrontendNodeDef-channel definition."""
from __future__ import annotations

from comfy_research.nodes.registry import frontend_node_def
from comfy_research.nodes.schema import InPort, PortAccept, EnumField, FrontendNodeDef, FrontendSpec

DEF = frontend_node_def(
    FrontendNodeDef(
        type="model_checkpoint",
        label="Model Checkpoint",
        category="checkpoint",
        hint="Save or load model + optimizer state.",
        fields=(
            EnumField(key="checkpoint_b64", label="Checkpoint b64", default=""),
            EnumField(key="memoryCheckpoint_b64", label="Memory Checkpoint b64", default=""),
            EnumField(key="checkpointSource", label="Checkpoint Source", default="memory"),
            EnumField(key="checkpointFileName", label="Checkpoint File Name", default=""),
        ),
        defaults=(
            ("checkpoint_b64", ""),
            ("memoryCheckpoint_b64", ""),
            ("checkpointSource", "memory"),
            ("checkpointFileName", ""),
        ),
        # cascade return-style 分支逐字转写(isTrainerLikeCanvasType 三型;
        # 使用 type accepts，不为此引入新 family。
        ports=(
            InPort(id="model_checkpoint", accepts=(
                PortAccept(handles=("checkpoint",), source_type="trainer"),
                PortAccept(handles=("checkpoint",), source_type="crl_trainer"),
            )),
        ),
        frontend=FrontendSpec(component_key="ModelCheckpointNode"),
    )
)
