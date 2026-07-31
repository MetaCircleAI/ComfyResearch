"""parametric_path_sampler — FrontendNodeDef-channel definition.

纯前端 analysis runner(Run 按钮 POST /api/parametric_path_sampler,endpoint
后续 E 批落地)。ports 转写自 repro-reference ResearchCanvas 的 cascade 分支:
checkpoint_sb/lb = model_checkpoint;model = model_checkpoint ∨ fullModelModelSocket
(canvas_full_model × ioMode model);dataset = *_dataset × sh dataset
(canvas_dataset_source family 转写);loss = mse/cross_entropy。
"""
from __future__ import annotations

from comfy_research.nodes.registry import frontend_node_def
from comfy_research.nodes.schema import (
    BoolField,
    EnumField,
    FrontendNodeDef,
    FrontendSpec,
    InPort,
    IntField,
    PortAccept,
)

DEF = frontend_node_def(
    FrontendNodeDef(
        type="parametric_path_sampler",
        label="Parametric path sampler",
        category="analysis",
        hint="Linear path w(α)=w_SB+α(w_LB−w_SB); eval loss/accuracy vs α.",
        family=("analysis_runner",),
        fields=(
            IntField(key="alphaMin", label="Alpha Min", default=-1),
            IntField(key="alphaMax", label="Alpha Max", default=2),
            IntField(key="alphaSteps", label="Alpha Steps", default=50, min=2),
            EnumField(key="metric", label="Metric", default="loss"),
            EnumField(key="split", label="Split", default="test"),
            EnumField(key="computeDevice", label="Compute Device", default="auto"),
            BoolField(key="remoteGpu", label="Remote Gpu", default=False),
            EnumField(key="seriesLabel", label="Series Label", default="parametric path"),
        ),
        defaults=(
            ("alphaMin", -1),
            ("alphaMax", 2),
            ("alphaSteps", 50),
            ("metric", "loss"),
            ("split", "test"),
            ("computeDevice", "auto"),
            ("remoteGpu", False),
            ("alphaSeries", []),
            ("valueSeries", []),
            ("runSummary", None),
            ("runError", None),
            ("seriesLabel", "parametric path"),
        ),
        ports=(
            InPort(id="checkpoint_sb", accepts=(
                PortAccept(handles=("model",), source_type="model_checkpoint"),
            )),
            InPort(id="checkpoint_lb", accepts=(
                PortAccept(handles=("model",), source_type="model_checkpoint"),
            )),
            InPort(id="model", accepts=(
                PortAccept(handles=("model",), source_type="model_checkpoint"),
                PortAccept(handles=("model",), source_family="canvas_full_model", source_io_mode="model"),
            )),
            InPort(id="dataset", accepts=(
                PortAccept(handles=("dataset",), source_family="canvas_dataset_source"),
            )),
            InPort(id="loss", accepts=(
                PortAccept(handles=("loss",), source_type="mse_loss"),
                PortAccept(handles=("loss",), source_type="cross_entropy_loss"),
            )),
        ),
        frontend=FrontendSpec(component_key="ParametricPathSamplerNode"),
    )
)
