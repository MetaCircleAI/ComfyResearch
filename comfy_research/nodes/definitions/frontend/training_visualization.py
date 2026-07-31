"""training_visualization — FrontendNodeDef-channel definition."""
from __future__ import annotations

from comfy_research.nodes.registry import frontend_node_def
from comfy_research.nodes.schema import InPort, PortAccept, BoolField, EnumField, FrontendNodeDef, FrontendSpec

DEF = frontend_node_def(
    FrontendNodeDef(
        type="training_visualization",
        label="Training visualization",
        category="visualization",
        hint="Plots loss curves during and after training.",
        family=("canvas_comment_source",),
        fields=(
            BoolField(key="logScaleX", label="Log Scale X", default=False),
            BoolField(key="logScaleY", label="Log Scale Y", default=False),
            BoolField(key="showTrainCurve", label="Show Train Curve", default=True),
            BoolField(key="showTestCurve", label="Show Test Curve", default=True),
            EnumField(key="yPlotMetric", label="Y Plot Metric", default="loss"),
            EnumField(key="lossView", label="Loss View", default="loss"),
        ),
        defaults=(
            ("logScaleX", False),
            ("logScaleY", False),
            ("showTrainCurve", True),
            ("showTestCurve", True),
            ("yPlotMetric", "loss"),
            ("lossView", "loss"),
        ),
        # cascade 反式守卫分支逐字转写(trainer 三型 × loss_results;
        # 原分支对本 target 全 th return-style,ports 全权语义恒等)。
        ports=(
            InPort(id="tensor_list", accepts=(
                PortAccept(handles=("loss_results",), source_type="trainer"),
                PortAccept(handles=("loss_results",), source_type="crl_trainer"),
            )),
        ),
        frontend=FrontendSpec(component_key="TrainingVisualizationNode"),
    )
)
