"""Minimum quadratic Bezier mode-connectivity evaluator for two classifier checkpoints."""
from __future__ import annotations

from comfy_research.nodes.registry import frontend_node_def
from comfy_research.nodes.schema import BoolField, FloatField, FrontendNodeDef, FrontendSpec, InPort, IntField, PortAccept


_LMC_PORTS = (
    InPort(id="checkpoint_a", accepts=(PortAccept(handles=("model",), source_type="model_checkpoint"),)),
    InPort(id="checkpoint_b", accepts=(PortAccept(handles=("model",), source_type="model_checkpoint"),)),
    InPort(id="model", accepts=(PortAccept(handles=("model",), source_family="canvas_full_model", source_io_mode="model"),)),
    InPort(id="dataset", accepts=(PortAccept(handles=("dataset",), source_type="cifar10_dataset"),)),
    InPort(id="loss", accepts=(PortAccept(handles=("loss",), source_type="cross_entropy_loss"),)),
)


DEF = frontend_node_def(
    FrontendNodeDef(
        type="observable_bezier_mode_connectivity",
        label="Bezier mode connectivity",
        category="analysis",
        hint="Optimize a quadratic Bezier control point between two checkpoints, then compare its loss path to linear interpolation.",
        family=("analysis_runner",),
        fields=(
            IntField(key="alphaSteps", label="Alpha Steps", default=21, min=2, max=201),
            IntField(key="curveOptimizationSteps", label="Curve Optimization Steps", default=500, min=1, max=10_000),
            IntField(key="curveSamplesPerStep", label="Curve Samples Per Step", default=4, min=1, max=32),
            IntField(key="curveBatchSize", label="Curve Batch Size", default=256, min=1, max=4096),
            FloatField(key="curveLearningRate", label="Curve Learning Rate", default=0.01, min=0.000001),
            BoolField(key="showTrainCurve", label="Show Train Curve", default=True),
            BoolField(key="showTestCurve", label="Show Test Curve", default=True),
            BoolField(key="recomputeBnStats", label="Recompute BN Stats", default=True),
            IntField(key="bnCalibrationBatches", label="BN Calibration Batches", default=100, min=1, max=100),
            IntField(key="evalBatchSize", label="Eval Batch Size", default=256, min=1, max=4096),
        ),
        defaults=(
            ("alphaSteps", 21),
            ("curveOptimizationSteps", 500),
            ("curveSamplesPerStep", 4),
            ("curveBatchSize", 256),
            ("curveLearningRate", 0.01),
            ("showTrainCurve", True),
            ("showTestCurve", True),
            ("recomputeBnStats", True),
            ("bnCalibrationBatches", 100),
            ("evalBatchSize", 256),
            ("linearTrainLoss", []),
            ("linearTestLoss", []),
            ("bezierTrainLoss", []),
            ("bezierTestLoss", []),
            ("linearLossBarrier", None),
            ("bezierLossBarrier", None),
            ("runSummary", ""),
            ("lastError", ""),
        ),
        ports=_LMC_PORTS,
        frontend=FrontendSpec(component_key="BezierModeConnectivityNode"),
    )
)
