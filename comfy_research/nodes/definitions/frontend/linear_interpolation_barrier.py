"""Post-training linear mode connectivity evaluator for two classifier checkpoints."""
from __future__ import annotations

from comfy_research.nodes.registry import frontend_node_def
from comfy_research.nodes.schema import BoolField, FloatField, FrontendNodeDef, FrontendSpec, InPort, IntField, PortAccept


DEF = frontend_node_def(
    FrontendNodeDef(
        type="observable_linear_interpolation_barrier",
        label="Linear interpolation barrier",
        category="analysis",
        hint="Evaluate train/test loss and accuracy along theta(alpha) between two model checkpoints.",
        family=("analysis_runner",),
        fields=(
            FloatField(key="alphaMin", label="Alpha Min", default=0.0),
            FloatField(key="alphaMax", label="Alpha Max", default=1.0),
            IntField(key="alphaSteps", label="Alpha Steps", default=21, min=2, max=201),
            BoolField(key="showTrainCurve", label="Show Train Curve", default=True),
            BoolField(key="showTestCurve", label="Show Test Curve", default=True),
            BoolField(key="recomputeBnStats", label="Recompute BN Stats", default=False),
            IntField(key="bnCalibrationBatches", label="BN Calibration Batches", default=100, min=1, max=100),
            IntField(key="evalBatchSize", label="Eval Batch Size", default=256, min=1, max=4096),
        ),
        defaults=(
            ("alphaMin", 0.0),
            ("alphaMax", 1.0),
            ("alphaSteps", 21),
            ("showTrainCurve", True),
            ("showTestCurve", True),
            ("recomputeBnStats", False),
            ("bnCalibrationBatches", 100),
            ("evalBatchSize", 256),
            ("lossBarrier", None),
            ("accuracyDrop", None),
            ("interpolationCurvePng", ""),
            ("lastError", ""),
        ),
        ports=(
            InPort(id="checkpoint_a", accepts=(PortAccept(handles=("model",), source_type="model_checkpoint"),)),
            InPort(id="checkpoint_b", accepts=(PortAccept(handles=("model",), source_type="model_checkpoint"),)),
            InPort(id="model", accepts=(PortAccept(handles=("model",), source_family="canvas_full_model", source_io_mode="model"),)),
            InPort(id="dataset", accepts=(PortAccept(handles=("dataset",), source_type="cifar10_dataset"),)),
            InPort(id="loss", accepts=(PortAccept(handles=("loss",), source_type="cross_entropy_loss"),)),
        ),
        frontend=FrontendSpec(component_key="LinearInterpolationBarrierNode"),
    )
)
