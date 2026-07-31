"""symbolic_func_dataset — NodeDef-channel definition。程序化生成自 manifest。"""
from __future__ import annotations

from comfy_research.nodes.definitions.datasets._singleton_common import (
    materialize_symbolic_func,
)
from comfy_research.nodes.definitions.datasets._linear_common import preview_direct_arrays
from comfy_research.nodes.registry import dataset_def, dataset_materializer_for, dataset_preview_for
from comfy_research.nodes.schema import DatasetDef, EnumField, FloatField, FrontendSpec, IntField

DEF = dataset_def(
    DatasetDef(
        type="symbolic_func_dataset",
        label="Symbolic func dataset",
        hint="LaTeX y(x) + numeric params → sampled (x, y) for the trainer (SymPy on server).",
        family=("vector_regression_dataset", "diffusion_noise_dataset", "activation_sample_dataset", "canvas_dataset_source", "canvas_activation_dataset_source", "canvas_trainer_autoconnect_dataset", "dataset_tensor_direct_arrays",),
        fields=(
            EnumField(key="equationLatex", label="Equation Latex", default='\\exp(\\sin(\\pi x_1) + x_2^2)'),
            IntField(key="inputDim", label="Input Dim", default=2, min=1),
            IntField(key="outputDim", label="Output Dim", default=1, min=1),
            EnumField(key="inputDistribution", label="Input Distribution", default='standard_normal', options=('standard_normal', 'uniform_neg1_1', 'uniform_0_1')),
            EnumField(key="evaluationPrecision", label="Evaluation Precision", default='input', options=('input', 'float64')),
            EnumField(key="outputDistribution", label="Output Distribution", default='deterministic', options=('additive_gaussian', 'deterministic')),
            IntField(key="trainSize", label="Train Size", default=500, min=1),
            IntField(key="testSize", label="Test Size", default=0, min=0),
            FloatField(key="noiseLevel", label="Noise Level", default=0, min=0),
            IntField(key="seed", label="Seed", default=0),
            EnumField(key="samplingMode", label="Sampling Mode", default='fixed'),
        ),
        frontend=FrontendSpec(component_key="SymbolicFuncDatasetNode", codegen_key="symbolic_func_dataset"),
    )
)
dataset_materializer_for(DEF)(materialize_symbolic_func)
dataset_preview_for(DEF)(preview_direct_arrays)
