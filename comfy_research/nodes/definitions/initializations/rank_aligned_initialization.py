"""Balanced two-factor initialization for the linear Rank-Collapse toy."""
from __future__ import annotations

from comfy_research.nodes.registry import initialization_def
from comfy_research.nodes.schema import EnumField, FloatField, FrontendSpec, InitializationDef, IntField

DEF = initialization_def(
    InitializationDef(
        type="rank_aligned_initialization",
        label="Rank-aligned initialization",
        hint=(
            "Balanced two-factor initialization. With orthogonal Figure-5 samples S=I it gives "
            "the paper condition F^T F = W W^T; on legacy repeated-sample datasets it only gives "
            "factor-Gram alignment. The frequency-structured basis is the verified N=60, C=4, "
            "d in {2, 3} geometric-frequency protocol. robust_v1 locks the searched input "
            "rotation angle/plane to 3.0/1 and output rotation angle/plane to 1.7/2. Both "
            "Linear biases are pinned to zero."
        ),
        fields=(
            EnumField(
                key="basisMode",
                label="Basis Mode",
                default="random_orthogonal",
                options=("random_orthogonal", "frequency_structured"),
                manifest_options=True,
                sweepable=False,
            ),
            EnumField(
                key="structuredProfile",
                label="Structured Profile",
                default="robust_v1",
                options=("robust_v1",),
                manifest_options=True,
                sweepable=False,
            ),
            FloatField(key="amplitude", label="Amplitude", default=1e-4, min=0),
            FloatField(key="scale", label="Singular Scale", default=1.2, min=0),
            FloatField(key="singularRatio", label="Singular Ratio", default=1.3, min=1),
            FloatField(
                key="frequencyRatio",
                label="Frequency Ratio",
                default=2.0,
                min=1,
                sweepable=False,
            ),
            FloatField(
                key="perturbationScale",
                label="Perturbation Scale",
                default=0.05,
                min=0,
                sweepable=False,
            ),
            FloatField(
                key="inputRotationAngleRadians",
                label="Input Rotation Angle (rad)",
                default=3.0,
                sweepable=False,
            ),
            IntField(
                key="inputRotationPlane",
                label="Input Rotation Plane",
                default=1,
                min=0,
                max=2,
                sweepable=False,
            ),
            FloatField(
                key="outputRotationAngleRadians",
                label="Output Rotation Angle (rad)",
                default=1.7,
                sweepable=False,
            ),
            IntField(
                key="outputRotationPlane",
                label="Output Rotation Plane",
                default=2,
                min=0,
                max=2,
                sweepable=False,
            ),
            IntField(key="seed", label="Seed", default=0, min=0),
        ),
        frontend=FrontendSpec(
            component_key="SaxeInitializationNode",
            codegen_key="rank_aligned_initialization",
        ),
    )
)
