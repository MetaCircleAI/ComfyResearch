"""trainer — TrainerDef-channel thin definition.

This node has no provider because its runtime is the prepare pipeline. The
registry uses TrainerNode with spec-level fields. computeDevice is not
sweepable, and disableExtraObservables is boolean.
"""
from __future__ import annotations

from comfy_research.nodes.registry import trainer_def
from comfy_research.nodes.schema import BoolField, EnumField, FloatField, FrontendSpec, IntField, TrainerDef

DEF = trainer_def(
    TrainerDef(
        type="trainer",
        label="Trainer",
        hint="Training loop: datasets, model, optimizer, loss, and observables.",
        family=("trainer_runner",),
        fields=(
            IntField(key="trainingSteps", label="Training steps", default=1000),
            IntField(key="logFrequency", label="Log frequency", default=10),
            EnumField(
                key="logSchedule",
                label="Log schedule",
                default="fixed_interval",
                options=("fixed_interval", "idnns_logspace"),
                manifest_options=True,
                sweepable=False,
            ),
            IntField(key="logSamples", label="Log-space samples", default=1800, min=1),
            EnumField(
                key="logAggregation",
                label="Log aggregation",
                default="last_batch",
                options=("last_batch", "interval_sample_mean"),
                manifest_options=True,
                sweepable=False,
            ),
            EnumField(
                key="logTiming",
                label="Log timing",
                default="post_update",
                options=("post_update", "pre_update"),
                manifest_options=True,
                sweepable=False,
            ),
            EnumField(
                key="testEvaluation",
                label="Test evaluation",
                default="log_ticks",
                options=("log_ticks", "final_only", "disabled"),
                manifest_options=True,
                sweepable=False,
            ),
            IntField(
                key="trainSeed",
                label="Train seed",
                default=-1,
                min=-1,
                sweepable=False,
            ),
            EnumField(key="computeDevice", label="Compute device", default="cpu",
                      options=("cpu", "auto", "cuda", "mps"), manifest_options=True, sweepable=False),
            IntField(key="batchSize", label="Batch size", default=-1),
            EnumField(
                key="minibatchSampling",
                label="Minibatch sampling",
                default="independent_step",
                options=("independent_step", "epoch_shuffle", "affine_epoch"),
                manifest_options=True,
                sweepable=False,
            ),
            IntField(
                key="minibatchSeed",
                label="Minibatch seed",
                default=-1,
                min=-1,
                sweepable=False,
            ),
            FloatField(key="gradClipMaxNorm", label="Gradient clip max norm", default=0),
            BoolField(key="disableExtraObservables", label="Disable extra observables", default=False),
        ),
        frontend=FrontendSpec(component_key="TrainerNode", codegen_key="trainer"),
    )
)
