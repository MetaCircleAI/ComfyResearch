"""crl_trainer — TrainerDef-channel thin definition.

computeDevice is not sweepable, and disableEntropy is boolean; the remaining
fields provide ten sweep axes. Code generation uses crlServerSideStub.
"""
from __future__ import annotations

from comfy_research.nodes.registry import trainer_def
from comfy_research.nodes.schema import BoolField, EnumField, FloatField, FrontendSpec, IntField, TrainerDef

DEF = trainer_def(
    TrainerDef(
        type="crl_trainer",
        label="CRL trainer (PyTorch)",
        hint="Contrastive RL (PyTorch): env, CRL residual MLP, optimizer, replay + InfoNCE + actor.",
        family=("trainer_runner",),
        fields=(
            IntField(key="trainingSteps", label="Training steps", default=40),
            IntField(key="logFrequency", label="Log frequency", default=5),
            EnumField(key="computeDevice", label="Compute device", default="cpu",
                      options=("cpu", "auto", "cuda", "mps"), manifest_options=True, sweepable=False),
            IntField(key="batchSize", label="Batch size", default=32),
            IntField(key="unrollLength", label="Unroll length", default=24),
            IntField(key="sgdStepsPerTrainStep", label="SGD steps per train step", default=4),
            FloatField(key="gamma", label="Gamma", default=0.99),
            FloatField(key="logsumexpPenaltyCoeff", label="Logsumexp penalty coeff", default=0.1),
            FloatField(key="entropyParam", label="Entropy param", default=0.5),
            BoolField(key="disableEntropy", label="Disable entropy", default=False),
            IntField(key="maxReplayChunks", label="Max replay chunks", default=200),
            IntField(key="seed", label="Seed", default=0),
        ),
        frontend=FrontendSpec(component_key="CrlTrainerNode", codegen_key="crl_trainer"),
    )
)
