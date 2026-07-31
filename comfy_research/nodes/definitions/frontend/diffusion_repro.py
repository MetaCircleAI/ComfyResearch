"""Frontend-only nodes that invoke the diffusion reproducibility API."""
from __future__ import annotations

from comfy_research.nodes.registry import frontend_node_def
from comfy_research.nodes.schema import FloatField, FrontendNodeDef, FrontendSpec, InPort, IntField, PortAccept


SAMPLER = frontend_node_def(
    FrontendNodeDef(
        type="deterministic_diffusion_sampler",
        label="Deterministic diffusion sampler",
        category="analysis",
        hint="DDIM-like sampler with a fixed seed; generated images persist as runtime data.",
        fields=(
            IntField(key="noiseSeed", label="Noise Seed", default=0, min=0),
            IntField(key="sampleCount", label="Sample Count", default=64, min=1, max=512),
            IntField(key="numSteps", label="Sampling Steps", default=50, min=2),
        ),
        defaults=(("noiseSeed", 0), ("sampleCount", 64), ("numSteps", 50), ("runId", ""), ("previewGrid", "")),
        ports=(InPort(id="checkpoint", accepts=(PortAccept(handles=("model_checkpoint",), source_type="model_checkpoint"),)),),
        frontend=FrontendSpec(component_key="DeterministicDiffusionSamplerNode"),
    )
)


PAIRED = frontend_node_def(
    FrontendNodeDef(
        type="observable_paired_generation_similarity",
        label="Paired generation similarity",
        category="analysis",
        fields=(),
        defaults=(("meanMae", None), ("meanMse", None), ("lastError", "")),
        ports=(
            InPort(id="sampler_a", accepts=(PortAccept(handles=("samples",), source_type="deterministic_diffusion_sampler"),)),
            InPort(id="sampler_b", accepts=(PortAccept(handles=("samples",), source_type="deterministic_diffusion_sampler"),)),
        ),
        frontend=FrontendSpec(component_key="PairedGenerationSimilarityNode"),
    )
)


RP = frontend_node_def(
    FrontendNodeDef(
        type="observable_rp_score_sscd",
        label="RP reproducibility score",
        category="analysis",
        fields=(FloatField(key="threshold", label="Threshold", default=0.95, min=0.0, max=1.0),),
        defaults=(("threshold", 0.95), ("rp", None), ("meanSimilarity", None), ("lastError", "")),
        ports=(
            InPort(id="sampler_a", accepts=(PortAccept(handles=("samples",), source_type="deterministic_diffusion_sampler"),)),
            InPort(id="sampler_b", accepts=(PortAccept(handles=("samples",), source_type="deterministic_diffusion_sampler"),)),
        ),
        frontend=FrontendSpec(component_key="RpScoreSscdNode"),
    )
)


NEAREST = frontend_node_def(
    FrontendNodeDef(
        type="observable_nearest_train_gl",
        label="Nearest-train GL",
        category="analysis",
        fields=(FloatField(key="glThreshold", label="GL Threshold", default=0.95, min=0.0, max=1.0),),
        defaults=(("glThreshold", 0.95), ("glScore", None), ("lastError", "")),
        ports=(
            InPort(id="generated", accepts=(PortAccept(handles=("samples",), source_type="deterministic_diffusion_sampler"),)),
            InPort(id="train_dataset", accepts=(PortAccept(handles=("dataset",), source_type="cifar10_dataset"),)),
        ),
        frontend=FrontendSpec(component_key="NearestTrainGlNode"),
    )
)
