"""visualize_kan — FrontendNodeDef-channel definition."""
from __future__ import annotations

from comfy_research.nodes.registry import frontend_node_def
from comfy_research.nodes.schema import EnumField, FloatField, IntField, FrontendNodeDef, FrontendSpec

DEF = frontend_node_def(
    FrontendNodeDef(
        type="visualize_kan",
        label="Visualize KAN",
        category="visualization",
        hint="Runs pykan KAN.plot() when wired to a KAN or checkpoint; optional dataset → train/test split sets sample count and input law.",
        fields=(
            EnumField(key="plotPngBase64", label="Plot Png Base64", default=""),
            EnumField(key="datasetSampleSplit", label="Dataset Sample Split", default="train"),
            IntField(key="sampleCount", label="Sample Count", default=256),
            FloatField(key="plotScale", label="Plot Scale", default=0.35),
            EnumField(key="plotMetric", label="Plot Metric", default="backward"),
        ),
        defaults=(
            ("plotPngBase64", ""),
            ("lastPlotError", None),
            ("datasetSampleSplit", "train"),
            ("sampleCount", 256),
            ("plotScale", 0.35),
            ("plotMetric", "backward"),
        ),
        frontend=FrontendSpec(component_key="VisualizeKanNode"),
    )
)
