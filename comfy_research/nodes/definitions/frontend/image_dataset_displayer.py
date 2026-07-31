"""image_dataset_displayer — FrontendNodeDef-channel definition."""
from __future__ import annotations

from comfy_research.nodes.registry import frontend_node_def
from comfy_research.nodes.schema import InPort, PortAccept, EnumField, IntField, FrontendNodeDef, FrontendSpec

DEF = frontend_node_def(
    FrontendNodeDef(
        type="image_dataset_displayer",
        label="Image dataset displayer",
        category="visualization",
        hint="Preview vision dataset rows (MNIST, Gaussian blob, shape world, hole counting): wire dataset → train/test split, index range, grid columns.",
        fields=(
            EnumField(key="split", label="Split", default="train"),
            EnumField(key="indexRange", label="Index Range", default="0-9"),
            IntField(key="columnsPerRow", label="Columns Per Row", default=5),
        ),
        defaults=(
            ("split", "train"),
            ("indexRange", "0-9"),
            ("columnsPerRow", 5),
        ),
        # cascade 分支逐字转写(VISION_DATASET_KINDS == vision_dataset
        # family 恰等,seam invariant 钉住)。
        ports=(
            InPort(id="dataset", accepts=(
                PortAccept(handles=("dataset",), source_family="vision_dataset"),
            )),
        ),
        frontend=FrontendSpec(component_key="ImageDatasetDisplayerNode"),
    )
)
