"""Preview vision dataset rows as PNG data URLs (MNIST / shape world / hole counting)."""

from __future__ import annotations

import base64
import io
from typing import Any, Literal

import numpy as np
from fastapi import APIRouter, HTTPException
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from pydantic import BaseModel, Field

from comfy_research.api.dataset_tensor import _scalar_int
from comfy_research.engine.datasets.vision_datasets_runtime import build_vision_numpy_arrays
from comfy_research.generated.node_capabilities import node_types_with_capability
from comfy_research.schemas.graph import NodeKind

router = APIRouter(prefix="/api", tags=["vision_dataset_gallery"])

_VISION_KINDS = frozenset(NodeKind(node_type) for node_type in node_types_with_capability("vision_dataset"))
_VISION_KIND_LABEL = ", ".join(node_type.value for node_type in sorted(_VISION_KINDS, key=lambda t: t.value))

MAX_IMAGES = 64
MAX_SIDE = 256


class VisionDatasetGalleryRequest(BaseModel):
    dataset_node_type: str
    dataset_data: dict[str, Any] = Field(default_factory=dict)
    split: Literal["train", "test"]
    start_index: int = Field(ge=0)
    end_index: int = Field(ge=0)


class VisionDatasetGalleryResponse(BaseModel):
    images: list[str]
    labels: list[int]
    warnings: list[str] = Field(default_factory=list)


def _encode_chw_png_data_url(x_chw: np.ndarray) -> str:
    if x_chw.ndim != 3:
        raise HTTPException(status_code=500, detail="Internal: expected CHW image tensor.")
    _c, h, w = (int(x_chw.shape[0]), int(x_chw.shape[1]), int(x_chw.shape[2]))
    if h > MAX_SIDE or w > MAX_SIDE:
        raise HTTPException(
            status_code=400,
            detail=f"Image side length too large for preview (max {MAX_SIDE}px per side).",
        )
    g = x_chw[0] if x_chw.shape[0] == 1 else np.mean(np.asarray(x_chw, dtype=np.float32), axis=0)
    g = np.asarray(g, dtype=np.float32)
    lo, hi = float(np.min(g)), float(np.max(g))
    if hi > lo:
        g = (g - lo) / (hi - lo)
    else:
        g = np.zeros_like(g)
    dpi = 96.0
    fig = Figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    FigureCanvasAgg(fig)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(g, cmap="gray", vmin=0, vmax=1, interpolation="nearest", aspect="equal")
    ax.axis("off")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", pad_inches=0, bbox_inches="tight")
    try:
        import matplotlib.pyplot as plt  # local import: keep module import graph small

        plt.close(fig)
    except Exception:
        pass
    b64 = base64.standard_b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


@router.post("/vision_dataset_gallery", response_model=VisionDatasetGalleryResponse)
def post_vision_dataset_gallery(body: VisionDatasetGalleryRequest) -> VisionDatasetGalleryResponse:
    try:
        nk = NodeKind(body.dataset_node_type)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Unknown node type: {body.dataset_node_type!r}.") from exc
    if nk not in _VISION_KINDS:
        raise HTTPException(
            status_code=400,
            detail=f"Only vision dataset nodes are supported: {_VISION_KIND_LABEL}.",
        )
    data = body.dataset_data or {}
    train_n = max(0, _scalar_int(data.get("trainSize"), 2048))
    test_n = max(0, _scalar_int(data.get("testSize"), 512))
    seed = _scalar_int(data.get("initSeed"), _scalar_int(data.get("seed"), 0))
    rng = np.random.default_rng(seed)
    x_tr, y_tr, x_te, y_te = build_vision_numpy_arrays(nk, data, train_n, test_n, rng)
    if body.split == "train":
        x, y = x_tr, y_tr
    else:
        if x_te is None or y_te is None or int(x_te.shape[0]) == 0:
            raise HTTPException(status_code=400, detail="No test split for this dataset (test size is 0).")
        x, y = x_te, y_te
    n = int(x.shape[0])
    if n <= 0:
        raise HTTPException(status_code=400, detail=f"No {body.split} samples available.")

    a = int(body.start_index)
    b = int(body.end_index)
    if a > b:
        a, b = b, a
    a = max(0, min(a, n - 1))
    b = max(0, min(b, n - 1))
    warnings: list[str] = []
    span = b - a + 1
    if span > MAX_IMAGES:
        b = a + MAX_IMAGES - 1
        b = min(b, n - 1)
        warnings.append(f"Showing at most {MAX_IMAGES} images per request.")

    images: list[str] = []
    labels: list[int] = []
    y1 = np.asarray(y)
    for i in range(a, b + 1):
        images.append(_encode_chw_png_data_url(x[i]))
        yi = int(y1[i]) if y1.ndim == 1 else int(y1[i].reshape(-1)[0])
        labels.append(yi)

    return VisionDatasetGalleryResponse(images=images, labels=labels, warnings=warnings)
