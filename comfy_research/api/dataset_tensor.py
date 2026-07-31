from __future__ import annotations

import json
from typing import Any, Literal

import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

from comfy_research.nodes.provider_types import DatasetPreviewRequest
from comfy_research.nodes.registry import dataset_defs_previews
from comfy_research.engine.datasets.toy_language_dyck_runtime import build_dyck_lm_arrays_from_seed
from comfy_research.engine.datasets.toy_language_inspect import TOY_LANGUAGE_INSPECT_TYPES, toy_language_word_inspect_lines
from comfy_research.engine.datasets.toy_language_external_runtime import (
    build_cogs_arrays_from_seed,
    build_listops_arrays_from_seed,
    build_scan_arrays_from_seed,
    phi_style_lm_from_seed,
    tiny_stories_lm_from_seed,
)
from comfy_research.engine.datasets.toy_language_formal_runtime import build_formal_suite_lm_arrays_from_seed
from comfy_research.engine.datasets.toy_language_ngram_runtime import build_ngram_lm_arrays_from_seed
from comfy_research.engine.datasets.toy_language_pcfg_runtime import build_pcfg_lm_arrays_from_seed
from comfy_research.engine.datasets.toy_language_physics_lm_runtime import (
    build_biography_lm_arrays_from_seed,
    build_multi_hop_fact_chain_lm_arrays_from_seed,
    build_relation_tuple_lm_arrays_from_seed,
    build_synthetic_playground_lm_arrays_from_seed,
)

router = APIRouter(prefix="/api", tags=["dataset_tensor"])


class DatasetTensorRequest(BaseModel):
    dataset_node_id: str
    dataset_node_type: str
    dataset_data: dict[str, Any]
    graph_nodes: list[dict[str, Any]] = Field(default_factory=list)
    graph_edges: list[dict[str, Any]] = Field(default_factory=list)
    split: Literal["train", "test"]
    tensor_key: Literal["input", "output"]


def _scalar_int(x: Any, default: int) -> int:
    if isinstance(x, list):
        if not x:
            return default
        x = x[0]
    try:
        return int(x)
    except (TypeError, ValueError):
        return default


def _scalar_float(x: Any, default: float) -> float:
    if isinstance(x, list):
        if not x:
            return default
        x = x[0]
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _tensor_inspect_mode(data: dict[str, Any]) -> str:
    raw = data.get("inspectFormat", data.get("format"))
    s = _scalar_str(raw, "id").strip().lower()
    if s in ("word", "words", "text"):
        return "word"
    return "id"


def _scalar_str(x: Any, default: str) -> str:
    if isinstance(x, list):
        if not x:
            return default
        x = x[0]
    if x is None:
        return default
    return str(x)



















@router.post("/dataset_tensor")
def post_dataset_tensor(body: DatasetTensorRequest) -> Response:
    ds_type = body.dataset_node_type
    original_type = ds_type
    data = body.dataset_data or {}
    # Prefer a registered preview provider after alias normalization.
    _preview_provider = dataset_defs_previews().get(ds_type)
    if _preview_provider is not None:
        x_train, y_train, x_test, y_test = _preview_provider(
            DatasetPreviewRequest(
                original_type=original_type,
                effective_type=ds_type,
                data=data,
                graph_nodes=body.graph_nodes,
                graph_edges=body.graph_edges,
                dataset_node_id=body.dataset_node_id,
            )
        )
    elif ds_type == "pcfg_dataset":
        train_sz = max(0, _scalar_int(data.get("trainSize"), 800))
        test_sz = max(0, _scalar_int(data.get("testSize"), 200))
        x_train, y_train, x_test, y_test = build_pcfg_lm_arrays_from_seed(data, train_sz, test_sz)
    elif ds_type == "dyck_dataset":
        train_sz = max(0, _scalar_int(data.get("trainSize"), 800))
        test_sz = max(0, _scalar_int(data.get("testSize"), 200))
        x_train, y_train, x_test, y_test = build_dyck_lm_arrays_from_seed(data, train_sz, test_sz)
    elif ds_type == "ngram_language_dataset":
        train_sz = max(0, _scalar_int(data.get("trainSize"), 800))
        test_sz = max(0, _scalar_int(data.get("testSize"), 200))
        x_train, y_train, x_test, y_test = build_ngram_lm_arrays_from_seed(data, train_sz, test_sz)
    elif ds_type == "formal_language_suite_dataset":
        train_sz = max(0, _scalar_int(data.get("trainSize"), 800))
        test_sz = max(0, _scalar_int(data.get("testSize"), 200))
        x_train, y_train, x_test, y_test = build_formal_suite_lm_arrays_from_seed(data, train_sz, test_sz)
    elif ds_type == "scan_dataset":
        train_sz = max(0, _scalar_int(data.get("trainSize"), 800))
        test_sz = max(0, _scalar_int(data.get("testSize"), 200))
        x_train, y_train, x_test, y_test = build_scan_arrays_from_seed(data, train_sz, test_sz)
    elif ds_type == "cogs_dataset":
        train_sz = max(0, _scalar_int(data.get("trainSize"), 800))
        test_sz = max(0, _scalar_int(data.get("testSize"), 200))
        x_train, y_train, x_test, y_test = build_cogs_arrays_from_seed(data, train_sz, test_sz)
    elif ds_type == "listops_dataset":
        train_sz = max(0, _scalar_int(data.get("trainSize"), 800))
        test_sz = max(0, _scalar_int(data.get("testSize"), 200))
        x_train, y_train, x_test, y_test = build_listops_arrays_from_seed(data, train_sz, test_sz)
    elif ds_type == "tinystories_dataset":
        train_sz = max(0, _scalar_int(data.get("trainSize"), 800))
        test_sz = max(0, _scalar_int(data.get("testSize"), 200))
        x_train, y_train, x_test, y_test = tiny_stories_lm_from_seed(data, train_sz, test_sz)
    elif ds_type == "phi1_style_dataset":
        train_sz = max(0, _scalar_int(data.get("trainSize"), 800))
        test_sz = max(0, _scalar_int(data.get("testSize"), 200))
        x_train, y_train, x_test, y_test = phi_style_lm_from_seed(data, train_sz, test_sz)
    elif ds_type == "biography_lm_dataset":
        train_sz = max(0, _scalar_int(data.get("trainSize"), 800))
        test_sz = max(0, _scalar_int(data.get("testSize"), 200))
        x_train, y_train, x_test, y_test = build_biography_lm_arrays_from_seed(data, train_sz, test_sz)
    elif ds_type == "relation_tuple_dataset":
        train_sz = max(0, _scalar_int(data.get("trainSize"), 800))
        test_sz = max(0, _scalar_int(data.get("testSize"), 200))
        x_train, y_train, x_test, y_test = build_relation_tuple_lm_arrays_from_seed(data, train_sz, test_sz)
    elif ds_type == "synthetic_playground_dataset":
        train_sz = max(0, _scalar_int(data.get("trainSize"), 800))
        test_sz = max(0, _scalar_int(data.get("testSize"), 200))
        x_train, y_train, x_test, y_test = build_synthetic_playground_lm_arrays_from_seed(data, train_sz, test_sz)
    elif ds_type == "multi_hop_fact_chain_dataset":
        train_sz = max(0, _scalar_int(data.get("trainSize"), 800))
        test_sz = max(0, _scalar_int(data.get("testSize"), 200))
        x_train, y_train, x_test, y_test = build_multi_hop_fact_chain_lm_arrays_from_seed(data, train_sz, test_sz)
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported dataset node type: {ds_type}.")

    if body.split == "train":
        arr = x_train if body.tensor_key == "input" else y_train
    else:
        arr = x_test if body.tensor_key == "input" else y_test
    if arr.shape[0] == 0:
        raise HTTPException(status_code=400, detail=f"No {body.split} samples available for this dataset.")

    if ds_type in TOY_LANGUAGE_INSPECT_TYPES and _tensor_inspect_mode(data) == "word":
        try:
            lines, shape_list, note = toy_language_word_inspect_lines(ds_type, data, body.split, body.tensor_key)
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Word inspect preview failed for {ds_type}: {exc}",
            ) from exc
        payload: dict[str, Any] = {
            "inspect": "word",
            "shape": shape_list,
            "tensorKey": body.tensor_key,
            "split": body.split,
            "lines": lines,
        }
        if note:
            payload["note"] = note
        return JSONResponse(
            content=payload,
            headers={"X-Dataset-Inspect": "word", "X-Tensor-Shape": json.dumps(shape_list)},
        )

    f32 = np.ascontiguousarray(arr, dtype=np.float32)
    return Response(
        content=f32.tobytes(),
        media_type="application/octet-stream",
        headers={"X-Tensor-Shape": json.dumps(list(f32.shape))},
    )
