from __future__ import annotations

import json

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response

from comfy_research.engine.analysis.activation_run_store import get_activation_numpy

router = APIRouter(prefix="/api", tags=["activation_tensor"])


@router.get("/activation_tensor")
def get_activation_tensor(
    run_id: str = Query(..., min_length=8),
    rep_id: str = Query(..., min_length=1),
) -> Response:
    """Return raw float32 row-major bytes for one activation tensor (lazy browser fetch)."""
    arr = get_activation_numpy(run_id, rep_id)
    if arr is None:
        raise HTTPException(
            status_code=404,
            detail="Activation run not found or expired — collect activations again.",
        )
    f32 = np.ascontiguousarray(arr, dtype=np.float32)
    return Response(
        content=f32.tobytes(),
        media_type="application/octet-stream",
        headers={"X-Tensor-Shape": json.dumps(list(f32.shape))},
    )
