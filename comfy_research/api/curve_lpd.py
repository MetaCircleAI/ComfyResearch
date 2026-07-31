"""LPD curve-phase prediction API for the CurveStarer analysis flow.

v1 trim: the LMech panel product (web UI, templates, fit endpoints) was
removed; the LPD/LMN algorithm core in ``comfy_research.lmech`` stays because
CurveStarer's phase-transition analysis calls this endpoint per curve.
"""
from __future__ import annotations

import asyncio
from typing import List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from comfy_research.lmech.lpd.serve import predict_curve_from_points

api_router = APIRouter(prefix="/api/curve-lpd", tags=["curve-lpd"])


class Point(BaseModel):
    t: float
    loss: float


class LPDPredictRequest(BaseModel):
    points: List[Point] = Field(..., min_length=5)
    K: int = Field(3, ge=1, le=8)
    auto_k: bool = False
    k_max: int = Field(6, ge=2, le=8)
    auto_k_target: float = Field(0.99, ge=0.0, le=1.0)
    simplicity: float = Field(0.25, ge=0.0, le=1.0)


@api_router.post("/predict")
async def curve_lpd_predict(req: LPDPredictRequest):
    try:
        return await asyncio.to_thread(
            predict_curve_from_points,
            [p.model_dump() for p in req.points],
            num_phases=req.K,
            auto_k=req.auto_k,
            k_max=req.k_max,
            auto_k_target=req.auto_k_target,
            simplicity=req.simplicity,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
