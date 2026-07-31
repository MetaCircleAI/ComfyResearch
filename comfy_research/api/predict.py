from __future__ import annotations

from typing import Any, Literal

import torch
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from comfy_research.generated.node_capabilities import has_capability
from comfy_research.engine.runs.trainer_run import (
    _forward_reg,
    _incoming,
    _node_map,
    load_model_weights_from_checkpoint_b64,
    prepare_trainer_run,
)
from comfy_research.engine.losses.loss_selection import choose_loss_kind
from comfy_research.schemas.graph import Edge, Node, NodeKind

router = APIRouter(prefix="/api", tags=["predict"])


class PredictForwardRequest(BaseModel):
    prediction_node_id: str
    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]
    split: Literal["both", "train", "test"] = "both"
    max_values: int = Field(default=200_000, ge=1024, le=2_000_000)


def _active_checkpoint_b64(data: dict[str, Any]) -> str:
    src = str(data.get("checkpointSource") or "memory").strip().lower()
    ck_file = str(data.get("checkpoint_b64") or "").strip()
    ck_mem = str(data.get("memoryCheckpoint_b64") or "").strip()
    if src == "file" and ck_file:
        return ck_file
    if ck_file:
        return ck_file
    return ck_mem


def _resolve_model_arch_for_checkpoint(
    ckpt_node: Node,
    nmap: dict[str, Node],
    edges: list[Edge],
) -> Node:
    # Preferred path: trainer checkpoint -> model_checkpoint; reuse that trainer's model architecture.
    trainer_src = _incoming(edges, nmap, ckpt_node.id, "model_checkpoint")
    if trainer_src is not None and has_capability(trainer_src.type, "trainer_runner"):
        arch = _incoming(edges, nmap, trainer_src.id, "model")
        if arch is not None and arch.type != NodeKind.model_checkpoint:
            return arch

    raise HTTPException(
        status_code=400,
        detail=(
            "prediction node: model_checkpoint input needs a trainer-linked model architecture. "
            "Connect model_checkpoint from a trainer whose model input is a concrete model node."
        ),
    )


def _tensor_payload(t: torch.Tensor, *, max_values: int) -> dict[str, Any]:
    tcpu = t.detach().to("cpu")
    shape = [int(x) for x in tcpu.shape]
    vals = tcpu.reshape(-1).tolist()
    truncated = False
    if len(vals) > max_values:
        vals = vals[:max_values]
        truncated = True
    return {"shape": shape, "values": vals, "truncated": truncated}


@router.post("/predict/forward")
def post_predict_forward(body: PredictForwardRequest) -> dict[str, Any]:
    try:
        nodes = [Node.model_validate(n) for n in body.nodes]
        edges = [Edge.model_validate(e) for e in body.edges]
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid graph payload: {exc}") from exc

    nmap = _node_map(nodes)
    pred_node = nmap.get(body.prediction_node_id)
    if pred_node is None:
        raise HTTPException(status_code=404, detail="Prediction node not found.")
    if pred_node.type != NodeKind.prediction:
        raise HTTPException(status_code=400, detail=f"Target node must be prediction, got {pred_node.type}.")

    model_in = _incoming(edges, nmap, pred_node.id, "model")
    ds_in = _incoming(edges, nmap, pred_node.id, "dataset")
    if model_in is None:
        raise HTTPException(status_code=400, detail="Prediction node is missing connection: model.")
    if ds_in is None:
        raise HTTPException(status_code=400, detail="Prediction node is missing connection: dataset.")

    checkpoint_b64: str | None = None
    model_arch = model_in
    if model_in.type == NodeKind.model_checkpoint:
        md = model_in.data or {}
        checkpoint_b64 = _active_checkpoint_b64(md)
        if not checkpoint_b64:
            raise HTTPException(
                status_code=400,
                detail="model_checkpoint has no active checkpoint bytes (load from file or memory first).",
            )
        model_arch = _resolve_model_arch_for_checkpoint(model_in, nmap, edges)

    tmp_trainer = Node(
        id=f"{pred_node.id}::__prediction_tmp_trainer",
        type=NodeKind.trainer,
        data={"trainingSteps": 1, "logFrequency": 1, "batchSize": -1},
    )
    tmp_optimizer = Node(
        id=f"{pred_node.id}::__prediction_tmp_optimizer",
        type=NodeKind.adam_optimizer,
        data={"learningRate": 0.001},
    )
    tmp_loss = Node(
        id=f"{pred_node.id}::__prediction_tmp_loss",
        type=choose_loss_kind(model_arch.type, ds_in.type),
        data={},
    )

    run_nodes = [*nodes, tmp_trainer, tmp_optimizer, tmp_loss]
    run_edges = [
        *edges,
        Edge(
            id=f"{pred_node.id}::__prediction_tmp_e_model",
            source=model_arch.id,
            sourceHandle="model",
            target=tmp_trainer.id,
            targetHandle="model",
        ),
        Edge(
            id=f"{pred_node.id}::__prediction_tmp_e_dataset",
            source=ds_in.id,
            sourceHandle="dataset",
            target=tmp_trainer.id,
            targetHandle="dataset",
        ),
        Edge(
            id=f"{pred_node.id}::__prediction_tmp_e_optimizer",
            source=tmp_optimizer.id,
            sourceHandle="optimizer",
            target=tmp_trainer.id,
            targetHandle="optimizer",
        ),
        Edge(
            id=f"{pred_node.id}::__prediction_tmp_e_loss",
            source=tmp_loss.id,
            sourceHandle="loss",
            target=tmp_trainer.id,
            targetHandle="loss",
        ),
    ]

    ctx = prepare_trainer_run(run_nodes, run_edges, tmp_trainer.id)
    ctx.model.eval()
    if checkpoint_b64:
        load_model_weights_from_checkpoint_b64(ctx.model, checkpoint_b64)

    with torch.no_grad():
        train_pred = _forward_reg(ctx.model, ctx.x_t) if body.split in ("both", "train") else None
        test_pred = _forward_reg(ctx.model, ctx.x_test_t) if body.split in ("both", "test") and ctx.x_test_t is not None else None

    out: dict[str, Any] = {
        "trainerTask": ctx.trainer_task,
        "trainPrediction": _tensor_payload(train_pred, max_values=body.max_values) if train_pred is not None else None,
    }
    if test_pred is not None:
        out["testPrediction"] = _tensor_payload(test_pred, max_values=body.max_values)
    else:
        out["testPrediction"] = None
    return out
