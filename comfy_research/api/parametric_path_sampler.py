from __future__ import annotations

from dataclasses import asdict
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from comfy_research.api.train import _remote_train_not_ready_detail
from comfy_research.config.remote_train_config import (
    RemoteTrainConfig,
    get_effective_remote_train_config,
    save_remote_train_config,
)
from comfy_research.engine.analysis.parametric_path_sampler import run_parametric_path_sampler
from comfy_research.remote.ssh import (
    iter_remote_bootstrap_events,
    remote_train_via_ssh_enabled,
    run_remote_parametric_path_sampler_json,
)
from comfy_research.schemas.graph import Edge, Node

router = APIRouter(prefix="/api", tags=["analysis"])


class ParametricPathSamplerRequest(BaseModel):
    sampler_node_id: str
    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]


def _sampler_prefers_remote_gpu(sampler: Node) -> bool:
    data = sampler.data if isinstance(sampler.data, dict) else {}
    spec = str(data.get("computeDevice", "")).strip().lower()
    if not spec:
        return False
    is_cuda = spec == "cuda" or spec.startswith("cuda:")
    if not is_cuda:
        return False
    return data.get("remoteGpu") is True


@router.post("/parametric_path_sampler")
def post_parametric_path_sampler(body: ParametricPathSamplerRequest) -> dict[str, Any]:
    nodes = [Node.model_validate(n) for n in body.nodes]
    edges = [Edge.model_validate(e) for e in body.edges]
    nmap = {n.id: n for n in nodes}
    sampler = nmap.get(body.sampler_node_id)
    if sampler is None:
        raise HTTPException(status_code=404, detail="Parametric path sampler node not found.")

    prefers_remote = _sampler_prefers_remote_gpu(sampler)
    remote_cfg = get_effective_remote_train_config()
    if prefers_remote:
        if not remote_train_via_ssh_enabled(remote_cfg):
            raise HTTPException(status_code=400, detail=_remote_train_not_ready_detail())
        cfg_holder: dict[str, RemoteTrainConfig] = {"cfg": remote_cfg}
        boot: dict[str, object] | None = None
        for ev in iter_remote_bootstrap_events(cfg_holder["cfg"]):
            if ev.get("type") == "bootstrap_result":
                boot = ev
                break
        if boot is None:
            raise HTTPException(status_code=500, detail="Remote bootstrap produced no result.")
        # Mirror /api/train: when bootstrap discovers a different python interpreter,
        # persist it and use it for the sampler invocation (otherwise the remote run
        # would use the stale configured interpreter and can fail to import torch).
        selected_python = str(boot.get("python", "")).strip()
        if (
            selected_python
            and cfg_holder["cfg"].source == "stored"
            and selected_python != cfg_holder["cfg"].python
        ):
            cfg_holder["cfg"] = save_remote_train_config(
                RemoteTrainConfig(
                    **{
                        **asdict(cfg_holder["cfg"]),
                        "python": selected_python,
                        "source": "stored",
                    }
                ).normalized()
            )
        try:
            return run_remote_parametric_path_sampler_json(
                body.model_dump(mode="json"),
                config=cfg_holder["cfg"],
            )
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc

    try:
        return run_parametric_path_sampler(nodes, edges, body.sampler_node_id)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc
