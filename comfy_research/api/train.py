from __future__ import annotations

import json
from dataclasses import asdict

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from comfy_research.engine.runs.sweep_control import request_sweep_abort
from comfy_research.engine.runs.train_coordinate_descent import (
    TrainCoordinateDescentRequest,
    iter_coordinate_descent_events,
    validate_coordinate_descent_request,
)
from comfy_research.engine.runs.train_control import request_abort, request_pause
from comfy_research.engine.runs.train_sweep import TrainSweepRequest, iter_sweep_events, validate_sweep_request
from comfy_research.engine.runs.ai4science_alias import remap_ai4science_node_types
from comfy_research.engine.crl.crl_run import iter_crl_events_from_context, prepare_crl_run
from comfy_research.engine.runs.cuda_devices import list_local_cuda_devices
from comfy_research.engine.runs.trainer_run import iter_trainer_events_from_context, prepare_trainer_run
from comfy_research.schemas.graph import NodeKind
from comfy_research.config.remote_train_config import (
    RemoteTrainConfig,
    get_effective_remote_train_config,
    load_saved_remote_train_config,
    save_remote_train_config,
    set_last_validation_result,
)
from comfy_research.remote.ssh import (
    bootstrap_remote_train_environment,
    iter_remote_bootstrap_events,
    iter_remote_train_stdout_lines,
    remote_train_via_ssh_enabled,
    run_remote_train_control,
    validate_remote_connection,
    validate_remote_train,
)
from comfy_research.engine.runs.train_phase import remote_phase_message
from comfy_research.remote.ssh_proc_registry import kill_remote_train_ssh_proc
from comfy_research.remote.session_registry import get_remote_train_session
from comfy_research.schemas.train_request import TrainRequest, sanitize_train_ndjson_value

router = APIRouter(prefix="/api", tags=["train"])


class TrainControlRequest(BaseModel):
    trainer_node_id: str
    action: str


class TrainSweepControlRequest(BaseModel):
    sweep_session_id: str = Field(min_length=1)
    action: str = Field(min_length=1)


class TrainCoordinateDescentControlRequest(BaseModel):
    session_id: str = Field(min_length=1)
    action: str = Field(min_length=1)


class RemoteTrainConfigBody(BaseModel):
    host: str = ""
    user: str = "ubuntu"
    remote_path: str = ""
    python: str = "/root/miniconda3/bin/python3"
    identity: str = ""
    password: str = ""
    extra_opts: str = ""
    enabled: bool = False
    upload_dataset: bool = False


class RemoteTrainConfigResponse(BaseModel):
    host: str
    user: str
    remote_path: str
    python: str
    identity: str
    password: str
    extra_opts: str
    enabled: bool
    upload_dataset: bool
    source: str
    active: bool
    last_validation_ok: bool | None = None
    last_validation_error: str = ""


class RemoteTrainValidationRequest(BaseModel):
    config: RemoteTrainConfigBody | None = None


class RemoteTrainBootstrapRequest(BaseModel):
    config: RemoteTrainConfigBody | None = None


def _cfg_from_body(body: RemoteTrainConfigBody) -> RemoteTrainConfig:
    return RemoteTrainConfig(
        host=body.host,
        user=body.user,
        remote_path=body.remote_path,
        python=body.python,
        identity=body.identity,
        password=body.password,
        extra_opts=body.extra_opts,
        enabled=body.enabled,
        upload_dataset=body.upload_dataset,
        source="stored",
    ).normalized()


def _cfg_response(cfg: RemoteTrainConfig) -> RemoteTrainConfigResponse:
    c = cfg.normalized()
    return RemoteTrainConfigResponse(
        host=c.host,
        user=c.user,
        remote_path=c.remote_path,
        python=c.python,
        identity=c.identity,
        password=c.password,
        extra_opts=c.extra_opts,
        enabled=c.enabled,
        upload_dataset=c.upload_dataset,
        source=c.source,
        active=c.is_active(),
        last_validation_ok=c.last_validation_ok,
        last_validation_error=c.last_validation_error,
    )


def _ndjson_encode(event: dict) -> bytes:
    safe = sanitize_train_ndjson_value(event)
    return (json.dumps(safe, separators=(",", ":")) + "\n").encode("utf-8")


def _trainer_prefers_remote_gpu(body: TrainRequest) -> bool:
    n_by_id = {n.id: n for n in body.nodes}
    tnode = n_by_id.get(body.trainer_node_id)
    if tnode is None:
        return False
    data = getattr(tnode, "data", None)
    if not isinstance(data, dict):
        return False
    spec = str(data.get("computeDevice", "")).strip().lower()
    if not spec:
        return False
    is_cuda = spec == "cuda" or spec.startswith("cuda:")
    if not is_cuda:
        return False
    remote_gpu = data.get("remoteGpu")
    return remote_gpu is True


@router.post("/train/control")
def post_train_control(body: TrainControlRequest) -> dict[str, bool]:
    """Signal pause or abort for an in-flight training stream (same trainer_node_id)."""
    trainer_id = body.trainer_node_id
    if body.action == "pause":
        ok = request_pause(trainer_id)
        if not ok:
            remote = get_remote_train_session(trainer_id)
            if remote is not None:
                ok = run_remote_train_control(
                    remote.session_id,
                    "pause",
                    config=get_effective_remote_train_config(),
                )
        if not ok:
            raise HTTPException(status_code=404, detail="No active training for this trainer.")
        return {"ok": True}
    if body.action == "abort":
        remote_handled = False
        remote = get_remote_train_session(trainer_id)
        if remote is not None:
            remote_handled = run_remote_train_control(
                remote.session_id,
                "abort",
                config=get_effective_remote_train_config(),
            )
        ok = request_abort(trainer_id)
        killed_ssh = kill_remote_train_ssh_proc(trainer_id)

        if not ok and not remote_handled and not killed_ssh:
            raise HTTPException(status_code=404, detail="No active training for this trainer.")
        return {"ok": True}
    raise HTTPException(status_code=400, detail='action must be "pause" or "abort"')


@router.post("/train/sweep/control")
def post_train_sweep_control(body: TrainSweepControlRequest) -> dict[str, bool]:
    """Abort a parameter sweep between single-train iterations (``sweep_session_id`` from sweep start)."""
    if body.action != "abort":
        raise HTTPException(status_code=400, detail='action must be "abort"')
    request_sweep_abort(body.sweep_session_id.strip())
    return {"ok": True}


@router.post("/train/coordinate-descent/control")
def post_train_coordinate_descent_control(body: TrainCoordinateDescentControlRequest) -> dict[str, bool]:
    """Abort an in-flight coordinate-descent tuning stream by session id."""
    if body.action != "abort":
        raise HTTPException(status_code=400, detail='action must be "abort"')
    request_sweep_abort(body.session_id.strip())
    return {"ok": True}


@router.post("/train/sweep")
def post_train_sweep(body: TrainSweepRequest) -> StreamingResponse:
    """Stream NDJSON: sweep_started, sweep_progress, sweep_row, sweep_complete | sweep_aborted."""
    try:
        validate_sweep_request(body)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    def generate():
        for ev in iter_sweep_events(body):
            yield _ndjson_encode(ev)

    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-store"},
    )


@router.post("/train/coordinate-descent")
def post_train_coordinate_descent(body: TrainCoordinateDescentRequest) -> StreamingResponse:
    """Stream NDJSON for coordinate-descent curve matching."""
    try:
        validate_coordinate_descent_request(body)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    def generate():
        for ev in iter_coordinate_descent_events(body):
            yield _ndjson_encode(ev)

    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-store"},
    )


@router.get("/train/cuda-devices")
def get_train_cuda_devices() -> dict[str, object]:
    devices = list_local_cuda_devices()
    return {
        "ok": True,
        "cudaAvailable": bool(devices),
        "devices": devices,
    }


@router.get("/train/remote/config")
def get_train_remote_config() -> RemoteTrainConfigResponse:
    cfg = load_saved_remote_train_config()
    return _cfg_response(cfg)


@router.post("/train/remote/config")
def post_train_remote_config(body: RemoteTrainConfigBody) -> RemoteTrainConfigResponse:
    candidate = _cfg_from_body(body)
    cleared = RemoteTrainConfig(
        **{
            **asdict(candidate),
            "last_validation_ok": None,
            "last_validation_error": "",
        }
    ).normalized()
    saved = save_remote_train_config(cleared)
    return _cfg_response(saved)


@router.post("/train/remote/validate")
def post_train_remote_validate(body: RemoteTrainValidationRequest) -> dict[str, object]:
    candidate = _cfg_from_body(body.config) if body.config is not None else get_effective_remote_train_config()
    try:
        validate_remote_connection(candidate)
        if body.config is None:
            set_last_validation_result(True, "")
        else:
            save_remote_train_config(candidate)
            set_last_validation_result(True, "")
    except HTTPException as exc:
        if body.config is None:
            set_last_validation_result(False, str(exc.detail))
        else:
            save_remote_train_config(candidate)
            set_last_validation_result(False, str(exc.detail))
        raise
    return {"ok": True}


@router.post("/train/remote/bootstrap")
def post_train_remote_bootstrap(body: RemoteTrainBootstrapRequest) -> dict[str, object]:
    candidate = _cfg_from_body(body.config) if body.config is not None else get_effective_remote_train_config()
    try:
        result = bootstrap_remote_train_environment(candidate)
        selected_python = str(result.get("python", "")).strip()
        cfg_to_save = RemoteTrainConfig(
            **{
                **asdict(candidate),
                "python": selected_python or candidate.python,
                "source": "stored",
            }
        ).normalized()
        save_remote_train_config(cfg_to_save)
        set_last_validation_result(True, "")
        return result
    except HTTPException as exc:
        if body.config is None:
            set_last_validation_result(False, str(exc.detail))
        else:
            save_remote_train_config(candidate)
            set_last_validation_result(False, str(exc.detail))
        raise


@router.get("/train/remote/status")
def get_train_remote_status() -> dict[str, object]:
    cfg = get_effective_remote_train_config()
    return {
        "ok": True,
        "mode": "remote" if cfg.is_active() else "local",
        "remoteHost": cfg.host,
        "remoteUser": cfg.user,
        "remotePath": cfg.remote_path,
        "source": cfg.source,
        "lastValidationOk": cfg.last_validation_ok,
        "lastValidationError": cfg.last_validation_error,
    }


def _remote_train_not_ready_detail() -> str:
    stored = load_saved_remote_train_config().normalized()
    if not stored.host.strip():
        return (
            "Trainer is set to AutoDL GPU but SSH host is missing. "
            "Fill SSH command + remote path in the Trainer AutoDL panel."
        )
    if not stored.remote_path.strip():
        return (
            "Trainer is set to AutoDL GPU but remote path is empty. "
            "Set Remote path (e.g. /root/ComfyResearch) in the Trainer AutoDL panel."
        )
    if not stored.enabled:
        return (
            "Trainer is set to AutoDL GPU but remote training is disabled. "
            "Re-select AutoDL GPU or enable remote in the Trainer panel."
        )
    return (
        "Trainer is set to AutoDL GPU (remoteGpu=true) but remote SSH training is not configured. "
        "Fill SSH command + remote path in the Trainer AutoDL panel (or Settings), or switch compute device to CPU / MPS."
    )


@router.post("/train")
def post_train(body: TrainRequest) -> StreamingResponse:
    """Streams NDJSON: progress, optional paused or aborted, then complete (or paused/aborted only)."""
    remote_cfg = get_effective_remote_train_config()
    prefers_remote = _trainer_prefers_remote_gpu(body)
    remote_enabled = remote_train_via_ssh_enabled(remote_cfg)
    if prefers_remote and not remote_enabled:
        raise HTTPException(status_code=400, detail=_remote_train_not_ready_detail())
    if remote_enabled and prefers_remote:
        cfg_holder: dict[str, RemoteTrainConfig] = {"cfg": remote_cfg}

        def generate_remote():
            try:
                boot: dict[str, object] | None = None
                # trainer_node_id registers the blocking bootstrap ssh subprocesses in the
                # Register the process so /api/train/control can terminate it; abort must
                # cover bootstrap/validate, not just the streaming train proc).
                for ev in iter_remote_bootstrap_events(cfg_holder["cfg"], trainer_node_id=body.trainer_node_id):
                    if ev.get("type") == "bootstrap_result":
                        boot = ev
                        break
                    yield _ndjson_encode(ev)
                if boot is None:
                    raise HTTPException(status_code=500, detail="Remote bootstrap produced no result.")
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
                yield _ndjson_encode(
                    {
                        "type": "phase",
                        "phase": "validate",
                        "message": remote_phase_message("validating graph..."),
                    }
                )
                validate_remote_train(body, config=cfg_holder["cfg"])
                set_last_validation_result(True, "")
            except HTTPException as exc:
                set_last_validation_result(False, str(exc.detail))
                yield _ndjson_encode({"type": "error", "detail": str(exc.detail)})
                return
            except Exception as exc:
                set_last_validation_result(False, str(exc))
                yield _ndjson_encode({"type": "error", "detail": str(exc)})
                return
            yield from iter_remote_train_stdout_lines(body, config=cfg_holder["cfg"])

        return StreamingResponse(
            generate_remote(),
            media_type="application/x-ndjson",
            headers={"Cache-Control": "no-store"},
        )

    # Validate and build model before StreamingResponse so HTTPException becomes a normal 400.
    # (Otherwise Starlette has already sent 200 headers and the client sees a broken stream / "network error".)
    n_by_id = {n.id: n for n in body.nodes}
    tnode = n_by_id.get(body.trainer_node_id)
    is_crl = tnode is not None and str(tnode.type) == str(NodeKind.crl_trainer)
    if is_crl:
        crl_ctx = prepare_crl_run(
            body.nodes,
            body.edges,
            body.trainer_node_id,
            resume=body.resume,
        )

        def generate_crl():
            for event in iter_crl_events_from_context(crl_ctx):
                yield _ndjson_encode(event)

        return StreamingResponse(
            generate_crl(),
            media_type="application/x-ndjson",
            headers={"Cache-Control": "no-store"},
        )

    mapped_nodes = remap_ai4science_node_types(body.nodes)
    ctx = prepare_trainer_run(
        mapped_nodes,
        body.edges,
        body.trainer_node_id,
        resume=body.resume,
        hessian_oversized_policy=body.hessian_oversized_policy,
    )

    def generate():
        for event in iter_trainer_events_from_context(ctx):
            yield _ndjson_encode(event)

    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-store"},
    )
