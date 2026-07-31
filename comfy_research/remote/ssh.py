"""SSH delegation for POST /api/train using stored or env remote config."""

from __future__ import annotations

import hashlib
import io
import json
import os.path
import os
import shlex
import shutil
import subprocess
import tarfile
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from comfy_research.config.remote_train_config import (
    RemoteTrainConfig,
    get_effective_remote_train_config,
)
from comfy_research.remote.session_registry import (
    register_remote_train_session,
    unregister_remote_train_session,
)
from comfy_research.remote.execution_artifacts import externalize_checkpoint_artifacts
from comfy_research.remote.execution_graph import select_execution_graph
from comfy_research.remote.ssh_proc_registry import (
    register_remote_train_ssh_proc,
    unregister_remote_train_ssh_proc,
)
from comfy_research.schemas.train_request import TrainRequest


def _effective_cfg(config: RemoteTrainConfig | None = None) -> RemoteTrainConfig:
    return (config or get_effective_remote_train_config()).normalized()


def _resolve_local_path(path: str) -> str:
    p = (path or "").strip()
    if not p:
        return ""
    return os.path.expandvars(os.path.expanduser(p))


def remote_train_via_ssh_enabled(config: RemoteTrainConfig | None = None) -> bool:
    return _effective_cfg(config).is_active()


def _find_sshpass() -> str | None:
    found = shutil.which("sshpass")
    if found:
        return found
    for cand in ("/opt/homebrew/bin/sshpass", "/usr/local/bin/sshpass"):
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    return None


def _ssh_prefix(config: RemoteTrainConfig) -> tuple[list[str], dict[str, str] | None]:
    cfg = config.normalized()
    cmd: list[str] = ["ssh", "-T"]
    env: dict[str, str] | None = None
    sshpass_bin = _find_sshpass()
    if cfg.password:
        if sshpass_bin is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Password auth requires sshpass on the local host. "
                    "macOS: brew install hudochenkov/sshpass/sshpass"
                ),
            )
        env = os.environ.copy()
        env["SSHPASS"] = cfg.password
        cmd = [sshpass_bin, "-e", *cmd]
        cmd.extend(
            [
                "-o",
                "BatchMode=no",
                "-o",
                "PreferredAuthentications=password",
                "-o",
                "StrictHostKeyChecking=accept-new",
                "-o",
                "ConnectTimeout=30",
            ]
        )
    else:
        cmd.extend(
            [
                "-o",
                "BatchMode=yes",
                "-o",
                "StrictHostKeyChecking=accept-new",
                "-o",
                "ConnectTimeout=30",
            ]
        )
    return cmd, env


def _ssh_shell_command(cfg: RemoteTrainConfig, shell_script: str) -> tuple[list[str], dict[str, str] | None]:
    cmd, env = _ssh_prefix(cfg)
    identity = _resolve_local_path(cfg.identity)
    if identity:
        cmd.extend(["-i", identity])
    if cfg.extra_opts:
        cmd.extend(shlex.split(cfg.extra_opts))
    target = f"{cfg.user}@{cfg.host}"
    cmd.append(target)
    cmd.extend(["bash", "-lc", shlex.quote(shell_script)])
    return cmd, env


def _tar_filter(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
    name = info.name
    base = os.path.basename(name)
    if base in {"__pycache__", ".DS_Store"}:
        return None
    if base.endswith(".pyc") or base.endswith(".pyo"):
        return None
    if "/__pycache__/" in name:
        return None
    return info


def _bundle_member_excluded(rel_posix: str) -> bool:
    base = os.path.basename(rel_posix)
    if base in {"__pycache__", ".DS_Store"}:
        return True
    if base.endswith(".pyc") or base.endswith(".pyo"):
        return True
    if base == "cifar-10-python.tar.gz":
        return True
    return "/__pycache__/" in rel_posix


def _iter_training_bundle_files(*, include_dataset: bool = False) -> list[tuple[str, Path]]:
    root = Path.cwd()
    include_paths = [
        "comfy_research",
        "requirements.txt",
        "data/reproduction_inputs/information_bottleneck/var_u.mat",
    ]
    if include_dataset:
        include_paths.extend(
            rel for rel in ("data/cifar10", "data/mnist") if (root / rel).exists()
        )
    missing: list[str] = []
    files: list[tuple[str, Path]] = []
    for rel in include_paths:
        src = root / rel
        if not src.exists():
            missing.append(rel)
            continue
        if src.is_file():
            files.append((rel, src))
            continue
        for fp in sorted(src.rglob("*")):
            if not fp.is_file():
                continue
            rel_posix = fp.relative_to(root).as_posix()
            if not _bundle_member_excluded(rel_posix):
                files.append((rel_posix, fp))
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Local workspace missing required paths for bootstrap: {', '.join(missing)}",
        )
    return files


_BundleCacheKey = tuple[tuple[str, int, int], ...]
_BUNDLE_DIGEST_CACHE: tuple[str, _BundleCacheKey] | None = None


def _bundle_files_cache_key(files: list[tuple[str, Path]]) -> _BundleCacheKey:
    key: list[tuple[str, int, int]] = []
    for rel_posix, path in files:
        stat = path.stat()
        key.append((rel_posix, stat.st_mtime_ns, stat.st_size))
    return tuple(key)


def _local_training_bundle_digest(*, include_dataset: bool = False) -> str:
    global _BUNDLE_DIGEST_CACHE
    files = _iter_training_bundle_files(include_dataset=include_dataset)
    cache_key = _bundle_files_cache_key(files)
    if _BUNDLE_DIGEST_CACHE is not None and _BUNDLE_DIGEST_CACHE[1] == cache_key:
        return _BUNDLE_DIGEST_CACHE[0]
    digest = hashlib.sha256()
    for rel_posix, path in files:
        digest.update(rel_posix.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    hex_digest = digest.hexdigest()
    _BUNDLE_DIGEST_CACHE = (hex_digest, cache_key)
    return hex_digest


def _run_ssh_subprocess_registered(
    cmd: list[str],
    *,
    trainer_node_id: str | None,
    input_bytes: bytes | None = None,
    env: dict[str, str] | None = None,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[bytes]:
    """``subprocess.run(capture_output=True)`` equivalent that registers the child in
    the remote-train ssh proc registry under ``trainer_node_id``, so POST
    /api/train/control abort can kill blocking bootstrap/validate subprocesses that
    run inside a train request (before the streaming train proc exists).

    ``trainer_node_id=None`` falls back to a plain blocking ``subprocess.run`` — the
    standalone /train/remote/bootstrap and /train/remote/validate endpoints have no
    trainer node in scope, so abort has no id to target them and their behavior is
    intentionally unchanged.
    """
    if trainer_node_id is None:
        return subprocess.run(cmd, input=input_bytes, capture_output=True, timeout=timeout, env=env)
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )
    register_remote_train_ssh_proc(trainer_node_id, proc)
    try:
        out, err = proc.communicate(input=input_bytes, timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        raise
    finally:
        unregister_remote_train_ssh_proc(trainer_node_id, proc)
    if proc.returncode is not None and proc.returncode < 0:
        # 注册子进程被 abort 信号杀死时不得伪装成普通失败
        #(否则 bundle 探测 kill 会被折叠为 False,bootstrap 继续上传)。
        raise HTTPException(status_code=409, detail="remote train aborted during ssh bootstrap step")

    return subprocess.CompletedProcess(cmd, proc.returncode, out, err)


def _sync_remote_execution_artifacts(
    cfg: RemoteTrainConfig,
    blobs: dict[str, bytes],
    *,
    trainer_node_id: str | None = None,
) -> None:
    """Upload only content-addressed artifacts missing from this remote checkout."""
    if not blobs:
        return
    remote_repo = cfg.remote_path.rstrip("/") or cfg.remote_path
    artifact_root = f"{remote_repo}/.comfyresearch/artifacts"
    checks = "\n".join(
        (
            f"[ -f \"$ARTIFACT_ROOT/{rel}\" ] && "
            f"[ \"$(sha256sum \"$ARTIFACT_ROOT/{rel}\" | cut -d ' ' -f 1)\" = {shlex.quote(Path(rel).stem)} ] "
            f"|| echo {shlex.quote(rel)}"
        )
        for rel in sorted(blobs)
    )
    probe = f"""
set -euo pipefail
ARTIFACT_ROOT={shlex.quote(artifact_root)}
{checks}
""".strip()
    cmd, env = _ssh_shell_command(cfg, probe)
    result = _run_ssh_subprocess_registered(
        cmd,
        trainer_node_id=trainer_node_id,
        env=env,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).decode("utf-8", errors="replace").strip()
        raise HTTPException(status_code=400, detail=detail or "Could not inspect remote artifact cache.")
    missing = {
        line.strip()
        for line in result.stdout.decode("utf-8", errors="replace").splitlines()
        if line.strip() in blobs
    }
    if not missing:
        return

    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w") as tar:
        for rel in sorted(missing):
            raw = blobs[rel]
            info = tarfile.TarInfo(rel)
            info.size = len(raw)
            info.mode = 0o600
            tar.addfile(info, io.BytesIO(raw))
    installs = "\n".join(
        (
            f"[ \"$(sha256sum \"$STAGING/{rel}\" | cut -d ' ' -f 1)\" = {shlex.quote(Path(rel).stem)} ]\n"
            f"mv -f \"$STAGING/{rel}\" \"$ARTIFACT_ROOT/{rel}\""
        )
        for rel in sorted(missing)
    )
    upload = f"""
set -euo pipefail
ARTIFACT_ROOT={shlex.quote(artifact_root)}
mkdir -p "$ARTIFACT_ROOT/checkpoints"
STAGING=$(mktemp -d "$ARTIFACT_ROOT/.upload.XXXXXX")
trap 'rm -rf "$STAGING"' EXIT
tar -xf - -C "$STAGING"
{installs}
""".strip()
    cmd, env = _ssh_shell_command(cfg, upload)
    result = _run_ssh_subprocess_registered(
        cmd,
        trainer_node_id=trainer_node_id,
        input_bytes=archive.getvalue(),
        env=env,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).decode("utf-8", errors="replace").strip()
        raise HTTPException(status_code=400, detail=detail or "Could not upload remote execution artifacts.")


def prepare_remote_execution_json(
    payload: dict[str, Any],
    *,
    target_node_id: str,
    config: RemoteTrainConfig | None = None,
    trainer_node_id: str | None = None,
) -> bytes:
    """The single artifact-aware JSON transport for remote execution CLIs."""
    cfg = _effective_cfg(config)
    graph_payload = select_execution_graph(payload, target_node_id)
    encoded, blobs = externalize_checkpoint_artifacts(graph_payload)
    _sync_remote_execution_artifacts(cfg, blobs, trainer_node_id=trainer_node_id)
    return json.dumps(encoded, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _remote_bundle_is_current(
    cfg: RemoteTrainConfig,
    bundle_digest: str,
    *,
    trainer_node_id: str | None = None,
) -> bool:
    if not bundle_digest:
        return False
    remote_repo = cfg.remote_path.rstrip("/") or cfg.remote_path
    script = f"""
set -euo pipefail
REMOTE_REPO={shlex.quote(remote_repo)}
MARKER="$REMOTE_REPO/.comfyresearch/bundle.sha256"
EXPECTED={shlex.quote(bundle_digest)}
if [ -f "$MARKER" ] && [ "$(cat "$MARKER")" = "$EXPECTED" ] && [ -f "$REMOTE_REPO/comfy_research/__init__.py" ] && [ -f "$REMOTE_REPO/requirements.txt" ]; then
  echo BUNDLE_UP_TO_DATE=1
fi
""".strip()
    cmd, env = _ssh_shell_command(cfg, script)
    r = _run_ssh_subprocess_registered(cmd, trainer_node_id=trainer_node_id, env=env)
    if r.returncode != 0:
        return False
    out = r.stdout.decode("utf-8", errors="replace")
    return "BUNDLE_UP_TO_DATE=1" in out


def _build_local_training_bundle(*, include_dataset: bool = False) -> Path:
    tmp = tempfile.NamedTemporaryFile(prefix="cr-remote-bootstrap-", suffix=".tar.gz", delete=False)
    tmp_path = Path(tmp.name)
    tmp.close()
    try:
        with tarfile.open(tmp_path, mode="w:gz") as tar:
            for rel_posix, src in _iter_training_bundle_files(include_dataset=include_dataset):
                tar.add(src, arcname=rel_posix, filter=_tar_filter)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise
    return tmp_path


def _sync_local_training_bundle(
    cfg: RemoteTrainConfig,
    *,
    bundle_digest: str,
    include_dataset: bool = False,
    trainer_node_id: str | None = None,
) -> bool:
    """Upload local training bundle when remote digest differs. Returns True if synced."""
    if bundle_digest and _remote_bundle_is_current(cfg, bundle_digest, trainer_node_id=trainer_node_id):
        return False

    bundle = _build_local_training_bundle(include_dataset=include_dataset)
    remote_repo = cfg.remote_path.rstrip("/") or cfg.remote_path
    script = f"""
set -euo pipefail
REMOTE_REPO={shlex.quote(remote_repo)}
BUNDLE_HASH={shlex.quote(bundle_digest)}
mkdir -p "$REMOTE_REPO/.comfyresearch"
rm -rf "$REMOTE_REPO/comfy_research"
tar -xzf - -C "$REMOTE_REPO"
if [ -n "$BUNDLE_HASH" ]; then
  echo "$BUNDLE_HASH" > "$REMOTE_REPO/.comfyresearch/bundle.sha256"
fi
""".strip()
    cmd, env = _ssh_shell_command(cfg, script)
    try:
        with bundle.open("rb") as fp:
            r = _run_ssh_subprocess_registered(
                cmd,
                trainer_node_id=trainer_node_id,
                input_bytes=fp.read(),
                env=env,
            )
        if r.returncode != 0:
            err = (r.stderr.decode("utf-8", errors="replace") or r.stdout.decode("utf-8", errors="replace")).strip()
            raise HTTPException(status_code=400, detail=err or "Failed to sync local training bundle to remote.")
    finally:
        bundle.unlink(missing_ok=True)
    return True


def _remote_python_module_command(
    module: str,
    module_args: str,
    *,
    config: RemoteTrainConfig | None = None,
) -> tuple[list[str], dict[str, str] | None]:
    cfg = _effective_cfg(config)
    repo = cfg.remote_path
    if not repo:
        raise HTTPException(
            status_code=500,
            detail="Remote path must be set to the repo root on the remote host.",
        )
    py = cfg.python
    inner = (
        f"cd {shlex.quote(repo)} && "
        f"export PYTHONPATH={shlex.quote(repo)}:${{PYTHONPATH:-}} && "
        f"trap 'kill -TERM 0 2>/dev/null || true' TERM INT HUP && "
        f"exec {shlex.quote(py)} -m {module} {module_args}"
    )
    return _ssh_shell_command(cfg, inner)


def _ssh_base_command(
    *,
    remote_shell_inner: str,
    config: RemoteTrainConfig | None = None,
) -> tuple[list[str], dict[str, str] | None]:
    return _remote_python_module_command("comfy_research.remote_train_cli", remote_shell_inner, config=config)


def validate_remote_connection(config: RemoteTrainConfig | None = None) -> None:
    cfg = _effective_cfg(config)
    if not cfg.host:
        raise HTTPException(status_code=400, detail="Remote host is required.")
    if not cfg.remote_path:
        raise HTTPException(status_code=400, detail="Remote path is required.")
    probe = "import sys; print('remote-ready')"
    inner = f"cd {shlex.quote(cfg.remote_path)} && exec {shlex.quote(cfg.python)} -c {shlex.quote(probe)}"
    cmd, env = _ssh_shell_command(cfg, inner)
    r = subprocess.run(
        cmd,
        capture_output=True,
        timeout=None,
        env=env,
    )
    if r.returncode != 0:
        err = (r.stderr.decode("utf-8", errors="replace") or r.stdout.decode("utf-8", errors="replace")).strip()
        raise HTTPException(status_code=400, detail=err or "Remote connection validation failed.")


def _local_requirements_digest() -> str:
    req = Path.cwd() / "requirements.txt"
    if not req.is_file():
        return ""
    return hashlib.sha256(req.read_bytes()).hexdigest()


def _setup_remote_train_environment(
    cfg: RemoteTrainConfig,
    *,
    req_digest: str,
    trainer_node_id: str | None = None,
) -> dict[str, object]:
    """Remote pip (if needed) + import check. Does not sync bundle or download datasets."""
    remote_repo = cfg.remote_path.rstrip("/") or cfg.remote_path
    preferred_python = cfg.python.strip()
    script = f"""
set -euo pipefail
REMOTE_REPO={shlex.quote(remote_repo)}
PREFERRED_PY={shlex.quote(preferred_python)}
REQ_HASH={shlex.quote(req_digest)}
REQ_MARKER="$REMOTE_REPO/.comfyresearch/requirements.sha256"

PY=""
if [ -n "$PREFERRED_PY" ] && command -v "$PREFERRED_PY" >/dev/null 2>&1; then
  PY="$PREFERRED_PY"
fi

if [ -z "$PY" ]; then
  for CANDIDATE in python3 python /root/miniconda3/bin/python /opt/conda/bin/python /usr/bin/python3; do
    if [ -x "$CANDIDATE" ] || command -v "$CANDIDATE" >/dev/null 2>&1; then
      PY="$CANDIDATE"
      break
    fi
  done
fi

if [ -z "$PY" ]; then
  echo "No usable python interpreter found on the remote host." >&2
  exit 3
fi

SKIP_PIP=""
if [ -n "$REQ_HASH" ] && [ -f "$REQ_MARKER" ] && [ "$(cat "$REQ_MARKER")" = "$REQ_HASH" ]; then
  SKIP_PIP=1
fi
if [ -z "$SKIP_PIP" ] && [ -f "$REMOTE_REPO/requirements.txt" ]; then
  "$PY" -m pip install -r "$REMOTE_REPO/requirements.txt"
  if [ -n "$REQ_HASH" ]; then
    mkdir -p "$REMOTE_REPO/.comfyresearch"
    echo "$REQ_HASH" > "$REQ_MARKER"
  fi
fi

cd "$REMOTE_REPO"
"$PY" -c "import comfy_research; print('bootstrap-ready')"
printf 'BOOTSTRAP_PYTHON=%s\\n' "$PY"
""".strip()

    cmd, env = _ssh_shell_command(cfg, script)
    r = _run_ssh_subprocess_registered(cmd, trainer_node_id=trainer_node_id, env=env)
    out = r.stdout.decode("utf-8", errors="replace").strip()
    err = r.stderr.decode("utf-8", errors="replace").strip()
    if r.returncode != 0:
        detail = err or out or "Remote bootstrap failed."
        raise HTTPException(status_code=400, detail=detail)

    selected_python = ""
    for line in out.splitlines():
        if line.startswith("BOOTSTRAP_PYTHON="):
            selected_python = line.split("=", 1)[1].strip()
            break
    return {
        "ok": True,
        "remotePath": remote_repo,
        "python": selected_python or preferred_python,
    }


def iter_remote_bootstrap_events(
    config: RemoteTrainConfig | None = None,
    *,
    trainer_node_id: str | None = None,
) -> Iterator[dict[str, object]]:
    """Yield bootstrap phase events, then a final bootstrap_result dict.

    ``trainer_node_id`` (when in scope, i.e. bootstrap running inside a train
    request) registers the blocking ssh subprocesses in the proc registry so
    abort can kill them mid-bootstrap.
    """
    from comfy_research.engine.runs.train_phase import remote_phase_message

    cfg = _effective_cfg(config)
    if not cfg.host:
        raise HTTPException(status_code=400, detail="Remote host is required.")
    if not cfg.remote_path:
        raise HTTPException(status_code=400, detail="Remote path is required.")

    include_dataset = cfg.upload_dataset
    bundle_digest = _local_training_bundle_digest(include_dataset=include_dataset)
    req_digest = _local_requirements_digest()

    yield {
        "type": "phase",
        "phase": "bootstrap",
        "message": remote_phase_message(
            "syncing training code and local dataset..."
            if include_dataset
            else "syncing training code..."
        ),
    }
    bundle_synced = _sync_local_training_bundle(
        cfg,
        bundle_digest=bundle_digest,
        include_dataset=include_dataset,
        trainer_node_id=trainer_node_id,
    )

    yield {
        "type": "phase",
        "phase": "bootstrap",
        "message": remote_phase_message("verifying remote environment..."),
    }
    setup = _setup_remote_train_environment(cfg, req_digest=req_digest, trainer_node_id=trainer_node_id)
    yield {
        "type": "bootstrap_result",
        **setup,
        "bundleSynced": bundle_synced,
        "message": (
            "Remote bootstrap finished (synced local training code)."
            if bundle_synced
            else "Remote bootstrap finished (training code already up to date)."
        ),
    }


def bootstrap_remote_train_environment(config: RemoteTrainConfig | None = None) -> dict[str, object]:
    """Sync local comfy_research to remote, install deps if needed, verify import.

    Serves the standalone /train/remote/bootstrap endpoint, where no trainer node id
    exists — so these subprocesses are deliberately NOT registered for abort (there is
    no id for /api/train/control to target). The in-train-request bootstrap path goes
    through iter_remote_bootstrap_events(trainer_node_id=...) instead.
    """
    boot: dict[str, object] | None = None
    for ev in iter_remote_bootstrap_events(config):
        if ev.get("type") == "bootstrap_result":
            boot = ev
    if boot is None:
        raise HTTPException(status_code=500, detail="Remote bootstrap produced no result.")
    return boot


def validate_remote_train(body: TrainRequest, *, config: RemoteTrainConfig | None = None) -> None:
    payload = body.model_dump(mode="json")
    raw = prepare_remote_execution_json(
        payload,
        target_node_id=body.trainer_node_id,
        config=config,
        trainer_node_id=body.trainer_node_id,
    )
    cmd, env = _ssh_base_command(remote_shell_inner="--validate-only", config=config)
    r = _run_ssh_subprocess_registered(
        cmd,
        trainer_node_id=body.trainer_node_id,
        input_bytes=raw,
        env=env,
    )
    if r.returncode != 0:
        err = (r.stderr.decode("utf-8", errors="replace") or r.stdout.decode("utf-8", errors="replace")).strip()
        raise HTTPException(status_code=400, detail=err or "Remote validate-only failed.")


def run_remote_train_control(
    session_id: str,
    action: str,
    *,
    config: RemoteTrainConfig | None = None,
) -> bool:
    """Second SSH invocation: write pause/abort into the remote session control file."""
    if action not in ("pause", "abort"):
        return False
    args = f"--session-id {shlex.quote(session_id)} --action {shlex.quote(action)}"
    cmd, env = _remote_python_module_command(
        "comfy_research.remote_train_control_cli",
        args,
        config=config,
    )
    r = subprocess.run(
        cmd,
        capture_output=True,
        timeout=30,
        env=env,
    )
    return r.returncode == 0


def _maybe_register_remote_session_line(line: bytes, trainer_node_id: str) -> None:
    try:
        ev = json.loads(line.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return
    if not isinstance(ev, dict) or ev.get("type") != "remote_session":
        return
    session_id = ev.get("session_id")
    if isinstance(session_id, str) and session_id.strip():
        register_remote_train_session(trainer_node_id, session_id.strip())


def iter_remote_train_stdout_lines(
    body: TrainRequest,
    *,
    config: RemoteTrainConfig | None = None,
) -> Iterator[bytes]:
    import json

    payload: dict[str, Any] = body.model_dump(mode="json")
    trainer_id = body.trainer_node_id
    raw = prepare_remote_execution_json(
        payload,
        target_node_id=trainer_id,
        config=config,
        trainer_node_id=trainer_id,
    )
    cmd, env = _ssh_base_command(remote_shell_inner="", config=config)
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )
    register_remote_train_ssh_proc(trainer_id, proc)
    assert proc.stdin is not None
    proc.stdin.write(raw)
    proc.stdin.close()

    assert proc.stdout is not None
    assert proc.stderr is not None
    last_event_type: str | None = None
    try:
        while True:
            line = proc.stdout.readline()
            if not line:
                break
            try:
                ev = json.loads(line.decode("utf-8"))
                if isinstance(ev, dict):
                    last_event_type = str(ev.get("type") or "")
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass
            _maybe_register_remote_session_line(line, trainer_id)
            yield line
    finally:
        proc.stdout.close()
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=3)
        unregister_remote_train_session(trainer_id)
        unregister_remote_train_ssh_proc(trainer_id, proc)
    rc = proc.wait()
    err = proc.stderr.read().decode("utf-8", errors="replace").strip()
    proc.stderr.close()
    terminal_types = {"complete", "paused", "aborted", "error"}
    if rc != 0 and last_event_type not in terminal_types:
        # Yield a synthetic terminal error line and return cleanly. Raising after the
        # synthetic line (or on empty stdout) would break the already-started NDJSON
        # stream mid-response — the typed error event IS the failure channel here.
        detail = err or f"Remote train process exited with code {rc}."
        yield (json.dumps({"type": "error", "detail": detail}, ensure_ascii=False) + "\n").encode("utf-8")


def run_remote_parametric_path_sampler_json(
    payload: dict[str, object],
    *,
    config: RemoteTrainConfig | None = None,
) -> dict[str, object]:
    """Run parametric path sampler on remote host; return parsed JSON result.

    Cross-machine contract: references the canonical module path
    ``comfy_research.remote.parametric_path_cli`` — no legacy checkouts exist for it.
    """
    cmd, env = _remote_python_module_command(
        "comfy_research.remote.parametric_path_cli",
        "",
        config=config,
    )
    raw = prepare_remote_execution_json(
        payload,
        target_node_id=str(payload.get("sampler_node_id") or ""),
        config=config,
    )
    proc = subprocess.run(
        cmd,
        input=raw,
        capture_output=True,
        timeout=None,
        env=env,
    )
    err = proc.stderr.decode("utf-8", errors="replace").strip()
    out = proc.stdout.decode("utf-8", errors="replace").strip()
    if proc.returncode != 0:
        raise HTTPException(status_code=502, detail=err or f"Remote parametric path exited with code {proc.returncode}.")
    try:
        parsed = json.loads(out)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=502, detail=f"Remote parametric path returned invalid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise HTTPException(status_code=502, detail="Remote parametric path returned non-object JSON.")
    return parsed
