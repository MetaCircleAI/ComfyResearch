"""Content-addressed binary artifacts for every remote execution payload."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


_CHECKPOINT_KEYS = frozenset({"checkpoint_b64", "memoryCheckpoint_b64"})
_REF_KEY = "$comfyArtifact"


def remote_artifact_root() -> Path:
    return Path.cwd() / ".comfyresearch" / "artifacts"


def _checkpoint_path(raw: bytes) -> str:
    return f"checkpoints/{hashlib.sha256(raw).hexdigest()}.pt"


def externalize_checkpoint_artifacts(payload: Any) -> tuple[Any, dict[str, bytes]]:
    """Replace checkpoint base64 strings with small, content-addressed references."""
    blobs: dict[str, bytes] = {}

    def visit(value: Any, key: str | None = None) -> Any:
        if key in _CHECKPOINT_KEYS and isinstance(value, str) and value:
            raw = base64.b64decode(value.encode("ascii"), validate=True)
            path = _checkpoint_path(raw)
            blobs[path] = raw
            return {_REF_KEY: {"kind": "checkpoint", "path": path}}
        if isinstance(value, dict):
            return {str(k): visit(v, str(k)) for k, v in value.items()}
        if isinstance(value, list):
            return [visit(item) for item in value]
        return value

    return visit(payload), blobs


def _artifact_path(root: Path, ref: Any) -> Path:
    if not isinstance(ref, dict) or ref.get("kind") != "checkpoint":
        raise ValueError("Unsupported remote execution artifact reference.")
    rel = str(ref.get("path") or "")
    parts = Path(rel).parts
    if len(parts) != 2 or parts[0] != "checkpoints" or not parts[1].endswith(".pt"):
        raise ValueError("Invalid remote execution artifact path.")
    digest = parts[1][:-3]
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise ValueError("Invalid remote execution artifact digest.")
    return root / rel


def write_artifact_blobs(root: Path, blobs: dict[str, bytes]) -> None:
    for rel, raw in blobs.items():
        path = _artifact_path(root, {"kind": "checkpoint", "path": rel})
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and hashlib.sha256(path.read_bytes()).digest() == hashlib.sha256(raw).digest():
            continue
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as tmp:
            tmp.write(raw)
            tmp_path = Path(tmp.name)
        try:
            os.replace(tmp_path, path)
        finally:
            tmp_path.unlink(missing_ok=True)


def materialize_checkpoint_artifacts(payload: Any, root: Path) -> Any:
    """Restore checkpoint base64 strings before request schema validation."""

    def visit(value: Any) -> Any:
        if isinstance(value, dict) and set(value) == {_REF_KEY}:
            raw = _artifact_path(root, value[_REF_KEY]).read_bytes()
            return base64.standard_b64encode(raw).decode("ascii")
        if isinstance(value, dict):
            return {str(k): visit(v) for k, v in value.items()}
        if isinstance(value, list):
            return [visit(item) for item in value]
        return value

    return visit(payload)


def load_remote_execution_json(raw: str, *, root: Path | None = None) -> Any:
    """Decode and materialize the common stdin contract used by remote CLIs."""
    return materialize_checkpoint_artifacts(json.loads(raw), root or remote_artifact_root())


def cache_checkpoint_artifacts(event: dict[str, Any], root: Path) -> None:
    """Persist checkpoint output on the remote host for later remote nodes."""
    _encoded, blobs = externalize_checkpoint_artifacts(event)
    write_artifact_blobs(root, blobs)
