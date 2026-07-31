"""Remote train session control files (AutoDL / SSH proxy, scheme A).

The training process on the remote host polls a per-session control file each step.
A separate SSH invocation writes pause/abort into that file.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

REMOTE_SESSION_DIR = Path("/tmp/comfyresearch/sessions")
ControlAction = Literal["pause", "abort"]


@dataclass(frozen=True)
class RemoteTrainSession:
    session_id: str
    trainer_node_id: str
    pid: int
    control_path: Path
    manifest_path: Path


def _session_paths(session_id: str) -> tuple[Path, Path]:
    base = REMOTE_SESSION_DIR
    return base / f"{session_id}.json", base / f"{session_id}.control.json"


def create_remote_session(trainer_node_id: str) -> RemoteTrainSession:
    REMOTE_SESSION_DIR.mkdir(parents=True, exist_ok=True)
    session_id = str(uuid.uuid4())
    manifest_path, control_path = _session_paths(session_id)
    manifest = {
        "session_id": session_id,
        "trainer_node_id": trainer_node_id,
        "pid": os.getpid(),
        "control_path": str(control_path),
    }
    manifest_path.write_text(json.dumps(manifest, separators=(",", ":")), encoding="utf-8")
    control_path.write_text("{}", encoding="utf-8")
    return RemoteTrainSession(
        session_id=session_id,
        trainer_node_id=trainer_node_id,
        pid=os.getpid(),
        control_path=control_path,
        manifest_path=manifest_path,
    )


def load_remote_session_manifest(session_id: str) -> dict[str, object]:
    manifest_path, _ = _session_paths(session_id)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Remote train session not found: {session_id}")
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Invalid session manifest")
    return data


def resolve_session_control_path(session_id: str, control_path: str) -> Path:
    """Reject control paths outside the session directory or wrong for this session_id."""
    _, expected = _session_paths(session_id.strip())
    resolved = Path(control_path).expanduser().resolve()
    expected_resolved = expected.resolve()
    try:
        resolved.relative_to(REMOTE_SESSION_DIR.resolve())
    except ValueError as e:
        raise ValueError(f"control_path must be under {REMOTE_SESSION_DIR}") from e
    if resolved != expected_resolved:
        raise ValueError(f"control_path does not match session {session_id}")
    return resolved


def write_remote_control_action(control_path: str | Path, action: ControlAction) -> None:
    path = Path(control_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps({"action": action}, separators=(",", ":")), encoding="utf-8")
    os.replace(tmp, path)


def read_and_clear_remote_control_action(control_path: str | Path) -> ControlAction | None:
    path = Path(control_path)
    if not path.is_file():
        return None
    try:
        raw = path.read_text(encoding="utf-8").strip()
        if not raw or raw == "{}":
            return None
        data = json.loads(raw)
        action = data.get("action") if isinstance(data, dict) else None
        if action not in ("pause", "abort"):
            return None
        return action
    except (json.JSONDecodeError, OSError):
        return None
    finally:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


def cleanup_remote_session(session_id: str) -> None:
    manifest_path, control_path = _session_paths(session_id)
    control_path.unlink(missing_ok=True)
    manifest_path.unlink(missing_ok=True)
