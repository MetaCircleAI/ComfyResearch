"""Per-trainer control flags for cooperative pause/abort during streaming training."""

from __future__ import annotations

import threading
from dataclasses import dataclass

from comfy_research.remote.session_ipc import read_and_clear_remote_control_action


class TrainingAborted(Exception):
    """Training loop should yield ``aborted`` and exit."""


@dataclass
class TrainControl:
    pause_requested: bool = False
    abort_requested: bool = False


_lock = threading.Lock()
_active: dict[str, TrainControl] = {}
_control_files: dict[str, str] = {}


def register_trainer(trainer_id: str) -> TrainControl:
    with _lock:
        c = TrainControl()
        _active[trainer_id] = c
        return c


def unregister_trainer(trainer_id: str) -> None:
    with _lock:
        _active.pop(trainer_id, None)
        _control_files.pop(trainer_id, None)


def bind_control_file(trainer_id: str, path: str) -> None:
    with _lock:
        _control_files[trainer_id] = path


def unbind_control_file(trainer_id: str) -> None:
    with _lock:
        _control_files.pop(trainer_id, None)


def _poll_control_file(trainer_id: str, c: TrainControl) -> None:
    path = _control_files.get(trainer_id)
    if not path:
        return
    action = read_and_clear_remote_control_action(path)
    if action == "abort":
        c.abort_requested = True
    elif action == "pause":
        c.pause_requested = True


def get_control(trainer_id: str) -> TrainControl | None:
    with _lock:
        c = _active.get(trainer_id)
        if c is not None:
            _poll_control_file(trainer_id, c)
        return c


def request_pause(trainer_id: str) -> bool:
    with _lock:
        c = _active.get(trainer_id)
        if c is None:
            return False
        c.pause_requested = True
        return True


def request_abort(trainer_id: str) -> bool:
    with _lock:
        c = _active.get(trainer_id)
        if c is None:
            return False
        c.abort_requested = True
        return True
