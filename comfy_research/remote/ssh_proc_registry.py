"""Track local SSH subprocesses for remote train so abort can kill prepare/download."""

from __future__ import annotations

import subprocess
import threading

_lock = threading.Lock()
_by_trainer: dict[str, subprocess.Popen[bytes]] = {}


def register_remote_train_ssh_proc(trainer_node_id: str, proc: subprocess.Popen[bytes]) -> None:
    with _lock:
        prev = _by_trainer.get(trainer_node_id)
        if prev is not None and prev.poll() is None:
            _terminate_proc(prev)
        _by_trainer[trainer_node_id] = proc


def unregister_remote_train_ssh_proc(trainer_node_id: str, proc: subprocess.Popen[bytes] | None) -> None:
    with _lock:
        current = _by_trainer.get(trainer_node_id)
        if proc is None or current is proc:
            _by_trainer.pop(trainer_node_id, None)


def kill_remote_train_ssh_proc(trainer_node_id: str) -> bool:
    with _lock:
        proc = _by_trainer.pop(trainer_node_id, None)
    if proc is None:
        return False
    return _terminate_proc(proc)


def _terminate_proc(proc: subprocess.Popen[bytes]) -> bool:
    if proc.poll() is not None:
        return True
    proc.terminate()
    try:
        proc.wait(timeout=8)
        return True
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            return False
        return True
