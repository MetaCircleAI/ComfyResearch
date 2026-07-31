"""Abort flag for multi-run parameter sweeps (between single-train iterations)."""

from __future__ import annotations

import threading


_lock = threading.Lock()
_abort_requested: set[str] = set()


def register_sweep_session(session_id: str) -> None:
    with _lock:
        _abort_requested.discard(session_id)


def unregister_sweep_session(session_id: str) -> None:
    with _lock:
        _abort_requested.discard(session_id)


def request_sweep_abort(session_id: str) -> bool:
    with _lock:
        _abort_requested.add(session_id)
        return True


def is_sweep_abort_requested(session_id: str) -> bool:
    with _lock:
        return session_id in _abort_requested
