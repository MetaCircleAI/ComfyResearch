"""Local registry of in-flight SSH-delegated remote train sessions."""

from __future__ import annotations

import threading
from dataclasses import dataclass


@dataclass(frozen=True)
class LocalRemoteTrainSession:
    trainer_node_id: str
    session_id: str


_lock = threading.Lock()
_by_trainer: dict[str, LocalRemoteTrainSession] = {}


def register_remote_train_session(trainer_node_id: str, session_id: str) -> None:
    with _lock:
        _by_trainer[trainer_node_id] = LocalRemoteTrainSession(
            trainer_node_id=trainer_node_id,
            session_id=session_id,
        )


def unregister_remote_train_session(trainer_node_id: str) -> None:
    with _lock:
        _by_trainer.pop(trainer_node_id, None)


def get_remote_train_session(trainer_node_id: str) -> LocalRemoteTrainSession | None:
    with _lock:
        return _by_trainer.get(trainer_node_id)
