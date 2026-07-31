from __future__ import annotations

from comfy_research.engine.reproductions.gpu_preflight import (
    _idle_comfyresearch_server_pids,
)


def test_idle_comfyresearch_listener_is_ignored() -> None:
    output = """
      TCP    0.0.0.0:8042           0.0.0.0:0              LISTENING       54936
      TCP    127.0.0.1:56980        127.0.0.1:56979        ESTABLISHED     54936
    """
    assert _idle_comfyresearch_server_pids(output) == {54936}


def test_active_request_keeps_server_busy() -> None:
    output = """
      TCP    0.0.0.0:8042           0.0.0.0:0              LISTENING       54936
      TCP    127.0.0.1:8042         127.0.0.1:57001        ESTABLISHED     54936
    """
    assert _idle_comfyresearch_server_pids(output) == set()


def test_other_ports_are_not_misclassified_as_comfyresearch() -> None:
    output = """
      TCP    0.0.0.0:8888           0.0.0.0:0              LISTENING       12345
    """
    assert _idle_comfyresearch_server_pids(output) == set()
