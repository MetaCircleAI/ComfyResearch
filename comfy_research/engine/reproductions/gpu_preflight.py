"""Conservative local-GPU ownership checks for reproduction scripts."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
from typing import Any

import torch


def _endpoint_port(endpoint: str) -> int | None:
    endpoint = endpoint.strip()
    if not endpoint:
        return None
    try:
        return int(endpoint.rsplit(":", 1)[-1])
    except ValueError:
        return None


def _idle_comfyresearch_server_pids(
    netstat_output: str,
    *,
    ports: tuple[int, ...] = (8042,),
) -> set[int]:
    """Find idle ComfyResearch listeners from Windows ``netstat -ano`` text.

    A server that merely retained a CUDA context after an earlier request is
    not an active training owner.  We only call it idle when it listens on a
    configured ComfyResearch port and has no active TCP request on that port.
    Any established/SYN connection keeps the conservative busy result.
    """
    listening: set[int] = set()
    active: set[int] = set()
    allowed_ports = set(int(port) for port in ports)
    for raw_line in netstat_output.splitlines():
        parts = raw_line.split()
        if len(parts) < 5 or parts[0].upper() != "TCP":
            continue
        local_port = _endpoint_port(parts[1])
        if local_port not in allowed_ports:
            continue
        state = parts[3].upper()
        try:
            pid = int(parts[4])
        except ValueError:
            continue
        if state in {"LISTEN", "LISTENING"}:
            listening.add(pid)
        elif state in {"ESTABLISHED", "SYN_SENT", "SYN_RECEIVED"}:
            active.add(pid)
    return listening - active


def _configured_local_ports() -> tuple[int, ...]:
    ports = {8042}
    raw = os.environ.get("COMFYRESEARCH_LOCAL_PORT", "").strip()
    if raw:
        try:
            ports.add(int(raw))
        except ValueError:
            pass
    return tuple(sorted(ports))


def _idle_local_server_pids() -> set[int]:
    if os.name != "nt":
        return set()
    try:
        completed = subprocess.run(
            ["netstat", "-ano", "-p", "tcp"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return set()
    return _idle_comfyresearch_server_pids(
        completed.stdout,
        ports=_configured_local_ports(),
    )


def local_compute_processes() -> list[dict[str, object]]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return []
    rows: list[dict[str, object]] = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        try:
            memory = int(parts[2])
        except ValueError:
            memory = -1
        looks_like_python = "python" in Path(parts[1]).name.lower()
        if pid != os.getpid() and (looks_like_python or memory >= 256):
            rows.append(
                {
                    "pid": pid,
                    "process_name": parts[1],
                    "used_memory_mib": memory if memory >= 0 else None,
                }
            )
    return rows


def resolve_reproduction_device(
    requested: str,
    *,
    allow_busy_gpu: bool,
) -> tuple[torch.device, dict[str, Any]]:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    report: dict[str, Any] = {
        "requested": requested,
        "resolved": str(device),
        "compute_processes": [],
        "ignored_idle_comfyresearch_servers": [],
        "busy_processes": [],
        "allow_busy_gpu": bool(allow_busy_gpu),
    }
    if device.type != "cuda":
        return device, report
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

    processes = local_compute_processes()
    idle_servers = _idle_local_server_pids()
    ignored = [row for row in processes if int(row["pid"]) in idle_servers]
    busy = [row for row in processes if int(row["pid"]) not in idle_servers]
    report.update(
        {
            "compute_processes": processes,
            "ignored_idle_comfyresearch_servers": ignored,
            "busy_processes": busy,
        }
    )
    if busy and not allow_busy_gpu:
        raise RuntimeError(
            "refusing to start because another compute process owns the local GPU: "
            f"{busy}. Re-run only after it finishes (or explicitly pass --allow-busy-gpu)."
        )
    return device, report
