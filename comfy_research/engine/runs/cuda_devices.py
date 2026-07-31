from __future__ import annotations

from typing import Any

import torch


def list_local_cuda_devices() -> list[dict[str, Any]]:
    """Enumerate CUDA devices on the machine running the API server."""
    if not torch.cuda.is_available():
        return []
    out: list[dict[str, Any]] = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        name = str(getattr(props, "name", "") or f"cuda:{i}")
        total_mem = int(getattr(props, "total_memory", 0) or 0)
        out.append(
            {
                "index": i,
                "name": name,
                "totalMemoryMb": total_mem // (1024 * 1024),
            }
        )
    return out
