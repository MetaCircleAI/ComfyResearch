"""跨机 exec 的永久兼容入口。与 remote_train_cli 保持一致——
ssh 在远端执行 ``python -m comfy_research.remote_train_control_cli``。"""
from __future__ import annotations

from comfy_research.remote.control_cli import main

__all__ = ["main"]

if __name__ == "__main__":
    main()
