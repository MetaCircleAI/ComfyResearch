"""跨机 exec 的永久兼容入口。

remote_train_ssh(现 comfy_research/remote/ssh.py)在**远端机器**上执行
``python -m comfy_research.remote_train_cli``;远端 checkout 版本可能滞后,
本模块路径是跨机协议,不是技术债。ssh 命令字符串有意永远指向本旧路径:
远端旧版本跑旧真实模块,远端新版本经本 shim 跑新模块,两边都通。
repo 内部代码禁止 import 此路径；AST 守卫会验证该约束。
"""
from __future__ import annotations

from comfy_research.remote.train_cli import main

__all__ = ["main"]

if __name__ == "__main__":
    main()
