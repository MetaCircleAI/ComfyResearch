"""stdin JSON (ParametricPathSamplerRequest) -> stdout JSON result. Used by AutoDL SSH delegation.

跨机合同：无旧 checkout 存量，ssh 命令字符串直接引用本 canonical
路径(comfy_research.remote.parametric_path_cli),不设旧路径 shim。
"""

from __future__ import annotations

import json
import sys

from fastapi import HTTPException

from comfy_research.engine.analysis.parametric_path_sampler import run_parametric_path_sampler
from comfy_research.remote.execution_artifacts import load_remote_execution_json
from comfy_research.schemas.graph import Edge, Node


def main() -> None:
    raw = sys.stdin.read()
    if not raw.strip():
        print("empty stdin", file=sys.stderr)
        sys.exit(2)
    try:
        payload = load_remote_execution_json(raw)
        sampler_node_id = str(payload["sampler_node_id"])
        nodes = [Node.model_validate(n) for n in payload["nodes"]]
        edges = [Edge.model_validate(e) for e in payload["edges"]]
    except Exception as e:
        print(f"invalid parametric path request JSON: {e}", file=sys.stderr)
        sys.exit(2)

    try:
        result = run_parametric_path_sampler(nodes, edges, sampler_node_id)
    except HTTPException as e:
        detail = e.detail
        msg = json.dumps(detail, ensure_ascii=False) if isinstance(detail, dict) else str(detail)
        print(msg, file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"{type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)

    sys.stdout.write(json.dumps(result, separators=(",", ":"), ensure_ascii=False))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
