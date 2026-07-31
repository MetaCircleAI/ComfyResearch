"""Read-only source for ``comfy_research`` Python modules (Code notebook import navigation)."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

router = APIRouter(prefix="/api/library", tags=["library"])

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_MAX_BYTES = 2_000_000


def _is_under_comfy_research_package(path: Path) -> bool:
    """Allow only files under ``<repo>/comfy_research/`` (the installed package tree)."""
    try:
        resolved = path.resolve()
        pkg_root = (_REPO_ROOT / "comfy_research").resolve()
    except OSError:
        return False
    try:
        resolved.relative_to(pkg_root)
    except ValueError:
        return False
    return True


@router.get("/source")
def get_library_source(module: str = Query(..., min_length=1, max_length=256)) -> dict[str, str]:
    """Return UTF-8 file source for an importable ``comfy_research.*`` submodule (``.py`` only)."""
    m = module.strip()
    if not m.startswith("comfy_research.") or ".." in m:
        raise HTTPException(status_code=400, detail="Only dotted modules under comfy_research are allowed.")
    parts = m.split(".")
    seg = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")
    if len(parts) < 2 or not all(seg.match(p) for p in parts):
        raise HTTPException(status_code=400, detail="Invalid module path.")

    spec = importlib.util.find_spec(m)
    if spec is None or not spec.origin:
        raise HTTPException(status_code=404, detail=f"Module not found: {m}")
    origin = Path(spec.origin)
    if origin.suffix.lower() != ".py":
        raise HTTPException(status_code=400, detail="Only .py modules are supported.")
    if not _is_under_comfy_research_package(origin):
        raise HTTPException(status_code=403, detail="Resolved path is outside comfy_research/.")

    try:
        size = origin.stat().st_size
    except OSError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    if size > _MAX_BYTES:
        raise HTTPException(status_code=413, detail="File too large to display.")

    try:
        text = origin.read_text(encoding="utf-8")
    except OSError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    try:
        rel = str(origin.resolve().relative_to(_REPO_ROOT.resolve()))
    except ValueError:
        rel = str(origin)

    return {"module": m, "path": rel, "source": text}
