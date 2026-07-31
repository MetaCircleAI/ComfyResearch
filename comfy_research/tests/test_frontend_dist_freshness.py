from __future__ import annotations

import os
from pathlib import Path

import pytest

from comfy_research.main import _assert_frontend_dist_current


def _write(path: Path, text: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_rejects_frontend_dist_older_than_source(tmp_path) -> None:
    frontend = tmp_path / "frontend"
    source = frontend / "src" / "App.tsx"
    index = frontend / "dist" / "index.html"
    _write(source)
    _write(index)
    os.utime(index, (100, 100))
    os.utime(source, (200, 200))

    with pytest.raises(RuntimeError, match=r"frontend/dist is stale.*src/App\.tsx"):
        _assert_frontend_dist_current(frontend)


def test_accepts_current_or_dist_only_frontend(tmp_path) -> None:
    frontend = tmp_path / "frontend"
    source = frontend / "src" / "App.tsx"
    index = frontend / "dist" / "index.html"
    _write(source)
    _write(index)
    os.utime(source, (100, 100))
    os.utime(index, (200, 200))

    _assert_frontend_dist_current(frontend)

    source.unlink()
    _assert_frontend_dist_current(frontend)
