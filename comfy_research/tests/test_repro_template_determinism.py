"""The repro templates are tool-generated, never hand-edited.

Two guards, in the spirit of the golden ``--check`` patterns:
1. Determinism — two runs of ``seed_repro_graph_templates.main()`` produce
   byte-identical files modulo the ``savedAt`` timestamp (``time.time()``).
2. Drift — the committed ``data/graph_library/templates/{slug}.json`` files match
   a fresh regeneration modulo ``savedAt``; any hand edit fails here and the fix
   is to change the tool and rerun it, not to edit the JSON.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from comfy_research.tools import seed_repro_graph_templates as seed_tool

_REPO = Path(__file__).resolve().parents[2]
_TEMPLATES = _REPO / "data" / "graph_library" / "templates"

SLUGS = (
    "repro-keskar-fig23-sb-lb",
    "repro-jastrzbski-fig1-vgg11",
    "repro-thilak-fig1-slingshot",
)

_SAVED_AT_RE = re.compile(r'^(\s*"savedAt": )\d+(?:\.\d+)?(,?)$', flags=re.MULTILINE)


def _mask_saved_at(text: str) -> str:
    masked, n = _SAVED_AT_RE.subn(r"\g<1>0\g<2>", text)
    assert n == 1, f"expected exactly one savedAt line, found {n}"
    return masked


def _generate_into(out_dir: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    monkeypatch.setattr(seed_tool, "_OUT", out_dir)
    seed_tool.main()
    files = {p.name: p.read_text(encoding="utf-8") for p in out_dir.glob("*.json")}
    assert set(files) == {f"{slug}.json" for slug in SLUGS}
    return files


def test_seed_tool_is_deterministic_modulo_saved_at(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = _generate_into(tmp_path / "run1", monkeypatch)
    second = _generate_into(tmp_path / "run2", monkeypatch)
    for name in first:
        assert _mask_saved_at(first[name]) == _mask_saved_at(second[name]), name


@pytest.mark.parametrize("slug", SLUGS)
def test_committed_template_matches_regenerated(
    slug: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    committed = _TEMPLATES / f"{slug}.json"
    assert committed.is_file(), f"missing committed template {committed}"
    fresh = _generate_into(tmp_path, monkeypatch)[f"{slug}.json"]
    assert _mask_saved_at(committed.read_text(encoding="utf-8")) == _mask_saved_at(fresh), (
        f"{committed} drifted from the seed tool output; regenerate it with "
        "`python comfy_research/tools/seed_repro_graph_templates.py` (never hand-edit)."
    )


def test_cli_check_semantics(tmp_path) -> None:
    """--check 真语义——干净时零退出零写入;漂移时非零退出。"""
    import shutil
    import subprocess
    import sys
    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[2]
    tool = ROOT / "comfy_research" / "tools" / "seed_repro_graph_templates.py"
    tpl_dir = ROOT / "data" / "graph_library" / "templates"
    tpl = tpl_dir / "repro-jastrzbski-fig1-vgg11.json"
    before = tpl.read_bytes()
    r = subprocess.run([sys.executable, str(tool), "--check"], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout + r.stderr
    assert tpl.read_bytes() == before  # check 不写工作树
    backup = tmp_path / "tpl.json"
    shutil.copy(tpl, backup)
    try:
        tpl.write_text(tpl.read_text().replace('"tier": "medium"', '"tier": "tampered"'), encoding="utf-8")
        r2 = subprocess.run([sys.executable, str(tool), "--check"], capture_output=True, text=True, cwd=ROOT)
        assert r2.returncode == 1 and "TEMPLATE DRIFT" in r2.stdout
    finally:
        shutil.copy(backup, tpl)
