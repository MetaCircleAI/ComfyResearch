"""Persist workflow/template libraries as JSON under ``data/graph_library/`` in the repo."""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import ValidationError

from comfy_research.schemas.saved_graph_library import SavedGraphEntry

router = APIRouter(prefix="/api/graph-library", tags=["graph-library"])

_REPO_ROOT = Path(__file__).resolve().parents[2]
_LIB_DIR = _REPO_ROOT / "data" / "graph_library"
_WORKFLOWS_FILE = _LIB_DIR / "workflows.json"
_TEMPLATES_FILE = _LIB_DIR / "templates.json"
_TEMPLATES_DIR = _LIB_DIR / "templates"

_MAX_ENTRIES = 200

_LIBRARY_JSON_SKIP = frozenset({"workflows.json", "templates.json"})


def _is_single_saved_graph_dict(raw: object) -> bool:
    return isinstance(raw, dict) and all(
        k in raw for k in ("id", "name", "tier", "document", "savedAt")
    )


def _supplemental_template_json_files() -> list[Path]:
    """Legacy one-entry JSON files at the graph_library root (e.g. ``mnist.json``)."""
    return sorted(
        p
        for p in _LIB_DIR.glob("*.json")
        if p.is_file() and p.name not in _LIBRARY_JSON_SKIP
    )


def _template_paths_in_dir() -> list[Path]:
    if not _TEMPLATES_DIR.is_dir():
        return []
    return sorted(p for p in _TEMPLATES_DIR.glob("*.json") if p.is_file())


def _template_file_path(entry_id: str) -> Path:
    if not entry_id or "/" in entry_id or "\\" in entry_id or entry_id.startswith("."):
        raise HTTPException(status_code=400, detail="invalid template id")
    return _TEMPLATES_DIR / f"{entry_id}.json"


def _template_file_path_if_valid(entry_id: str) -> Path | None:
    if not entry_id or "/" in entry_id or "\\" in entry_id or entry_id.startswith("."):
        return None
    return _TEMPLATES_DIR / f"{entry_id}.json"


def _try_parse_saved_graph_file(path: Path) -> SavedGraphEntry | None:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not _is_single_saved_graph_dict(raw):
        return None
    try:
        return SavedGraphEntry.model_validate(raw)
    except ValidationError:
        return None


def _read_entries(path: Path) -> list[SavedGraphEntry]:
    if not path.is_file():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Corrupt library file {path.name}: {e}") from e
    if not isinstance(raw, list):
        raise HTTPException(status_code=500, detail=f"Invalid library file {path.name}: expected a list")
    out: list[SavedGraphEntry] = []
    for item in raw:
        try:
            out.append(SavedGraphEntry.model_validate(item))
        except ValidationError:
            continue
    return out


def _gather_templates() -> list[SavedGraphEntry]:
    """Canonical ``templates/{id}.json``, then legacy ``templates.json``, then root scaffolds. Dedup by id (first wins)."""
    seen: set[str] = set()
    out: list[SavedGraphEntry] = []

    dir_entries: list[SavedGraphEntry] = []
    for path in _template_paths_in_dir():
        e = _try_parse_saved_graph_file(path)
        if e is None:
            continue
        dir_entries.append(e)
    dir_entries.sort(key=lambda e: e.savedAt, reverse=True)
    for e in dir_entries:
        if e.id not in seen:
            out.append(e)
            seen.add(e.id)

    for e in _read_entries(_TEMPLATES_FILE):
        if e.id not in seen:
            out.append(e)
            seen.add(e.id)

    for path in _supplemental_template_json_files():
        e = _try_parse_saved_graph_file(path)
        if e is None or e.id in seen:
            continue
        out.append(e)
        seen.add(e.id)

    return out


def _atomic_write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, indent=2, ensure_ascii=False)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _atomic_write_template_file(entry: SavedGraphEntry) -> None:
    path = _template_file_path(entry.id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = entry.model_dump(mode="json")
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _remove_template_id_from_templates_json(entry_id: str) -> None:
    if not _TEMPLATES_FILE.is_file():
        return
    try:
        raw = json.loads(_TEMPLATES_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return
    if not isinstance(raw, list):
        return
    filtered = [x for x in raw if isinstance(x, dict) and x.get("id") != entry_id]
    if len(filtered) == len(raw):
        return
    _atomic_write_json(_TEMPLATES_FILE, filtered)


def _trim_template_directory() -> None:
    """Keep at most ``_MAX_ENTRIES`` files under ``templates/`` (oldest ``savedAt`` removed first)."""
    items: list[tuple[float, Path]] = []
    for path in _template_paths_in_dir():
        e = _try_parse_saved_graph_file(path)
        if e is None:
            continue
        items.append((e.savedAt, path))
    items.sort(key=lambda x: x[0])
    while len(items) > _MAX_ENTRIES:
        _, path = items.pop(0)
        try:
            path.unlink()
        except OSError:
            pass


def _write_entries(path: Path, entries: list[SavedGraphEntry]) -> None:
    payload = [e.model_dump(mode="json") for e in entries]
    _atomic_write_json(path, payload)


def _path_for_kind(kind: str) -> Path:
    if kind == "workflows":
        return _WORKFLOWS_FILE
    if kind == "templates":
        return _TEMPLATES_FILE
    raise HTTPException(status_code=400, detail="kind must be workflows or templates")


@router.get("/{kind}")
def list_entries(kind: str) -> list[SavedGraphEntry]:
    path = _path_for_kind(kind)
    if kind == "templates":
        return _gather_templates()
    return _read_entries(path)


@router.post("/{kind}")
def add_entry(kind: str, entry: SavedGraphEntry) -> list[SavedGraphEntry]:
    path = _path_for_kind(kind)
    if kind == "templates":
        _atomic_write_template_file(entry)
        _remove_template_id_from_templates_json(entry.id)
        _trim_template_directory()
        return _gather_templates()
    entries = _read_entries(path)
    entries = [e for e in entries if e.id != entry.id]
    entries.insert(0, entry)
    entries = entries[:_MAX_ENTRIES]
    _write_entries(path, entries)
    return entries


@router.delete("/{kind}/{entry_id}")
def delete_entry(kind: str, entry_id: str) -> list[SavedGraphEntry]:
    path = _path_for_kind(kind)
    if kind == "templates":
        tpath = _template_file_path_if_valid(entry_id)
        removed = False
        if tpath is not None and tpath.is_file():
            try:
                tpath.unlink()
                removed = True
            except OSError:
                pass
        if not removed:
            stored = _read_entries(_TEMPLATES_FILE)
            filtered = [e for e in stored if e.id != entry_id]
            if len(filtered) < len(stored):
                _write_entries(_TEMPLATES_FILE, filtered)
                removed = True
        if not removed:
            for sup in _supplemental_template_json_files():
                extra = _try_parse_saved_graph_file(sup)
                if extra is not None and extra.id == entry_id:
                    try:
                        sup.unlink()
                    except OSError:
                        pass
                    break
        return _gather_templates()
    entries = [e for e in _read_entries(path) if e.id != entry_id]
    _write_entries(path, entries)
    return entries
