"""Built-in and user-saved curve templates."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

BUILTIN_TEMPLATES: List[Dict[str, str]] = [
    {"id": "four_phase", "label": "4-Phase Demo (exp → plateau → sigmoid → power)"},
    {"id": "exp_decay", "label": "Exponential Decay"},
    {"id": "sigmoid", "label": "Sigmoid Transition"},
    {"id": "power_law", "label": "Power Law Tail"},
    {"id": "linear_type", "label": "一次型 (Linear)"},
    {"id": "quadratic_type", "label": "二次型 (Quadratic)"},
    {"id": "warmup_plateau", "label": "Warmup + Plateau"},
]

_USER_PREFIX = "user_"
_STORE_PATH = Path(__file__).resolve().parents[1] / "data" / "user_templates.json"


def _builtin_points(name: str, n: int) -> List[Dict[str, float]]:
    t_max = 10000.0
    t = np.linspace(0, t_max, n)

    if name == "four_phase":
        from comfy_research.lmech.lmn.data import ground_truth_loss

        y = ground_truth_loss(t)
    elif name == "exp_decay":
        y = 1.5 * np.exp(-0.0003 * t) + 0.25
    elif name == "sigmoid":
        y = 0.2 + 0.8 / (1 + np.exp(-0.001 * (t - 5000)))
    elif name == "power_law":
        y = 1.0 / (1.0 + t / 1000.0)
    elif name == "linear_type":
        y = 1.7 - 0.00014 * t
    elif name == "quadratic_type":
        y = 1.7 - 1.4e-8 * t**2
    elif name == "warmup_plateau":
        y = np.where(t < 2000, 0.5 + 0.0002 * t, 0.9)
    else:
        raise ValueError(f"Unknown template: {name}")

    return [{"t": float(ti), "loss": float(yi)} for ti, yi in zip(t, y)]


def _resample_points(points: List[Dict[str, float]], n: int) -> List[Dict[str, float]]:
    if len(points) < 2:
        return list(points)
    if len(points) == n:
        return list(points)

    t_arr = np.array([p["t"] for p in points], dtype=float)
    y_arr = np.array([p["loss"] for p in points], dtype=float)
    order = np.argsort(t_arr)
    t_arr, y_arr = t_arr[order], y_arr[order]

    t_min, t_max = t_arr[0], t_arr[-1]
    if t_max - t_min < 1e-9:
        return [{"t": float(t_min), "loss": float(y_arr[0])}]

    t_new = np.linspace(t_min, t_max, n)
    y_new = np.interp(t_new, t_arr, y_arr)
    return [{"t": float(ti), "loss": float(yi)} for ti, yi in zip(t_new, y_new)]


def _load_store() -> List[Dict[str, Any]]:
    if not _STORE_PATH.is_file():
        return []
    try:
        data = json.loads(_STORE_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    templates = data.get("templates")
    return templates if isinstance(templates, list) else []


def _save_store(templates: List[Dict[str, Any]]) -> None:
    _STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _STORE_PATH.write_text(
        json.dumps({"templates": templates}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _is_user_id(template_id: str) -> bool:
    return template_id.startswith(_USER_PREFIX)


def list_templates() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = [
        {"id": t["id"], "label": t["label"], "builtin": True} for t in BUILTIN_TEMPLATES
    ]
    for entry in _load_store():
        tid = entry.get("id")
        label = entry.get("label")
        if isinstance(tid, str) and isinstance(label, str):
            out.append({"id": tid, "label": label, "builtin": False})
    return out


def get_template_points(name: str, n: int = 80) -> List[Dict[str, float]]:
    if _is_user_id(name):
        for entry in _load_store():
            if entry.get("id") == name:
                pts = entry.get("points")
                if not isinstance(pts, list) or len(pts) < 2:
                    raise ValueError(f"Template '{name}' has no valid points")
                normalized = [
                    {"t": float(p["t"]), "loss": float(p["loss"])}
                    for p in pts
                    if isinstance(p, dict) and "t" in p and "loss" in p
                ]
                if len(normalized) < 2:
                    raise ValueError(f"Template '{name}' has no valid points")
                return _resample_points(normalized, n)
        raise ValueError(f"Unknown template: {name}")

    builtin_ids = {t["id"] for t in BUILTIN_TEMPLATES}
    if name not in builtin_ids:
        raise ValueError(f"Unknown template: {name}")
    return _builtin_points(name, n)


def save_user_template(label: str, points: List[Dict[str, float]]) -> Dict[str, Any]:
    label = label.strip()
    if not label:
        raise ValueError("Template name is required")
    if len(label) > 80:
        raise ValueError("Template name must be 80 characters or fewer")
    if len(points) < 5:
        raise ValueError("Need at least 5 points to save a template")

    normalized = [
        {"t": float(p["t"]), "loss": float(p["loss"])}
        for p in points
        if isinstance(p, dict) and "t" in p and "loss" in p
    ]
    if len(normalized) < 5:
        raise ValueError("Need at least 5 valid points")

    order = np.argsort([p["t"] for p in normalized])
    normalized = [normalized[i] for i in order]

    template_id = f"{_USER_PREFIX}{uuid.uuid4().hex[:12]}"
    entry = {"id": template_id, "label": label, "points": normalized}

    store = _load_store()
    store.append(entry)
    _save_store(store)
    return {"id": template_id, "label": label, "builtin": False}


def delete_user_template(template_id: str) -> None:
    if not _is_user_id(template_id):
        raise ValueError("Only custom templates can be deleted")

    store = _load_store()
    new_store = [e for e in store if e.get("id") != template_id]
    if len(new_store) == len(store):
        raise ValueError(f"Template not found: {template_id}")
    _save_store(new_store)


def is_builtin(template_id: str) -> bool:
    return not _is_user_id(template_id)
