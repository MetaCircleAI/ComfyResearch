"""Import-smoke coverage for the app entrypoint and HTTP surface.

Regression guard for the class of bug where a refactor breaks a `from
trainer_run import <name>` in some module that the rest of the suite never
imports. The unit suite passed while `python app.py` crashed on startup
because nothing imported `comfy_research.main` / the `api` package. These
tests import them explicitly so any broken import fails CI, not the user.
"""
from __future__ import annotations

import importlib
import pkgutil

import pytest

# The app entrypoint imports trainer_run (and thus torch); skip where absent.
pytest.importorskip("torch")


def test_app_entrypoint_imports() -> None:
    # comfy_research.main pulls in the whole api package via `from comfy_research.api import (...)`.
    importlib.import_module("comfy_research.main")


def test_all_api_modules_import() -> None:
    api = importlib.import_module("comfy_research.api")
    failures: list[str] = []
    for mod in pkgutil.iter_modules(api.__path__, prefix="comfy_research.api."):
        if mod.name.endswith("._notebook_kernel_child"):
            continue
        try:
            importlib.import_module(mod.name)
        except Exception as exc:  # noqa: BLE001 — surface any import-time break
            failures.append(f"{mod.name}: {type(exc).__name__}: {exc}")
    assert not failures, "api modules failed to import:\n" + "\n".join(failures)
