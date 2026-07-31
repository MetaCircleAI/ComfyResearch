"""Regenerate ``frontend/src/graph/specCode/toyLanguageRuntimeNotebookEmbeds.generated.ts`` from Python sources."""

from __future__ import annotations

from pathlib import Path


def indent_block(path: Path, start_line: int, end_line: int) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()
    chunk = lines[start_line - 1 : end_line]
    return "\n".join("    " + ln for ln in chunk)


def escape_for_ts_template_literal(s: str) -> str:
    """So ``s`` can live inside ``String.raw`...` `` without closing the literal or interpolating."""
    return s.replace("\\", "\\\\").replace("`", "\\`").replace("${", "\\${")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    eng = root / "comfy_research" / "engine"
    out = root / "frontend" / "src" / "graph" / "specCode" / "toyLanguageRuntimeNotebookEmbeds.generated.ts"

    # Skip module docstring / __future__; keep imports through *_from_seed entrypoints.
    pcfg = escape_for_ts_template_literal(indent_block(eng / "toy_language_pcfg_runtime.py", 20, 178))
    ngram = escape_for_ts_template_literal(indent_block(eng / "toy_language_ngram_runtime.py", 5, 98))
    formal = escape_for_ts_template_literal(indent_block(eng / "toy_language_formal_runtime.py", 5, 83))
    external = escape_for_ts_template_literal(indent_block(eng / "toy_language_external_runtime.py", 5, 345))
    physics = escape_for_ts_template_literal(indent_block(eng / "toy_language_physics_lm_runtime.py", 1, 296))

    parts: list[str] = [
        "/**",
        " * AUTO-GENERATED — run: ``python scripts/gen_toy_language_notebook_embeds.py``",
        " * from the repo root after editing ``comfy_research/engine/toy_language_*.py``.",
        " * Python slices are escaped for TS template literals (`` ` ``, ``\\``, ``${``).",
        " */",
        "",
        "export const PCFG_LM_NOTEBOOK_IMPL_BLOCK = String.raw`",
        pcfg,
        "`;",
        "",
        "export const NGRAM_LM_NOTEBOOK_IMPL_BLOCK = String.raw`",
        ngram,
        "`;",
        "",
        "export const FORMAL_SUITE_LM_NOTEBOOK_IMPL_BLOCK = String.raw`",
        formal,
        "`;",
        "",
        "export const EXTERNAL_TOY_LM_NOTEBOOK_IMPL_BLOCK = String.raw`",
        external,
        "`;",
        "",
        "export const PHYSICS_LM_NOTEBOOK_IMPL_BLOCK = String.raw`",
        physics,
        "`;",
        "",
    ]
    out.write_text("\n".join(parts), encoding="utf-8")
    print(f"Wrote {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
