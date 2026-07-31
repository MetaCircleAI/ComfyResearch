#!/usr/bin/env python3
"""Experiment narrative PDF from ComfyResearch graph JSON (GraphDocument or library template).

Produces natural-language sections (dataset, model, loss, training, observables, analysis
chains) and embeds matplotlib figures when the snapshot includes loss curves, observable
histories, or a stored training-viz PNG.

Example — **sparse attention 1 - plot** (template with embedded viz / loss curves):

  python scripts/graph_report_pdf.py \\
    --input data/graph_library/templates/<template-id>.json \\
    --template-name "sparse attention 1 - plot" \\
    --output reports/sparse-attention-1-plot.pdf
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import re
import textwrap
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


def _pdf_savefig(pdf: PdfPages, fig: plt.Figure, **kwargs: Any) -> None:
    """Save figure into PdfPages; ignore missing-glyph UserWarnings (strict -Werror would abort otherwise)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        pdf.savefig(fig, **kwargs)


LETTER_W_IN = 8.5
LETTER_H_IN = 11.0
# Printable band in **figure fraction** (avoid ``plt.subplots()`` default ~0.125 side margins + transAxes offset).
PDF_PAGE_L = 0.075
PDF_PAGE_R = 0.925
PDF_PAGE_BODY_W = PDF_PAGE_R - PDF_PAGE_L
PDF_BODY_X0 = PDF_PAGE_L  # legacy name for composite / caption alignment
PDF_BODY_X1 = PDF_PAGE_R
PDF_BODY_W = PDF_PAGE_BODY_W

# Body text: wrap width must fit the printable column (~0.85 × 8.5 in at 10 pt — 92 was too wide and clipped).
PDF_TEXT_WRAP_WIDTH = 72
PDF_TEXT_LINES_PER_PAGE = 42
PDF_LINE_HEIGHT_FRAC = 0.0223
PDF_TEXT_Y_TOP = 0.902
PDF_PAGE_HEADER_TITLE_Y = 0.972
PDF_PAGE_HEADER_NUM_Y = 0.938
PDF_PAGE_FOOTER_TOP = 0.058

# Raster buffer for matplotlib figures embedded in letter pages (PDF vector + crisp bitmap).
PDF_RASTER_DPI = 320

def _mpl_figure_to_rgba(fig: plt.Figure, *, dpi: int) -> np.ndarray:
    """Rasterize a matplotlib figure to an RGBA array (figure is closed)."""
    buf = io.BytesIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fig.savefig(
            buf,
            format="png",
            dpi=dpi,
            bbox_inches="tight",
            facecolor=fig.get_facecolor() or "white",
            pad_inches=0.05,
        )
    plt.close(fig)
    buf.seek(0)
    return plt.imread(buf)


def _letter_header_and_page_line(
    letter: plt.Figure,
    *,
    doc_title: str,
    page_number: int,
    total_pages: int,
) -> None:
    letter.text(
        PDF_PAGE_L,
        PDF_PAGE_HEADER_TITLE_Y,
        _pdf_safe_text(doc_title),
        transform=letter.transFigure,
        fontsize=13,
        fontweight="bold",
        va="top",
        ha="left",
    )
    letter.text(
        PDF_PAGE_L,
        PDF_PAGE_HEADER_NUM_Y,
        f"Page {page_number} of {total_pages}",
        transform=letter.transFigure,
        fontsize=9,
        style="italic",
        va="top",
        ha="left",
    )


def _letter_draw_body_lines(letter: plt.Figure, page_lines: list[str], *, y_top: float) -> float:
    """Draw wrapped body lines in figure coordinates; returns y (figure frac) below last line."""
    y = y_top
    for raw in page_lines:
        safe_ln = _pdf_safe_text(raw)
        letter.text(
            PDF_PAGE_L,
            y,
            safe_ln,
            transform=letter.transFigure,
            fontsize=10,
            va="top",
            ha="left",
            family="sans-serif",
            fontweight="bold" if _pdf_line_is_role_heading(safe_ln) else "normal",
        )
        y -= PDF_LINE_HEIGHT_FRAC
    return y


def _letter_place_raster_below(
    letter: plt.Figure,
    img: np.ndarray,
    *,
    y_figure_top: float,
    y_figure_bottom: float,
    caption: str | None,
    caption_wrap_width: int,
    max_width_frac: float | None = None,
    caption_max_lines: int = 8,
) -> None:
    """Place raster image in horizontal band ``[y_figure_bottom, y_figure_top]``, preserving aspect."""
    if y_figure_top <= y_figure_bottom + 0.04:
        return
    max_h_frac = y_figure_top - y_figure_bottom
    if max_width_frac is None:
        max_w_frac = PDF_PAGE_BODY_W
    else:
        # Absolute figure-fraction cap (e.g. 0.96) for figures wider than the text column.
        max_w_frac = min(max(float(max_width_frac), 0.5), 0.97)
    ih, iw = int(img.shape[0]), int(img.shape[1])
    img_w_over_h = max(float(iw) / max(float(ih), 1.0), 1e-6)
    page_w_over_h = LETTER_W_IN / LETTER_H_IN
    w_frac = max_w_frac
    h_frac = w_frac * page_w_over_h / img_w_over_h
    if h_frac > max_h_frac:
        h_frac = max_h_frac
        w_frac = min(max_w_frac, h_frac * img_w_over_h / page_w_over_h)
    x_center = (PDF_PAGE_L + PDF_PAGE_R) / 2.0
    x0_axes = x_center - w_frac / 2.0
    ax_img = letter.add_axes([x0_axes, y_figure_bottom, w_frac, h_frac])
    ax_img.imshow(img, aspect="auto", interpolation="hanning")
    ax_img.axis("off")
    if caption:
        cap = _pdf_safe_text(caption.strip())
        wrapped = textwrap.wrap(cap, width=caption_wrap_width) if cap else []
        y = y_figure_bottom - 0.012
        for ln in wrapped[: max(1, int(caption_max_lines))]:
            letter.text(
                PDF_PAGE_L,
                y,
                ln,
                transform=letter.transFigure,
                fontsize=9,
                va="top",
                ha="left",
                family="sans-serif",
            )
            y -= 0.026


def _pdf_savefig_on_letter_page(
    pdf: PdfPages,
    fig: plt.Figure,
    *,
    doc_title: str,
    page_number: int,
    total_pages: int,
    figure_caption: str | None = None,
    raster_max_width_frac: float | None = None,
    caption_max_lines: int = 18,
) -> None:
    """Rasterize ``fig`` and append one **8.5×11 in** page (same margins as text pages)."""
    img = _mpl_figure_to_rgba(fig, dpi=PDF_RASTER_DPI)
    letter = plt.figure(figsize=(LETTER_W_IN, LETTER_H_IN), facecolor="white")
    _figure_face_white(letter)
    _letter_header_and_page_line(letter, doc_title=doc_title, page_number=page_number, total_pages=total_pages)
    y_fig_top = 0.88
    y_fig_bottom = 0.28
    _letter_place_raster_below(
        letter,
        img,
        y_figure_top=y_fig_top,
        y_figure_bottom=y_fig_bottom,
        caption=figure_caption,
        caption_wrap_width=PDF_TEXT_WRAP_WIDTH + 8,
        max_width_frac=raster_max_width_frac,
        caption_max_lines=caption_max_lines,
    )
    _pdf_savefig(pdf, letter)
    plt.close(letter)


def _pdf_savefig_letter_text_plus_first_figure(
    pdf: PdfPages,
    page_lines: list[str],
    mpl_fig: plt.Figure,
    *,
    doc_title: str,
    page_number: int,
    total_pages: int,
    figure_caption: str | None,
    raster_max_width_frac: float | None = None,
    caption_max_lines: int = 18,
) -> None:
    """One letter page: body text on top, first training figure below (uses slack on last pre-figure text page)."""
    img = _mpl_figure_to_rgba(mpl_fig, dpi=PDF_RASTER_DPI)
    letter = plt.figure(figsize=(LETTER_W_IN, LETTER_H_IN), facecolor="white")
    _figure_face_white(letter)
    _letter_header_and_page_line(letter, doc_title=doc_title, page_number=page_number, total_pages=total_pages)
    y_after_text = _letter_draw_body_lines(letter, page_lines, y_top=PDF_TEXT_Y_TOP)
    gap = 0.018
    y_fig_top = min(y_after_text - gap, 0.88)
    y_fig_bottom = PDF_PAGE_FOOTER_TOP + 0.04
    _letter_place_raster_below(
        letter,
        img,
        y_figure_top=y_fig_top,
        y_figure_bottom=y_fig_bottom,
        caption=figure_caption,
        caption_wrap_width=PDF_TEXT_WRAP_WIDTH + 8,
        max_width_frac=raster_max_width_frac,
        caption_max_lines=caption_max_lines,
    )
    _pdf_savefig(pdf, letter)
    plt.close(letter)


def _should_merge_last_text_page_with_first_figure(pages: list[list[str]], *, lines_per_page: int) -> bool:
    """True when the last text page has enough vertical slack to place the first plot underneath."""
    if not pages:
        return False
    n = len(pages[-1])
    slack = lines_per_page - n
    if slack <= 0:
        return False
    return slack >= 10 or n <= int(lines_per_page * 0.68)


def _strip_light_markdown_for_pdf(s: str) -> str:
    """Matplotlib text has no markdown renderer; strip common inline markers for readable PDF."""
    if not s:
        return s
    out = re.sub(r"`([^`\n]*)`", r"\1", s)
    # Non-greedy ** spans (no nested ** in our narrative strings).
    out = re.sub(r"\*\*(.+?)\*\*", r"\1", out, flags=re.DOTALL)
    return out


def _pdf_safe_text(s: str) -> str:
    """DejaVu Sans (PDF default) lacks some symbols; normalize so text pages always render."""
    if not s:
        return s
    t = (
        s.replace("\u24d8", "[i]")  # ⓘ circled small i
        .replace("\u2139", "[i]")  # ℹ information source
    )
    return _strip_light_markdown_for_pdf(t)


# Role headings in training setup (markdown ** stripped before this runs on each wrapped line).
_PDF_BOLD_HEADING_EXACT: frozenset[str] = frozenset(
    {
        "Dataset:",
        "Model:",
        "Optimizer:",
        "Loss:",
        "Trainer",
        "Trainer:",
        "Training data (legacy handle):",
        "Test data (legacy handle):",
        "Observables logged during training:",
    }
)


def _pdf_line_is_role_heading(line: str) -> bool:
    """True for subsection labels (Dataset, Model, …) so PDF uses bold weight (matplotlib has no ** markdown)."""
    s = (line or "").strip()
    if s in _PDF_BOLD_HEADING_EXACT:
        return True
    # Multi-trainer headings, e.g. "Training setup (trainer 2 of 3)"
    if re.match(r"^Training setup \(trainer \d+ of \d+\)$", s):
        return True
    if re.match(r"^Experimental setup \(trainer \d+ of \d+\)$", s):
        return True
    # Section banners from ``_sections_to_lines`` (``title.upper()``).
    if 4 <= len(s) <= 120 and s.upper() == s and re.match(r"^[A-Z0-9 ()/—:'-]+$", s):
        return True
    return False


_REPO_ROOT = Path(__file__).resolve().parent.parent

_NODE_INFO_LOOKUP_CACHE: dict[str, str] | None = None

# Info icon text for token MLP variants (mirrors ``MlpTokenModelNode.tsx`` blurbs).
_MLP_TOKEN_MODEL_INFO: dict[str, str] = {
    "mlp_token_model": (
        "Token ids -> embeddings -> hidden MLP (depth x width, activation) -> logits. "
        "Tie weights shares the embedding matrix with the final linear when shapes match."
    ),
    "gated_mlp_token_model": "Token ids -> embeddings -> gated hidden MLP blocks using act(Wg h) * (Wv h) -> logits.",
    "moe_mlp_token_model": (
        "Token ids -> embeddings -> softmax gate over expert MLPs -> weighted expert mixture -> logits."
    ),
}

_SKIP_PDF_PARAM_KEYS: frozenset[str] = frozenset(
    {
        "lossHistory",
        "testLossHistory",
        "stepTicks",
        "plotPngBase64",
        "valueHistory",
        "valueHistories",
        "testValueHistory",
        "observableMetricHistories",
        "weightTensorPayloads",
        "lastPlotError",
        "lastSweepSummary",
        "generatedCode",
        "instanceTitle",
        "lastTrainLoopSeconds",
        "lastTrainError",
    }
)


def _info_markdown_to_pdf_plain(s: str) -> str:
    """Strip KaTeX / markdown from UI info strings for matplotlib text."""
    if not s:
        return ""
    def _latex_to_plain(expr: str) -> str:
        x = expr or ""
        # Common wrappers: keep payload text, drop the macro shell.
        x = re.sub(r"\\(?:mathrm|mathbb|mathbf|mathcal|text|operatorname)\{([^}]*)\}", r"\1", x)
        # Keep command names that carry semantics (e.g. \propto -> propto, \alpha -> alpha).
        x = re.sub(r"\\([a-zA-Z]+)", r"\1", x)
        x = x.replace("{", "").replace("}", "")
        x = re.sub(r"\s+", " ", x).strip()
        return x

    t = s.replace("\r\n", "\n")
    t = re.sub(r"\$\$([\s\S]*?)\$\$", lambda m: " " + _latex_to_plain(m.group(1)) + " ", t)
    t = re.sub(r"\$([^$\n]+)\$", lambda m: " " + _latex_to_plain(m.group(1)) + " ", t)
    t = re.sub(r"```[\s\S]*?```", " ", t)
    t = _strip_light_markdown_for_pdf(t)
    t = re.sub(r"\\[a-zA-Z]+\{[^}]*\}", " ", t)
    t = re.sub(r"\\\(|\\\)|\\\[|\\\]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _parse_dataset_node_info_markdown_ts() -> dict[str, str]:
    """Load dataset [i] markdown from ``datasetNodeInfoContent.ts`` (same source as the web UI)."""
    path = _REPO_ROOT / "frontend/src/components/nodes/datasetNodeInfoContent.ts"
    if not path.is_file():
        return {}
    lines = path.read_text(encoding="utf-8").splitlines()
    start_idx = 0
    for i, ln in enumerate(lines):
        if "DATASET_NODE_INFO_MARKDOWN" in ln and "{" in ln:
            start_idx = i + 1
            break
    out: dict[str, str] = {}
    cur_key: str | None = None
    buf: list[str] = []
    key_open = re.compile(r"^\s{2}([a-z0-9_]+):\s*`(.*)$")
    for ln in lines[start_idx:]:
        if "} as const" in ln:
            break
        if cur_key is None:
            m = key_open.match(ln)
            if not m:
                continue
            cur_key = m.group(1)
            first = m.group(2)
            if first.rstrip().endswith("`,"):
                out[cur_key] = first.rstrip()[:-2].rstrip("`")
                cur_key = None
            else:
                buf = [first]
        else:
            if ln.rstrip().endswith("`,"):
                buf.append(ln.rstrip()[:-2])
                out[cur_key] = "\n".join(buf)
                cur_key = None
                buf = []
            else:
                buf.append(ln)
    return out


def _parse_optimizer_node_info_ts() -> dict[str, str]:
    """Load optimizer [i] text from ``optimizerNodeInfoContent.ts``."""
    path = _REPO_ROOT / "frontend/src/components/nodes/optimizerNodeInfoContent.ts"
    if not path.is_file():
        return {}
    lines = path.read_text(encoding="utf-8").splitlines()
    start_idx = 0
    for i, ln in enumerate(lines):
        if "OPTIMIZER_NODE_INFO_TEXT" in ln and "{" in ln:
            start_idx = i + 1
            break
    out: dict[str, str] = {}
    cur_key: str | None = None
    buf: list[str] = []
    key_open = re.compile(r"^\s{2}([a-z0-9_]+):\s*`(.*)$")
    for ln in lines[start_idx:]:
        if ln.strip() == "};":
            break
        if cur_key is None:
            m = key_open.match(ln)
            if not m:
                continue
            cur_key = m.group(1)
            first = m.group(2)
            if first.rstrip().endswith("`,"):
                out[cur_key] = first.rstrip()[:-2].rstrip("`")
                cur_key = None
            else:
                buf = [first]
        else:
            if ln.rstrip().endswith("`,"):
                buf.append(ln.rstrip()[:-2])
                out[cur_key] = "\n".join(buf)
                cur_key = None
                buf = []
            else:
                buf.append(ln)
    return out


def _read_ts_double_quoted_string(line: str, start: int) -> tuple[str, int] | None:
    """Parse a TypeScript double-quoted string starting at index ``start`` (first char inside quotes)."""
    i = start
    parts: list[str] = []
    while i < len(line):
        ch = line[i]
        if ch == "\\":
            if i + 1 >= len(line):
                parts.append("\\")
                break
            n = line[i + 1]
            if n == "n":
                parts.append("\n")
            elif n == "t":
                parts.append("\t")
            elif n == "r":
                parts.append("\r")
            else:
                parts.append(n)
            i += 2
            continue
        if ch == '"':
            return "".join(parts), i + 1
        parts.append(ch)
        i += 1
    return None


def _parse_observable_info_ts() -> dict[str, str]:
    """Load observable [i] blurbs from ``observableNodeInfoMarkdown.ts``."""
    path = _REPO_ROOT / "frontend/src/components/nodes/observableNodeInfoMarkdown.ts"
    if not path.is_file():
        return {}
    lines = path.read_text(encoding="utf-8").splitlines()
    out: dict[str, str] = {}
    in_obj = False
    i = 0
    while i < len(lines):
        ln = lines[i]
        if not in_obj:
            if "const OBSERVABLE_INFO" in ln and "Record" in ln:
                in_obj = True
            i += 1
            continue
        if ln.strip().startswith("};"):
            break
        m_key = re.match(r"^\s{2}([a-z0-9_]+):\s*$", ln)
        if m_key and i + 1 < len(lines):
            key = m_key.group(1)
            nxt = lines[i + 1]
            mq = re.search(r'"', nxt)
            if mq:
                parsed = _read_ts_double_quoted_string(nxt, mq.start() + 1)
                if parsed is not None:
                    val, _end = parsed
                    out[key] = val
                    i += 2
                    continue
        i += 1
    return out


def _parse_node_manifest_hints() -> dict[str, str]:
    """Load short catalog blurbs from the generated node manifest."""
    path = _REPO_ROOT / "comfy_research/generated/node_manifest.json"
    if not path.is_file():
        return {}
    entries = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, str] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        node_type = entry.get("type")
        hint = entry.get("hint")
        if isinstance(node_type, str) and isinstance(hint, str) and hint.strip():
            out[node_type] = hint
    return out


def _node_library_info_text(node_type: str) -> str:
    """Merged [i] description: hints, then richer dataset/optimizer/observable markdown where present."""
    global _NODE_INFO_LOOKUP_CACHE
    if _NODE_INFO_LOOKUP_CACHE is None:
        merged: dict[str, str] = {}
        merged.update(_parse_node_manifest_hints())
        merged.update(_parse_dataset_node_info_markdown_ts())
        merged.update(_parse_optimizer_node_info_ts())
        merged.update(_parse_observable_info_ts())
        for k, v in _MLP_TOKEN_MODEL_INFO.items():
            merged.setdefault(k, v)
        _NODE_INFO_LOOKUP_CACHE = merged
    return (_NODE_INFO_LOOKUP_CACHE.get(node_type) or "").strip()


def _serialize_param_value(v: Any, *, max_len: int = 140) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, str):
        one = v.replace("\n", " ").strip()
        return one if len(one) <= max_len else one[: max_len - 3] + "..."
    if isinstance(v, (list, tuple)):
        if len(v) > 32:
            return f"[{len(v)} items]"
        inner = ", ".join(_serialize_param_value(x, max_len=48) for x in v[:32])
        return f"[{inner}]"
    if isinstance(v, dict):
        s = json.dumps(v, default=str, separators=(",", ":"))
        return s if len(s) <= max_len else s[: max_len - 3] + "..."
    s = str(v)
    return s if len(s) <= max_len else s[: max_len - 3] + "..."


def _iter_node_hyperparameter_kv(d: dict[str, Any]) -> list[tuple[str, str]]:
    """Sorted (key, display value) pairs; large blobs and plot-only keys omitted."""
    if not d:
        return []
    out: list[tuple[str, str]] = []
    for k in sorted(d.keys()):
        if k in _SKIP_PDF_PARAM_KEYS:
            continue
        if k.startswith("plot") and k != "plotTitle":
            continue
        v = d[k]
        blob = json.dumps(v, default=str) if not isinstance(v, str) else v
        if len(blob) > 360:
            out.append((k, "(large value omitted)"))
            continue
        out.append((k, _serialize_param_value(v, max_len=160)))
    return out


def _format_node_hyperparameters(d: dict[str, Any]) -> str:
    """Single-line sorted key=value list for PDF (large / plot fields omitted)."""
    pairs = _iter_node_hyperparameter_kv(d)
    return "; ".join(f"{k}={v}" for k, v in pairs)


def _describe_node_blocks(
    n: dict[str, Any],
    nodes: dict[str, dict[str, Any]],
    edges: list[dict[str, Any]],
    *,
    include_hyperparameters: bool = True,
    include_library_blurb: bool = True,
) -> list[str]:
    """Per-node blocks: type + library blurb, optional parameters on one comma-separated line (no wiring snapshot)."""
    t = str(n.get("type", "?"))
    d = _d(n)
    blocks: list[str] = []
    if not include_library_blurb:
        blocks.append(_topology_node_label(n))
        return blocks
    info_raw = _node_library_info_text(t)
    if info_raw:
        plain = _info_markdown_to_pdf_plain(info_raw)
        if plain:
            blocks.append(f"**{t}** — {plain}")
        else:
            blocks.append(f"**{t}**")
    else:
        blocks.append(f"**{t}**")
    show_hp = include_hyperparameters and not str(t).startswith("observable_")
    if show_hp:
        pairs = _iter_node_hyperparameter_kv(d)
        if pairs:
            param_line = ", ".join(f"`{k}`={v}" for k, v in pairs)
            blocks.append(f"**Parameters:** {param_line}")
    return blocks


def _append_dataset_reference_blurb(paras: list[str], n: dict[str, Any]) -> None:
    """Append the same explanatory text as the dataset [i] panel (trimmed) when available."""
    t = str(n.get("type") or "")
    info_raw = _node_library_info_text(t)
    if not info_raw:
        return
    plain = _info_markdown_to_pdf_plain(info_raw)
    if len(plain) < 24:
        return
    cap = plain[:950] + ("…" if len(plain) > 950 else "")
    paras.append(f"**More about this dataset:** {cap}")


def _extend_role_paragraphs(
    paras: list[str],
    role_label: str,
    attached: list[dict[str, Any]],
    nodes: dict[str, dict[str, Any]],
    edges: list[dict[str, Any]],
    *,
    include_hyperparameters: bool = True,
    include_library_blurb: bool = True,
    use_narrative_descriptors: bool = False,
) -> None:
    """Emit **Role:** on the same line as the first node's headline; further nodes follow as usual."""
    if use_narrative_descriptors:
        dataset_roles = frozenset({"Dataset", "Training data (legacy handle)", "Test data (legacy handle)"})
        for ni, n in enumerate(attached):
            body = _describe_node(n, nodes, edges).strip()
            if not body:
                continue
            if ni == 0:
                paras.append(f"**{role_label}:** {body}")
            else:
                paras.append(body)
            if role_label in dataset_roles:
                _append_dataset_reference_blurb(paras, n)
        return
    for ni, n in enumerate(attached):
        blocks = _describe_node_blocks(
            n,
            nodes,
            edges,
            include_hyperparameters=include_hyperparameters,
            include_library_blurb=include_library_blurb,
        )
        if not blocks:
            continue
        if ni == 0:
            paras.append(f"**{role_label}:** {blocks[0]}")
            paras.extend(blocks[1:])
        else:
            paras.extend(blocks)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_document(raw: Any, template_id: str | None, template_name: str | None) -> tuple[dict[str, Any], dict[str, Any]]:
    meta: dict[str, Any] = {}
    if isinstance(raw, list):
        chosen = None
        for item in raw:
            if not isinstance(item, dict):
                continue
            if template_id and item.get("id") == template_id:
                chosen = item
                break
            if template_name and item.get("name") == template_name:
                chosen = item
                break
        if chosen is None:
            ids = [i.get("id") for i in raw if isinstance(i, dict)]
            raise SystemExit(
                "Could not find template in array. "
                f"Pass --template-id or --template-name. Known ids (first 12): {ids[:12]!r}"
            )
        doc = chosen.get("document")
        if not isinstance(doc, dict):
            raise SystemExit("Template entry has no 'document' object.")
        meta = {k: chosen.get(k) for k in ("id", "name", "tier") if k in chosen}
        return doc, meta
    if isinstance(raw, dict) and "document" in raw and isinstance(raw["document"], dict):
        meta = {k: raw.get(k) for k in ("id", "name", "tier") if k in raw}
        return raw["document"], meta
    if isinstance(raw, dict) and "nodes" in raw and "edges" in raw:
        return raw, meta
    raise SystemExit("Unsupported JSON: expected GraphDocument, template object, or array of templates.")


def _nodes_map(doc: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for n in doc.get("nodes") or []:
        if isinstance(n, dict) and n.get("id"):
            out[str(n["id"])] = n
    return out


def _edges(doc: dict[str, Any]) -> list[dict[str, Any]]:
    return [e for e in (doc.get("edges") or []) if isinstance(e, dict)]


def _incoming(edges: list[dict[str, Any]], node_id: str) -> list[dict[str, Any]]:
    return [e for e in edges if e.get("target") == node_id]


def _outgoing(edges: list[dict[str, Any]], node_id: str) -> list[dict[str, Any]]:
    return [e for e in edges if e.get("source") == node_id]


def _d(n: dict[str, Any]) -> dict[str, Any]:
    d = n.get("data")
    return d if isinstance(d, dict) else {}


def _fmt_list_or1(v: Any) -> str:
    """Format React ``ListOr1`` fields saved on nodes (scalar or small list)."""
    if v is None:
        return "?"
    if isinstance(v, list):
        if len(v) == 0:
            return "?"
        if len(v) == 1:
            return str(v[0])
        return ", ".join(str(x) for x in v[:12]) + ("…" if len(v) > 12 else "")
    return str(v)


def _mlp_hidden_layers_phrase(d: dict[str, Any]) -> str:
    """Human-readable hidden MLP geometry (handles ``depth``×``width`` as well as legacy ``hiddenSizes``)."""
    hs = d.get("hiddenSizes")
    if isinstance(hs, list) and len(hs) > 0:
        return _fmt_list_or1(hs)
    if isinstance(hs, (int, float)) and not isinstance(hs, bool):
        return str(int(hs)) if float(hs) == int(hs) else str(hs)

    def _to_float(x: Any) -> float | None:
        if x is None:
            return None
        if isinstance(x, list) and x:
            try:
                return float(x[0])
            except (TypeError, ValueError):
                return None
        try:
            return float(x)
        except (TypeError, ValueError):
            return None

    dep = _to_float(d.get("depth"))
    wid = _to_float(d.get("width"))
    if dep is not None and wid is not None and dep > 0:
        di, wi = int(round(dep)), int(round(wid))
        if di == 1:
            return f"one hidden layer with **{wi}** units"
        return f"**{di}** hidden layers with **{wi}** units each"
    legacy = d.get("widths")
    if legacy is not None:
        return _fmt_list_or1(legacy)
    return "?"


def _describe_token_prediction_dataset(n: dict[str, Any]) -> str:
    d = _d(n)
    v = d.get("vocabSize", "?")
    c = d.get("contextLength", "?")
    tr = d.get("trainSize", "?")
    te = d.get("testSize", "?")
    wt = d.get("whichToken", "?")
    mode = str(d.get("retrievalMode", "position")).strip().lower()
    seed = d.get("seed", "?")
    wt_txt = "the last token in each context window" if wt == -1 else f"a fixed token position (index {wt}) inside the window"
    if mode == "content":
        target_txt = "the earlier token that is closest in embedding space to the final token (a harder retrieval-style target)"
    else:
        target_txt = wt_txt
    return (
        f"We use a small synthetic **next-token** dataset: vocabulary size **{v}**, each example is a length-**{c}** "
        f"context, with **{tr}** training windows and **{te}** held-out windows. "
        f"The supervised target is {target_txt}. "
        f"Retrieval mode is **{mode}** (this changes how the “correct” next token is chosen). "
        f"Random seed **{seed}** controls how the synthetic text is generated."
    )


def _describe_linear_dataset(n: dict[str, Any]) -> str:
    d = _d(n)
    return (
        f"The data come from a **linear regression toy**: inputs live in **{d.get('inputDim', '?')}** dimensions, "
        f"targets in **{d.get('outputDim', '?')}** dimensions, with **{d.get('trainSize', '?')}** training points and "
        f"**{d.get('testSize', '?')}** test points. Inputs are drawn from **{d.get('inputDistribution', '?')}** with "
        f"additive noise at level **{d.get('noiseLevel', '?')}** (seed **{d.get('seed', '?')}** for reproducibility)."
    )


def _describe_symbolic_dataset(n: dict[str, Any]) -> str:
    d = _d(n)
    ts = d.get("trainSize", "?")
    te = d.get("testSize", "?")
    seed = d.get("seed", "?")
    xmin = d.get("xMin", "?")
    xmax = d.get("xMax", "?")
    return (
        "Inputs are sampled from a **closed-form symbolic function** (think “type an equation, get a curve”). "
        f"We draw **{ts}** training samples and **{te}** test samples on **[{xmin}, {xmax}]** with seed **{seed}**."
    )


def _describe_random_input_distribution(n: dict[str, Any]) -> str:
    d = _d(n)
    return (
        f"Random **inputs x** are drawn in **{d.get('inputDim', '?')}** dimensions from the law **{d.get('inputDistribution', '?')}**, "
        f"with extra jitter drawn from **{d.get('noiseDistribution', '?')}** at magnitude **{d.get('noiseLevel', '?')}** "
        f"(seed **{d.get('seed', '?')}**)."
    )


def _describe_teacher_dataset(n: dict[str, Any]) -> str:
    d = _d(n)
    return (
        "A **teacher–student** setup: random inputs are passed through a fixed “teacher” MLP to obtain clean targets "
        f"**y = f_teacher(x)**. We keep **{d.get('trainSize', '?')}** training pairs and **{d.get('testSize', '?')}** "
        f"test pairs (seed **{d.get('seed', '?')}**)."
    )


def _describe_attention_only(n: dict[str, Any]) -> str:
    d = _d(n)
    causal = d.get("causalAttention", "yes")
    causal_note = (
        "each position only attends to earlier positions (causal / autoregressive masking)"
        if str(causal).strip().lower() in ("yes", "true", "1", "on")
        else "positions can attend across the whole sequence (no causal mask)"
    )
    return (
        "The model block is a pure **multi-head self-attention** layer on sequence tensors shaped roughly "
        f"**[batch, time, d]** with **d = {d.get('embedDim', '?')}**, **T = {d.get('contextLength', '?')}**, "
        f"and **{d.get('numHeads', '?')}** heads. "
        f"Attention is configured so that **{causal_note}**. "
        f"Initialization uses seed **{d.get('seed', '?')}**. "
        "(There is no separate token embedding or language-modeling head in this block.)"
    )


def _describe_numeric_transformer(n: dict[str, Any]) -> str:
    d = _d(n)
    causal = d.get("causalAttention", "yes")
    causal_note = (
        "causal attention (left-to-right)"
        if str(causal).strip().lower() in ("yes", "true", "1", "on")
        else "bidirectional attention over the full sequence"
    )
    return (
        "We use a **numeric Transformer encoder**: it maps each timestep of a length-**T** sequence, "
        f"where **T = {d.get('contextLength', '?')}**, from **D_in = {d.get('inputDim', '?')}** features per position "
        f"to **D_out = {d.get('outputDim', '?')}** outputs. "
        f"Hidden size (model width) is **{d.get('modelDim', '?')}**, with **{d.get('numHeads', '?')}** heads, "
        f"**{d.get('numLayers', '?')}** stacked blocks, and feed-forward width **{d.get('ffDim', '?')}**. "
        f"Attention style: **{causal_note}**. Random seed **{d.get('seed', '?')}** sets weight init."
    )


def _describe_transformer_token(n: dict[str, Any]) -> str:
    d = _d(n)
    tie = d.get("tieEmbeddingLmHead", "yes")
    tied = str(tie).strip().lower() not in ("no", "false", "0", "off")
    tie_note = "the token embedding matrix is **tied** to the final logits projection" if tied else "token embeddings and the logits layer use **separate** weights"
    causal = d.get("causalAttention", "yes")
    causal_note = (
        "runs as a **causal** language model (standard GPT-style masking)"
        if str(causal).strip().lower() in ("yes", "true", "1", "on")
        else "uses **bidirectional** context at every position"
    )
    return (
        "The core architecture is a small **Transformer language model** on discrete tokens: it reads a batch of "
        f"context windows of length **T = {d.get('contextLength', '?')}** over a vocabulary of size **V = {d.get('vocabSize', '?')}** "
        "and predicts **next-token logits at the final position**. "
        f"Width **{d.get('modelDim', '?')}**, **{d.get('numHeads', '?')}** heads, **{d.get('numLayers', '?')}** layers, "
        f"feed-forward width **{d.get('ffDim', '?')}**. "
        f"We use **{tie_note}**, and the stack **{causal_note}**. "
        f"Initialization seed: **{d.get('seed', '?')}**."
    )


def _describe_transformer_multi_token(n: dict[str, Any]) -> str:
    d = _d(n)
    causal = d.get("causalAttention", "yes")
    causal_note = (
        "with causal masking along the sequence axis"
        if str(causal).strip().lower() in ("yes", "true", "1", "on")
        else "with bidirectional attention along the sequence axis"
    )
    return (
        f"This variant predicts **several token slots at once**: each timestep carries **K = {d.get('tokensPerPosition', '?')}** "
        f"parallel tokens over a length-**L = {d.get('contextLength', '?')}** context, vocabulary **V = {d.get('vocabSize', '?')}**, "
        f"and emits logits for every slot at the **final timestep**. "
        f"Model width **{d.get('modelDim', '?')}**, **{d.get('numHeads', '?')}** heads, **{d.get('numLayers', '?')}** layers, "
        f"FF width **{d.get('ffDim', '?')}**, configured **{causal_note}**. "
        f"Init seed **{d.get('seed', '?')}**."
    )


def _describe_mlp(n: dict[str, Any]) -> str:
    d = _d(n)
    hidden_txt = _mlp_hidden_layers_phrase(d)
    return (
        f"The learner is a fully-connected **MLP** with {hidden_txt}, "
        f"**{_fmt_list_or1(d.get('activation'))}** nonlinearities, and init seed **{_fmt_list_or1(d.get('seed'))}**. "
        "Input and output widths are whatever this block is configured to accept and produce in this run."
    )


def _describe_memorization_a_dataset(n: dict[str, Any]) -> str:
    d = _d(n)
    return (
        "A **memorization-style classification** benchmark: continuous inputs and class labels are drawn **independently**, "
        f"so the task measures raw capacity rather than a smooth input→label rule. "
        f"Input dimension **{_fmt_list_or1(d.get('inputDim'))}**, **{_fmt_list_or1(d.get('outputDim'))}** classes, "
        f"**{_fmt_list_or1(d.get('trainSize'))}** training and **{_fmt_list_or1(d.get('testSize'))}** test examples; "
        f"input law **{_fmt_list_or1(d.get('inputDistribution'))}**, label law **{_fmt_list_or1(d.get('outputDistribution'))}** "
        f"(shape parameter **{_fmt_list_or1(d.get('alpha'))}** where it applies). Seed **{_fmt_list_or1(d.get('seed'))}**."
    )


def _describe_memorization_b_dataset(n: dict[str, Any]) -> str:
    d = _d(n)
    return (
        "A **memorization B** benchmark: **categorical** inputs and class labels are drawn **independently** from a shared "
        f"vocabulary of size **{_fmt_list_or1(d.get('vocabSize', d.get('inputDim')))}**, "
        f"with **{_fmt_list_or1(d.get('trainSize'))}** training and **{_fmt_list_or1(d.get('testSize'))}** test draws; "
        f"label law **{_fmt_list_or1(d.get('outputDistribution'))}** (shape **{_fmt_list_or1(d.get('alpha'))}**). "
        f"Seed **{_fmt_list_or1(d.get('seed'))}**."
    )


def _describe_kan(n: dict[str, Any]) -> str:
    d = _d(n)
    return (
        f"We fit a **KAN-style** regression net (learnable univariate splines per edge) with **{d.get('inputDim', '?')}** inputs, "
        f"**{d.get('outputDim', '?')}** outputs, **{d.get('depth', '?')}** depth, hidden width **{d.get('width', '?')}**, "
        f"spline grid resolution **{d.get('grid', '?')}**, order **{d.get('k', '?')}**, base activation **{d.get('baseFun', '?')}**, "
        f"seed **{d.get('seed', '?')}**."
    )


def _describe_residual_ln(n: dict[str, Any]) -> str:
    d = _d(n)
    return (
        f"The residual tower repeats **{d.get('depth', '?')}** blocks, each operating on vectors of width **{d.get('dim', '?')}** "
        f"with residual strength **α = {d.get('alpha', '?')}**, **{d.get('lnMode', '?')}** LayerNorm placement, "
        f"and **{d.get('activation', '?')}** activations (seed **{d.get('seed', '?')}**)."
    )


def _describe_adam(n: dict[str, Any]) -> str:
    d = _d(n)
    return (
        f"We optimize with **Adam** using learning rate **{d.get('learningRate', '?')}**, momentum decay "
        f"**β₁ = {d.get('beta1', '?')}**, variance decay **β₂ = {d.get('beta2', '?')}**, and numerical floor "
        f"**ε = {d.get('epsilon', '?')}** (the usual Adam defaults unless you changed them)."
    )


def _describe_adamw(n: dict[str, Any]) -> str:
    d = _d(n)
    return (
        f"We optimize with **AdamW** using learning rate **{d.get('learningRate', '?')}**, momentum decay "
        f"**beta1 = {d.get('beta1', '?')}**, variance decay **beta2 = {d.get('beta2', '?')}**, numerical floor "
        f"**eps = {d.get('epsilon', '?')}**, and decoupled weight decay **{d.get('weightDecay', '?')}**."
    )


def _describe_sgd(n: dict[str, Any]) -> str:
    d = _d(n)
    return (
        f"We optimize with **SGD** using learning rate **{d.get('learningRate', '?')}**, momentum "
        f"**{d.get('momentum', 0)}**, and weight decay **{d.get('weightDecay', 0)}**."
    )


def _describe_signsgd(n: dict[str, Any]) -> str:
    d = _d(n)
    return (
        f"We optimize with **SignSGD** using learning rate **{d.get('learningRate', '?')}** and weight decay "
        f"**{d.get('weightDecay', 0)}**, stepping by the sign of each gradient component."
    )


def _describe_shampoo(n: dict[str, Any]) -> str:
    d = _d(n)
    return (
        f"We optimize with **Shampoo** using learning rate **{d.get('learningRate', '?')}**, momentum "
        f"**{d.get('momentum', 0)}**, inverse-root refresh frequency **{d.get('preconditionFrequency', 10)}**, "
        f"max preconditioner dimension **{d.get('maxPreconditionerDim', 1024)}**, and weight decay "
        f"**{d.get('weightDecay', 0)}**."
    )


def _describe_soap(n: dict[str, Any]) -> str:
    d = _d(n)
    return (
        f"We optimize with **SOAP** using learning rate **{d.get('learningRate', '?')}**, beta1 "
        f"**{d.get('beta1', '?')}**, beta2 **{d.get('beta2', '?')}**, basis refresh frequency "
        f"**{d.get('preconditionFrequency', 10)}**, max preconditioner dimension "
        f"**{d.get('maxPreconditionerDim', 1024)}**, and weight decay **{d.get('weightDecay', 0)}**."
    )


def _describe_ce_loss(n: dict[str, Any]) -> str:
    d = _d(n)
    return (
        f"The objective is **average cross-entropy** over the supervised targets (classification / next-token), "
        f"optionally scaled by **{d.get('lossScale', 1)}**."
    )


def _describe_mse_loss(n: dict[str, Any]) -> str:
    d = _d(n)
    mode = str(d.get("lossMaskMode") or "all").strip().lower()
    tmask = d.get("lossMaskContextLength", 1)
    mask = ""
    if mode == "last_context":
        mask = (
            " Only the **last time step** along the prediction horizon is supervised (earlier positions are masked out), "
            f"grouping spans of length **{tmask}** when the loss groups context that way."
        )
    elif mode == "custom":
        mask = (
            " A **custom mask** picks which positions contribute to the average squared error, "
            f"with context grouping length **{tmask}**."
        )
    return (
        f"We minimize **mean squared error** between predictions and targets, scaled by **{d.get('lossScale', 1)}**."
        f"{mask}"
    )


def _describe_trainer(n: dict[str, Any]) -> str:
    d = _d(n)
    steps = d.get("trainingSteps", "?")
    logf = d.get("logFrequency", "?")
    return (
        f"Training runs for **{steps}** optimizer steps, writing scalars to disk every **{logf}** steps so we can plot curves. "
        "Each step updates the model using the chosen dataset, optimizer, and loss; any optional observables are logged on the same schedule."
    )


def _describe_obs_weight_l2(_n: dict[str, Any]) -> str:
    return (
        "**Weight L2 observable:** logs the L2 norm of model parameters during training "
        "(overall weight vector magnitude)."
    )


def _describe_observable_accuracy(_n: dict[str, Any]) -> str:
    return (
        "**Accuracy observable:** tracks supervised prediction accuracy (top-1 style) on the batches used during training."
    )


def _describe_obs_weight_l1(_n: dict[str, Any]) -> str:
    return "**Weight L1 observable:** logs the L1 norm of model parameters during training."


def _describe_kan_reg(n: dict[str, Any]) -> str:
    d = _d(n)
    metric = d.get("regMetric", "edge_forward_spline_n")
    return (
        f"**KAN reg** (pykan): adds λ·get_reg(metric, …) to the trainer MSE objective for **kan_model**; "
        f"metric `{metric}` with coefficients taken from the saved configuration."
    )


def _describe_obs_emb_traj(n: dict[str, Any]) -> str:
    lab = _d(n).get("label") or "embedding trajectory"
    return (
        f"**Embedding trajectory observable** ({lab}): tracks how token embeddings move over training "
        "(relative drift / trajectory-style summary in the app)."
    )


def _describe_obs_user(n: dict[str, Any], nodes: dict[str, dict[str, Any]], edges: list[dict[str, Any]]) -> str:
    d = _d(n)
    label = d.get("label") or "user observable"
    chain = _summarize_upstream_chain(n["id"], nodes, edges, max_hops=14)
    chain_note = ""
    if chain and chain != "(no tensor path edges found in snapshot)":
        chain_note = f" Conceptually the signal is built from: {chain}."
    return (
        f"**User-defined scalar observable** “{label}”: each logging step stores **one number** derived from tensors "
        f"in the upstream analysis chain.{chain_note}"
    )


def _summarize_upstream_chain(
    start_id: str,
    nodes: dict[str, dict[str, Any]],
    edges: list[dict[str, Any]],
    max_hops: int,
) -> str:
    labels: list[str] = []
    cur = start_id
    seen: set[str] = set()
    for _ in range(max_hops):
        if cur in seen:
            break
        seen.add(cur)
        inc = _incoming(edges, cur)
        if not inc:
            break
        e = inc[0]
        src = str(e.get("source", ""))
        sn = nodes.get(src)
        if not sn:
            labels.append("unknown")
            break
        t = sn.get("type", "?")
        labels.append(_short_node_label(sn))
        if t in (
            "token_prediction_dataset",
            "linear_dataset",
            "symbolic_func_dataset",
            "teacher_dataset",
            "random_input_distribution",
            "attention_only_model",
            "numeric_transformer_model",
            "transformer_token_model",
            "transformer_multi_token_model",
            "mlp_model",
            "kan_model",
            "visualize_kan",
            "residual_ln_model",
        ):
            break
        cur = src
    labels.reverse()
    return " → ".join(labels) if labels else "(no tensor path edges found in snapshot)"


def _short_node_label(n: dict[str, Any]) -> str:
    t = str(n.get("type", "?"))
    d = _d(n)
    if t == "tensor_selector":
        keys = d.get("selectedTensorKeys")
        if isinstance(keys, list) and keys:
            return "tensor selector (multi-field)"
        return "tensor selector"
    if t == "effective_rank":
        return "effective rank"
    if t == "series_endpoint_gap":
        return "series endpoint gap"
    if t == "basic_calculator":
        return "basic calculator"
    if t == "tensor_viz_0d":
        return "scalar tensor view"
    if t == "tensor_viz_general":
        return "tensor view"
    if t == "model_weight_tensors":
        return "weight tensors"
    if t == "attention_only_model":
        return "attention-only block"
    if t == "mlp_model":
        return "MLP"
    if t == "kan_model":
        return "KAN"
    if t == "visualize_kan":
        return "KAN plot"
    if t == "residual_ln_model":
        return "residual LayerNorm stack"
    if t == "numeric_transformer_model":
        d = _d(n)
        return f"numeric Transformer (T={d.get('contextLength', '?')})"
    if t == "transformer_token_model":
        d = _d(n)
        return f"token Transformer (V={d.get('vocabSize', '?')}, T={d.get('contextLength', '?')})"
    if t == "transformer_multi_token_model":
        d = _d(n)
        return (
            f"multi-token Transformer (V={d.get('vocabSize', '?')}, L={d.get('contextLength', '?')}, "
            f"K={d.get('tokensPerPosition', '?')})"
        )
    return t.replace("_", " ").strip() or "block"


def _describe_visualize_kan(n: dict[str, Any]) -> str:
    d = _d(n)
    has_png = bool(str(d.get("plotPngBase64", "")).strip())
    err = d.get("lastPlotError")
    err_txt = f" Last error: `{err}`." if err else ""
    png_txt = " Snapshot includes a cached KAN plot image." if has_png else " No cached plot PNG in snapshot."
    return (
        f"**Visualize KAN** (pykan `plot()`): when a dataset is attached, plots use that split's size and sampling law; "
        f"otherwise random `sampleCount`.{png_txt}{err_txt}"
    )


def _describe_training_viz(n: dict[str, Any]) -> str:
    d = _d(n)
    metric = d.get("yPlotMetric", "loss")
    sweep = d.get("lastSweepSummary")
    has_loss = bool(d.get("lossHistory"))
    sweep_txt = f" (Last sweep summary stored with the snapshot: {sweep})." if sweep else ""
    data_txt = (
        " The export already contains sampled training curves suitable for plotting."
        if has_loss
        else " No loss history was embedded in this JSON snapshot yet—train inside ComfyResearch and re-save if you need curves."
    )
    return (
        f"A **training visualization** panel tracks **{metric}** versus optimization step (train/test lines follow whatever you toggled in the UI)."
        f"{sweep_txt}{data_txt}"
    )


def _describe_observable_viz(n: dict[str, Any]) -> str:
    d = _d(n)
    name = d.get("observableName") or "observable"
    variant = d.get("vizVariant", "?")
    has_hist = bool(d.get("valueHistory"))
    hist_txt = " Saved per-step values are available for plotting." if has_hist else " No per-step history was stored in this snapshot."
    return (
        f"**Observable chart** for “{name}” (display variant **{variant}**): mirrors whatever scalar the trainer is logging for that signal. "
        f"{hist_txt}"
    )


def _describe_tensor_selector(n: dict[str, Any]) -> str:
    d = _d(n)
    keys = d.get("selectedTensorKeys")
    if isinstance(keys, list) and keys:
        joined = ", ".join(str(k) for k in keys)
        return (
            "**Tensor selector** routes whichever internal activations you chose downstream; here the selections are: "
            f"{joined}."
        )
    sk = d.get("selectedTensorKey", "?")
    return f"**Tensor selector** forwards the **{sk}** tensor slice to whatever is downstream."


def _describe_effective_rank(n: dict[str, Any]) -> str:
    d = _d(n)
    ot = d.get("outputTensor") if isinstance(d.get("outputTensor"), dict) else {}
    vals = ot.get("values") if isinstance(ot, dict) else None
    if isinstance(vals, list) and vals:
        return f"**Effective rank** computed on the selected matrix; latest cached scalar value ≈ {float(vals[0]):.6g}."
    return "**Effective rank:** summarizes effective rank of the selected matrix tensor (no cached scalar in this snapshot)."


def _describe_series_endpoint_gap(n: dict[str, Any]) -> str:
    d = _d(n)
    ot = d.get("outputTensor") if isinstance(d.get("outputTensor"), dict) else {}
    vals = ot.get("values") if isinstance(ot, dict) else None
    if isinstance(vals, list) and vals:
        return (
            f"**Series endpoint gap** implements **Δ = x_last − x_first** on a 1D logged series; the latest cached value is ≈ **{float(vals[0]):.6g}**."
        )
    return (
        "**Series endpoint gap** turns a 1D curve into a single number **Δ = x_last − x_first**, i.e. how much the series moved "
        "from the first logged step to the last (no cached scalar in this snapshot)."
    )


def _describe_model_weight_tensors(n: dict[str, Any]) -> str:
    d = _d(n)
    w = d.get("weightTensorPayloads")
    n_w = len(w) if isinstance(w, dict) else 0
    return (
        f"**Model weight tensors** materializes parameter tensors from the upstream model ({n_w} tensors in this snapshot)."
        if n_w
        else "**Model weight tensors** is present (no payload in this snapshot)."
    )


def _describe_basic_calculator(n: dict[str, Any]) -> str:
    d = _d(n)
    latex = str(d.get("equationLatex") or "").strip().replace("`", "'")
    if latex:
        snippet = latex if len(latex) <= 160 else latex[:157] + "…"
        return (
            "**Basic calculator** combines upstream **scalar** inputs with the algebraic rule configured for this run "
            f"(compact copy: `{snippet}`)."
        )
    return "**Basic calculator** combines scalar tensors using a small expression configured in the UI."


def _describe_generic(n: dict[str, Any]) -> str:
    label = _topology_node_label(n)
    return f"We include **{label}** in this experiment configuration."


_DESCRIBERS: dict[str, Callable[[dict[str, Any]], str]] = {
    "token_prediction_dataset": _describe_token_prediction_dataset,
    "linear_dataset": _describe_linear_dataset,
    "symbolic_func_dataset": _describe_symbolic_dataset,
    "random_input_distribution": _describe_random_input_distribution,
    "teacher_dataset": _describe_teacher_dataset,
    "memorization_a_dataset": _describe_memorization_a_dataset,
    "memorization_b_dataset": _describe_memorization_b_dataset,
    "attention_only_model": _describe_attention_only,
    "numeric_transformer_model": _describe_numeric_transformer,
    "transformer_token_model": _describe_transformer_token,
    "transformer_multi_token_model": _describe_transformer_multi_token,
    "mlp_model": _describe_mlp,
    "kan_model": _describe_kan,
    "residual_ln_model": _describe_residual_ln,
    "adam_optimizer": _describe_adam,
    "adamw_optimizer": _describe_adamw,
    "sgd_optimizer": _describe_sgd,
    "signsgd_optimizer": _describe_signsgd,
    "shampoo_optimizer": _describe_shampoo,
    "soap_optimizer": _describe_soap,
    "cross_entropy_loss": _describe_ce_loss,
    "mse_loss": _describe_mse_loss,
    "trainer": _describe_trainer,
    "observable_accuracy": _describe_observable_accuracy,
    "observable_weight_l2": _describe_obs_weight_l2,
    "observable_weight_l1": _describe_obs_weight_l1,
    "kan_reg": _describe_kan_reg,
    "observable_embedding_trajectory": _describe_obs_emb_traj,
    "training_visualization": _describe_training_viz,
    "visualize_kan": _describe_visualize_kan,
    "observable_viz": _describe_observable_viz,
    "tensor_selector": _describe_tensor_selector,
    "effective_rank": _describe_effective_rank,
    "series_endpoint_gap": _describe_series_endpoint_gap,
    "model_weight_tensors": _describe_model_weight_tensors,
    "basic_calculator": _describe_basic_calculator,
}


def _describe_node(n: dict[str, Any], nodes: dict[str, dict[str, Any]], edges: list[dict[str, Any]]) -> str:
    t = str(n.get("type", ""))
    if t == "observable_user":
        return _describe_obs_user(n, nodes, edges)
    fn = _DESCRIBERS.get(t)
    if fn:
        return fn(n)
    if str(t).startswith("tensor_viz"):
        return f"A **{t}** view shows tensor values from this run (layout depends on tensor rank)."
    return _describe_generic(n)


def _trainer_attachments(
    trainer_id: str, edges: list[dict[str, Any]], nodes: dict[str, dict[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    by_role: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for e in edges:
        if e.get("target") != trainer_id:
            continue
        h = e.get("targetHandle") or "default"
        src = e.get("source")
        if not src or src not in nodes:
            continue
        by_role[str(h)].append(nodes[src])
    return by_role


_RE_RANDOM_NODE_ID_TAIL = re.compile(r"^(.+)-([a-z0-9]{6,14})$", flags=re.I)


def _strip_research_node_id_suffix(key: str) -> str:
    """Drop trailing ``-abc12def`` when it matches a typical RF node id tail (base36 random segment)."""
    k = str(key).strip()
    m = _RE_RANDOM_NODE_ID_TAIL.match(k)
    if not m:
        return k
    left, right = m.group(1), m.group(2)
    if not re.fullmatch(r"[a-z0-9]+", right, flags=re.I):
        return k
    return left


_OMH_OBSERVABLE_TYPE_LABEL: dict[str, str] = {
    "observable_weight_l2": "Weight L2",
    "observable_weight_l1": "Weight L1",
    "observable_accuracy": "Accuracy",
    "observable_capacity": "Capacity",
    "observable_gradient_norm": "Gradient norm",
    "observable_activation_stats": "Activation stats",
    "observable_hessian_eigenvalues": "Hessian eigenvalues",
    "observable_relu_nonlinear_count": "ReLU nonlinear count",
    "observable_embedding_trajectory": "Embedding trajectory",
    "observable_embedding_evolution": "Embedding evolution",
    "observable_train_test_gap": "Train / test gap",
    "observable_embedding_effective_rank": "Embedding effective rank",
    "observable_embedding_feature_drift": "Embedding feature drift",
    "observable_sink_attention_mass": "Sink attention mass",
    "observable_attention_entropy_mean": "Attention entropy (mean)",
    "observable_attention_max_weight_mean": "Attention max weight (mean)",
    "observable_attention_head_sink_max": "Attention head sink max",
    "observable_attention_position_bias_ratio": "Attention position bias ratio",
    "observable_activation_norm_mean": "Activation norm (mean)",
    "observable_activation_outlier_ratio": "Activation outlier ratio",
    "kan_reg": "KAN regularization",
}


_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _is_valid_png_bytes(b: bytes | bytearray | None) -> bool:
    if not b or len(b) < len(_PNG_MAGIC):
        return False
    return bytes(b[: len(_PNG_MAGIC)]) == _PNG_MAGIC


def _build_sections(doc: dict[str, Any], meta: dict[str, Any]) -> list[tuple[str, list[str]]]:
    nodes = _nodes_map(doc)
    edges = _edges(doc)
    trainers = [n for n in nodes.values() if n.get("type") == "trainer"]

    sections: list[tuple[str, list[str]]] = []

    if not trainers:
        sections.append(
            (
                "Training setup",
                ["No trainer was found in this snapshot—it may still be under construction."],
            )
        )
        return sections

    for ti, tr in enumerate(trainers, start=1):
        tid = str(tr.get("id"))
        head = "Training setup" if len(trainers) == 1 else f"Training setup (run {ti} of {len(trainers)})"
        paras: list[str] = []
        att = _trainer_attachments(tid, edges, nodes)
        role_order = [
            ("dataset", "Dataset"),
            ("train_dataset", "Training data (legacy handle)"),
            ("test_dataset", "Test data (legacy handle)"),
            ("model", "Model"),
            ("optimizer", "Optimizer"),
            ("loss", "Loss"),
            ("observables", "Observables logged during training"),
        ]
        for _handle, title in role_order:
            attached = att.get(_handle, [])
            if not attached:
                continue
            _extend_role_paragraphs(
                paras,
                title,
                attached,
                nodes,
                edges,
                include_hyperparameters=False,
                include_library_blurb=False,
                use_narrative_descriptors=True,
            )
        _extend_role_paragraphs(
            paras,
            "Trainer",
            [tr],
            nodes,
            edges,
            include_hyperparameters=False,
            include_library_blurb=False,
            use_narrative_descriptors=True,
        )
        sections.append((head, paras))

    # Side-chain: nodes not directly attached to any trainer as dataset/model/loss/optimizer/obs
    trainer_ids = {str(t.get("id")) for t in trainers}
    attached_ids: set[str] = set()
    for tid in trainer_ids:
        for e in edges:
            if e.get("target") == tid and e.get("targetHandle") in (
                "dataset",
                "train_dataset",
                "test_dataset",
                "model",
                "optimizer",
                "loss",
                "observables",
            ):
                s = e.get("source")
                if s:
                    attached_ids.add(str(s))
            if e.get("source") == tid:
                attached_ids.add(str(e.get("target", "")))

    side = [n for n in nodes.values() if n.get("id") not in attached_ids and n.get("type") != "trainer"]
    if side:
        sp: list[str] = [
            "Beyond the core training bundle, the project also includes extra analysis and visualization nodes "
            "(tensor views, calculators, effective-rank probes, and similar helpers). In plain terms:"
        ]
        for n in sorted(side, key=lambda x: str(x.get("type"))):
            blurb = _describe_node(n, nodes, edges).strip()
            if blurb:
                sp.append(blurb)
        sections.append(("Analysis and visualization", sp))

    return sections


def _sections_to_lines(sections: list[tuple[str, list[str]]]) -> list[str]:
    lines: list[str] = []
    for title, paras in sections:
        lines.append(title.upper())
        lines.append("")
        for p in paras:
            lines.extend(textwrap.wrap(_pdf_safe_text(p), width=PDF_TEXT_WRAP_WIDTH, replace_whitespace=False))
            lines.append("")
    return lines


def _split_sections_before_analysis(
    sections: list[tuple[str, list[str]]],
) -> tuple[list[tuple[str, list[str]]], list[tuple[str, list[str]]]]:
    """First block = intro + training setup; second = analysis / side chains (figures go between)."""
    pre: list[tuple[str, list[str]]] = []
    post: list[tuple[str, list[str]]] = []
    for name, paras in sections:
        nlow = str(name).strip().lower()
        if nlow.startswith("analysis and visualization"):
            post.append((name, paras))
        elif post:
            post.append((name, paras))
        else:
            pre.append((name, paras))
    return pre, post


def _paginate(lines: list[str], chars_per_line: int, lines_per_page: int) -> list[list[str]]:
    wrapped: list[str] = []
    for line in lines:
        if len(line) <= chars_per_line:
            wrapped.append(line)
        else:
            wrapped.extend(textwrap.wrap(line, width=chars_per_line, break_long_words=True, replace_whitespace=False))
    pages: list[list[str]] = []
    buf: list[str] = []
    for line in wrapped:
        if len(buf) >= lines_per_page:
            pages.append(buf)
            buf = []
        buf.append(line)
    if buf:
        pages.append(buf)
    return pages


def _figure_face_white(fig: plt.Figure) -> None:
    """Publication-style charts (avoid dark UI–like figure chrome in the PDF)."""
    fig.patch.set_facecolor("white")


def _ax_apply_single_point_xy_margins(ax: plt.Axes, x: float, y: float) -> None:
    """Expand limits so a lone (step, value) marker remains visible."""
    xpad = max(abs(x) * 0.02, 1.0) if x != 0 else 1.0
    ypad = max(abs(y) * 0.08, 1e-12) if y != 0 else 0.05
    ax.set_xlim(x - xpad, x + xpad)
    ax.set_ylim(y - ypad, y + ypad)


def _plot_loss_pair(
    title: str,
    step_ticks: list[float] | None,
    loss_hist: list[float] | None,
    test_loss: list[float] | None,
    y_metric: str,
    ax: plt.Axes,
    *,
    title_fontsize: int = 11,
    font_scale: float = 1.0,
) -> None:
    def fs(value: float) -> float:
        return round(float(value) * float(font_scale), 4)

    if not step_ticks or not loss_hist or len(step_ticks) != len(loss_hist):
        ax.text(
            0.5,
            0.5,
            "No matching step/loss arrays to plot.",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=fs(11),
        )
        ax.set_axis_off()
        return

    ax.set_facecolor("white")
    steps = np.asarray(step_ticks, dtype=float)
    train = np.asarray(loss_hist, dtype=float)
    if y_metric == "perplexity":
        train = np.exp(np.clip(train, -20, 20))

    has_test = bool(test_loss and len(test_loss) == len(step_ticks))
    test = np.asarray(test_loss, dtype=float) if has_test else None
    if test is not None and y_metric == "perplexity":
        test = np.exp(np.clip(test, -20, 20))

    single = len(steps) == 1
    use_markers = len(steps) < 50
    if single:
        ax.plot(steps, train, marker="o", linestyle="none", markersize=9, label="train", color="C0")
        _ax_apply_single_point_xy_margins(ax, float(steps[0]), float(train[0]))
    else:
        ax.plot(
            steps,
            train,
            label="train",
            color="C0",
            linewidth=1.2,
            marker="o" if use_markers else None,
            markersize=4.6 if use_markers else None,
        )

    if test is not None:
        if single:
            ax.plot(steps, test, marker="s", linestyle="none", markersize=8, label="test", color="C1", alpha=0.85)
        else:
            ax.plot(
                steps,
                test,
                label="test",
                color="C1",
                linewidth=1.2,
                alpha=0.85,
                marker="s" if use_markers else None,
                markersize=4.2 if use_markers else None,
            )

    y_label = "perplexity" if y_metric == "perplexity" else "loss"
    ax.set_xlabel("step", fontsize=fs(11))
    ax.set_ylabel(y_label, fontsize=fs(11))
    ax.tick_params(axis="both", which="major", labelsize=fs(10))
    ax.set_title(title, fontsize=fs(title_fontsize))
    ax.legend(loc="best", fontsize=max(5.0, fs(8)))
    ax.grid(True, alpha=0.25)


def _plot_observable_train_test(
    title: str,
    step_ticks: list[float] | None,
    train_vals: list[float] | None,
    test_vals: list[float] | None,
    ax: plt.Axes,
    *,
    title_fontsize: int = 11,
    font_scale: float = 1.0,
) -> None:
    """Plot a training history and its optional test history on one axis."""

    def fs(value: float) -> float:
        return round(float(value) * float(font_scale), 4)

    if not step_ticks or not train_vals or len(step_ticks) != len(train_vals):
        ax.text(
            0.5,
            0.5,
            "No matching history to plot.",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=fs(11),
        )
        ax.set_axis_off()
        return

    ax.set_facecolor("white")
    steps = np.asarray(step_ticks, dtype=float)
    train = np.asarray(train_vals, dtype=float)
    has_test = test_vals is not None and len(test_vals) == len(step_ticks)
    test = np.asarray(test_vals, dtype=float) if has_test else None
    single = len(steps) == 1
    use_markers = len(steps) < 50

    if single:
        ax.plot(steps, train, marker="o", linestyle="none", markersize=9, label="train", color="C0")
        _ax_apply_single_point_xy_margins(ax, float(steps[0]), float(train[0]))
    else:
        ax.plot(
            steps,
            train,
            label="train",
            color="C0",
            linewidth=1.2,
            marker="o" if use_markers else None,
            markersize=4.6 if use_markers else None,
        )

    if test is not None:
        if single:
            ax.plot(steps, test, marker="s", linestyle="none", markersize=8, label="test", color="C1", alpha=0.85)
        else:
            ax.plot(
                steps,
                test,
                label="test",
                color="C1",
                linewidth=1.2,
                alpha=0.85,
                marker="s" if use_markers else None,
                markersize=4.2 if use_markers else None,
            )

    title_lower = (title or "").lower()
    y_label = "acc" if "acc" in title_lower else ("loss" if "loss" in title_lower else "value")
    ax.set_xlabel("step", fontsize=fs(11))
    ax.set_ylabel(y_label, fontsize=fs(11))
    ax.tick_params(axis="both", which="major", labelsize=fs(10))
    ax.set_title(title, fontsize=fs(title_fontsize))
    ax.legend(loc="best", fontsize=max(5.0, fs(8)))
    ax.grid(True, alpha=0.25)

def _trainer_metric_plot_title(raw_key: str) -> str:
    """Readable label from trainer `observableMetricHistories` keys (no raw node ids)."""
    k = _strip_research_node_id_suffix(str(raw_key).strip())
    m = re.match(r"^obs-[a-z0-9]+:(.+)$", k, flags=re.I)
    if m:
        tail = str(m.group(1)).strip().replace("_", " ")
        return tail.title() if tail else "Observable"
    if re.fullmatch(r"obs-[a-z0-9]+", k, flags=re.I):
        return "Observable"
    return k.replace("_", " ") or "Observable"


def _title_preference_score(title: str) -> int:
    """Higher = better for choosing among duplicate series (same y vs step)."""
    t = (title or "").strip().lower()
    if not t:
        return 0
    if t.startswith("observable") or t.startswith("obs-"):
        return 0
    if "training observable" in t:
        return 2
    return 8


def _fp_series(steps: list[Any], values: list[Any]) -> int | None:
    """Stable fingerprint for deduplicating identical (step, value) curves."""
    if not isinstance(steps, list) or not isinstance(values, list) or len(steps) != len(values) or len(steps) == 0:
        return None
    try:
        s = np.asarray(steps, dtype=float)
        v = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        return None
    stride = max(1, len(s) // 24)
    s_s = tuple(np.round(s[::stride], 5).tolist())
    v_s = tuple(np.round(v[::stride], 5).tolist())
    return hash((len(s), s_s, v_s))


def _fp_loss(
    steps: list[Any],
    loss: list[Any],
    test_loss: list[Any] | None,
) -> int | None:
    if not isinstance(steps, list) or not isinstance(loss, list) or len(steps) != len(loss) or len(steps) == 0:
        return None
    try:
        s = np.asarray(steps, dtype=float)
        y = np.asarray(loss, dtype=float)
    except (TypeError, ValueError):
        return None
    tl = test_loss
    if isinstance(tl, list) and len(tl) == len(steps):
        try:
            yt = np.asarray(tl, dtype=float)
        except (TypeError, ValueError):
            yt = None
    else:
        yt = None
    stride = max(1, len(s) // 24)
    s_s = tuple(np.round(s[::stride], 5).tolist())
    y_s = tuple(np.round(y[::stride], 5).tolist())
    y_t = tuple(np.round(yt[::stride], 5).tolist()) if yt is not None and yt.size == s.size else ()
    return hash((len(s), s_s, y_s, y_t))


def _loss_figure_caption(*, y_metric: str, panel_index: int, n_panels: int) -> str:
    m = "perplexity (exp of loss)" if y_metric == "perplexity" else "loss"
    if n_panels > 1:
        return f"Train and test {m} vs. step ({panel_index} of {n_panels})"
    return f"Train and test {m} vs. step"


def _collect_paired_observable_rows(
    nodes: dict[str, Any],
    tr_nodes: list[dict[str, Any]],
) -> list[tuple[str, list[float], list[float], list[float] | None]]:
    """Observable panels: train + optional test on shared steps (``::test`` keys on trainer; ``testValueHistory`` on viz)."""
    scored: list[tuple[str, list[float], list[float], list[float] | None, int]] = []
    ov_nodes = [n for n in nodes.values() if n.get("type") == "observable_viz"]
    for n in ov_nodes:
        d = _d(n)
        steps, vals = d.get("stepTicks"), d.get("valueHistory")
        if not isinstance(steps, list) or not isinstance(vals, list) or len(steps) != len(vals):
            continue
        st_f = [float(x) for x in steps]
        tr_f = [float(x) for x in vals]
        tv_raw = d.get("testValueHistory")
        te_f: list[float] | None = None
        if isinstance(tv_raw, list) and len(tv_raw) == len(st_f) and len(tv_raw) >= 2:
            te_f = [float(x) for x in tv_raw]
        raw_name = str(d.get("observableName") or "Observable").strip() or "Observable"
        variant = str(d.get("vizVariant") or "").strip().lower()
        if te_f is not None or variant == "accuracy" or "accuracy" in raw_name.lower() or raw_name.strip().lower() in (
            "acc",
            "top-1",
            "top1",
        ):
            title = "Accuracy"
        else:
            title = raw_name[0].upper() + raw_name[1:] if raw_name else "Observable"
        sc = (28 if te_f else 0) + _title_preference_score(title) + 8
        scored.append((title, st_f, tr_f, te_f, sc))

    n_tr = len(tr_nodes)
    for tix, n in enumerate(tr_nodes, start=1):
        d = _d(n)
        steps_raw = d.get("stepTicks")
        om = d.get("observableMetricHistories")
        if not isinstance(om, dict):
            continue
        tsuf = f" (trainer {tix})" if n_tr > 1 else ""
        for obs_key, series in om.items():
            if not isinstance(obs_key, str) or obs_key.endswith("::test"):
                continue
            if not isinstance(series, list) or not series:
                continue
            st = steps_raw if isinstance(steps_raw, list) and len(steps_raw) == len(series) else list(range(len(series)))
            st_f = [float(x) for x in st]
            tr_f = [float(x) for x in series]
            tk = f"{obs_key}::test"
            raw_test = om.get(tk)
            te_f: list[float] | None = None
            if isinstance(raw_test, list) and len(raw_test) == len(st_f) and len(raw_test) >= 2:
                te_f = [float(x) for x in raw_test]
            base_title = _trainer_metric_plot_title(str(obs_key))
            kl = str(obs_key).lower()
            if "accuracy" in kl:
                disp = f"Accuracy{tsuf}"
            else:
                disp = f"{base_title}{tsuf}"
            sc = (22 if te_f else 0) + _title_preference_score(disp)
            scored.append((disp, st_f, tr_f, te_f, sc))

    best: dict[int, tuple[str, list[float], list[float], list[float] | None, int]] = {}
    for title, st_f, tr_f, te_f, sc in scored:
        fp = _fp_series(st_f, tr_f)
        if fp is None:
            continue
        prev = best.get(fp)
        if prev is None or sc > prev[4]:
            best[fp] = (title, st_f, tr_f, te_f, sc)
    out = [(t, s, tr, te) for t, s, tr, te, _ in best.values()]
    out.sort(key=lambda row: row[0].lower())
    return out


def _node_position_xy(n: dict[str, Any]) -> tuple[float, float] | None:
    p = n.get("position")
    if not isinstance(p, dict):
        return None
    try:
        return (float(p.get("x", 0.0)), float(p.get("y", 0.0)))
    except (TypeError, ValueError):
        return None


def _topology_node_label(n: dict[str, Any]) -> str:
    d = _d(n)
    raw = d.get("instanceTitle")
    if isinstance(raw, str) and raw.strip():
        s = raw.strip()
    else:
        t = str(n.get("type") or "node")
        s = t.replace("_", " ").strip().title()
    if len(s) > 46:
        return s[:43] + "..."
    return s


def _canvas_node_sort_x(n: dict[str, Any]) -> float:
    p = _node_position_xy(n)
    if p is None:
        return 0.0
    try:
        return float(p[0])
    except (TypeError, ValueError):
        return 0.0


def _decode_plot_png_dataurl_bytes(raw_val: Any) -> bytes | None:
    if not isinstance(raw_val, str):
        return None
    s = raw_val.strip()
    if not s:
        return None
    if "," in s:
        s = s.split(",")[-1].strip()
    try:
        out = base64.b64decode(s, validate=True)
    except Exception:
        try:
            out = base64.b64decode(s, validate=False)
        except Exception:
            return None
    return out if _is_valid_png_bytes(out) else None


def _observable_viz_panel_title(n: dict[str, Any]) -> str:
    d = _d(n)
    raw_name = str(d.get("observableName") or "Observable").strip() or "Observable"
    variant = str(d.get("vizVariant") or "").strip().lower()
    if variant == "accuracy" or "accuracy" in raw_name.lower() or raw_name.strip().lower() in (
        "acc",
        "top-1",
        "top1",
    ):
        return "Accuracy"
    return raw_name[0].upper() + raw_name[1:] if raw_name else "Observable"


def _collect_base_dashboard_row_dicts(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """Row descriptors for the base-canvas training strip (same ordering as the matplotlib composite)."""
    nodes = _nodes_map(doc)
    tr_nodes = [n for n in nodes.values() if n.get("type") == "trainer"]
    tv_nodes = [
        n for n in nodes.values() if n.get("type") == "training_visualization" and n.get("hidden") is not True
    ]
    ov_nodes = [n for n in nodes.values() if n.get("type") == "observable_viz" and n.get("hidden") is not True]

    rows: list[dict[str, Any]] = []
    seen_loss_fp: set[int] = set()
    seen_obs_fp: set[int] = set()

    for n in sorted(tv_nodes, key=_canvas_node_sort_x):
        d = _d(n)
        steps, loss, test_loss = d.get("stepTicks"), d.get("lossHistory"), d.get("testLossHistory")
        metric = str(d.get("yPlotMetric") or "loss")
        fp = _fp_loss(steps, loss, test_loss)
        xk = _canvas_node_sort_x(n)
        plot_title = str(d.get("plotTitle") or "").strip()
        if fp is not None and fp not in seen_loss_fp:
            seen_loss_fp.add(fp)
            st = [float(x) for x in steps]
            lo = [float(x) for x in loss]
            te = (
                [float(x) for x in test_loss]
                if isinstance(test_loss, list) and len(test_loss) == len(steps)
                else None
            )
            ttl = plot_title or _loss_figure_caption(y_metric=metric, panel_index=1, n_panels=1)
            rows.append(
                {
                    "x": xk,
                    "kind": "loss_series",
                    "title": ttl,
                    "steps": st,
                    "train": lo,
                    "test": te,
                    "metric": metric,
                    "source_node_id": str(n.get("id") or ""),
                }
            )
            continue
        raw_png = _decode_plot_png_dataurl_bytes(d.get("plotPngBase64"))
        if raw_png is not None:
            ttl = plot_title or "Training"
            rows.append(
                {
                    "x": xk,
                    "kind": "raster",
                    "title": ttl,
                    "png_bytes": raw_png,
                    "source_node_id": str(n.get("id") or ""),
                }
            )

    if not any(r["kind"] == "loss_series" for r in rows):
        for n in sorted(tr_nodes, key=_canvas_node_sort_x):
            d = _d(n)
            steps, loss, test_loss = d.get("stepTicks"), d.get("lossHistory"), d.get("testLossHistory")
            fp = _fp_loss(steps, loss, test_loss)
            if fp is None or fp in seen_loss_fp:
                continue
            seen_loss_fp.add(fp)
            st = [float(x) for x in steps]
            lo = [float(x) for x in loss]
            te = (
                [float(x) for x in test_loss]
                if isinstance(test_loss, list) and len(test_loss) == len(steps)
                else None
            )
            ttl = _loss_figure_caption(y_metric="loss", panel_index=1, n_panels=1)
            rows.append(
                {
                    "x": _canvas_node_sort_x(n) - 1e-6,
                    "kind": "loss_series",
                    "title": ttl,
                    "steps": st,
                    "train": lo,
                    "test": te,
                    "metric": "loss",
                    "source_node_id": str(n.get("id") or ""),
                }
            )
            break

    for n in sorted(ov_nodes, key=_canvas_node_sort_x):
        d = _d(n)
        steps, vals = d.get("stepTicks"), d.get("valueHistory")
        if not isinstance(steps, list) or not isinstance(vals, list) or len(steps) != len(vals) or len(steps) == 0:
            continue
        st_f = [float(x) for x in steps]
        tr_f = [float(x) for x in vals]
        fp = _fp_series(st_f, tr_f)
        if fp is None or fp in seen_obs_fp:
            continue
        seen_obs_fp.add(fp)
        tv_raw = d.get("testValueHistory")
        te_f: list[float] | None = None
        if isinstance(tv_raw, list) and len(tv_raw) == len(st_f) and len(tv_raw) >= 2:
            te_f = [float(x) for x in tv_raw]
        ttl = _observable_viz_panel_title(n)
        rows.append(
            {
                "x": _canvas_node_sort_x(n),
                "kind": "obs_series",
                "title": ttl,
                "steps": st_f,
                "train": tr_f,
                "test": te_f,
                "source_node_id": str(n.get("id") or ""),
            }
        )

    tail_x = 1_000_000.0
    for title, st_f, tr_f, te_f in _collect_paired_observable_rows(nodes, tr_nodes):
        fp = _fp_series(st_f, tr_f)
        if fp is None or fp in seen_obs_fp:
            continue
        seen_obs_fp.add(fp)
        rows.append(
            {
                "x": tail_x,
                "kind": "obs_series",
                "title": title,
                "steps": st_f,
                "train": tr_f,
                "test": te_f,
                "source_node_id": "",
            }
        )
        tail_x += 1.0

    rows.sort(key=lambda r: (float(r["x"]), str(r.get("title", "")).lower()))
    return rows


_UPSTREAM_WALK_STOP_TYPES = frozenset(
    {
        "token_prediction_dataset",
        "linear_dataset",
        "symbolic_func_dataset",
        "teacher_dataset",
        "random_input_distribution",
        "attention_only_model",
        "numeric_transformer_model",
        "transformer_token_model",
        "transformer_multi_token_model",
        "mlp_model",
        "kan_model",
        "residual_ln_model",
        "adam_optimizer",
        "adamw_optimizer",
        "sgd_optimizer",
        "signsgd_optimizer",
        "muon_optimizer",
        "shampoo_optimizer",
        "soap_optimizer",
        "cross_entropy_loss",
        "mse_loss",
    }
)


def _md_display_math(latex: str) -> str:
    s = (latex or "").strip().replace("`", "'")
    if not s:
        return ""
    if s.startswith("$$") and s.endswith("$$"):
        return s
    if s.startswith("$") and s.endswith("$") and len(s) >= 2:
        s = s[1:-1].strip()
    return f"$${s}$$"


def _calc_symbol_from_handle(target_handle: str, default_index: int) -> tuple[int, str]:
    h = (target_handle or "").strip()
    m = re.search(r"x[_\-\s]*([0-9]+)", h, flags=re.I)
    if m:
        idx = int(m.group(1))
        return idx, f"x_{idx}"
    idx = max(1, int(default_index))
    return idx, f"x_{idx}"


def _upstream_series_label_until_viz(start_id: str, nodes: dict[str, dict[str, Any]], edges: list[dict[str, Any]]) -> str:
    """Walk upstream until we hit a clear plotted/logged series source."""
    cur = str(start_id or "").strip()
    seen: set[str] = set()
    for _ in range(16):
        if not cur or cur in seen:
            break
        seen.add(cur)
        n = nodes.get(cur)
        if not n:
            break
        t = str(n.get("type") or "")
        d = _d(n)
        if t == "observable_viz":
            nm = str(d.get("observableName") or "").strip()
            if nm:
                return f"{nm.replace('_', ' ')} training series"
            vv = str(d.get("vizVariant") or "").strip()
            if vv:
                return f"{vv.replace('_', ' ')} training series"
            return "the upstream logged training series"
        if t == "training_visualization":
            metric = str(d.get("yPlotMetric") or "loss").strip() or "loss"
            return "training loss series" if metric == "loss" else f"{metric} training series"
        if t.startswith("observable_"):
            short = _OMH_OBSERVABLE_TYPE_LABEL.get(t, t.replace("observable_", "").replace("_", " ").strip())
            return f"{short} training series"
        inc = _incoming(edges, cur)
        if not inc:
            break
        cur = str(inc[0].get("source") or "")
    return "the upstream logged training series"


def _tensor_slice_source_phrase(n: dict[str, Any], nodes: dict[str, dict[str, Any]], edges: list[dict[str, Any]]) -> str:
    d = _d(n)
    src_label = _upstream_series_label_until_viz(str(n.get("id") or ""), nodes, edges)
    slices = d.get("slices")
    if not isinstance(slices, list) or not slices:
        return f"a scalar selected from {src_label}"
    first = slices[0]
    if isinstance(first, dict):
        dim = first.get("dimension")
        idx = str(first.get("indices", "")).strip()
        if str(dim) in ("0", "0.0") and idx:
            if idx.lower() in ("-1", "last"):
                return f"the last value of {src_label}"
            if re.fullmatch(r"-?\d+", idx):
                nidx = int(idx)
                if nidx >= 0:
                    suf = "th"
                    if nidx % 10 == 1 and nidx % 100 != 11:
                        suf = "st"
                    elif nidx % 10 == 2 and nidx % 100 != 12:
                        suf = "nd"
                    elif nidx % 10 == 3 and nidx % 100 != 13:
                        suf = "rd"
                    return f"the {nidx}{suf} value of {src_label}"
            return f"value(s) at index {idx} from {src_label}"
    return f"a scalar selected from {src_label}"


def _calculator_input_source_phrase(
    src_id: str,
    nodes: dict[str, dict[str, Any]],
    edges: list[dict[str, Any]],
) -> str:
    n = nodes.get(src_id)
    if not n:
        return "an upstream scalar source"
    t = str(n.get("type") or "")
    if t == "tensor_slicing":
        return _tensor_slice_source_phrase(n, nodes, edges)
    if t == "series_endpoint_gap":
        src_label = _upstream_series_label_until_viz(src_id, nodes, edges)
        return f"the endpoint gap (last minus first) of {src_label}"
    if t == "effective_rank":
        src_label = _upstream_series_label_until_viz(src_id, nodes, edges)
        return f"effective-rank scalar derived from {src_label}"
    return f"the scalar output of {_short_node_label(n)}"


def _basic_calculator_input_bindings_md(
    calc_node_id: str,
    nodes: dict[str, dict[str, Any]],
    edges: list[dict[str, Any]],
) -> str:
    inc = _incoming(edges, calc_node_id)
    if not inc:
        return ""
    rows: list[tuple[int, str, str]] = []
    for i, e in enumerate(inc, start=1):
        src = str(e.get("source") or "").strip()
        if not src:
            continue
        idx, sym = _calc_symbol_from_handle(str(e.get("targetHandle") or ""), i)
        rows.append((idx, sym, _calculator_input_source_phrase(src, nodes, edges)))
    if not rows:
        return ""
    rows.sort(key=lambda r: (r[0], r[1]))
    defs = "; ".join(f"$${sym}$$ = {phrase}" for _idx, sym, phrase in rows[:8])
    return f"Input mapping: {defs}."


def _upstream_metric_math_fragments(source_node_id: str, nodes: dict[str, dict[str, Any]], edges: list[dict[str, Any]]) -> str:
    """Plain-language + optional LaTeX for analysis nodes upstream of a viz sink."""
    if not source_node_id or source_node_id not in nodes:
        return ""
    frags: list[str] = []
    cur = source_node_id
    seen: set[str] = set()
    for _ in range(22):
        if cur in seen:
            break
        seen.add(cur)
        inc = _incoming(edges, cur)
        if not inc:
            break
        e = inc[0]
        src = str(e.get("source", ""))
        sn = nodes.get(src)
        if not sn:
            break
        t = str(sn.get("type") or "")
        if t == "series_endpoint_gap":
            frags.append(
                "Upstream, a **series endpoint gap** forms **$\\Delta=x_{\\mathrm{last}}-x_{0}$**: the last entry minus the first entry "
                "of a 1D logged series."
            )
        elif t == "tensor_slicing":
            d = _d(sn)
            slices = d.get("slices")
            parts: list[str] = []
            if isinstance(slices, list):
                for sl in slices[:6]:
                    if not isinstance(sl, dict):
                        continue
                    dim = sl.get("dimension", "?")
                    ix = str(sl.get("indices", "")).strip() or "?"
                    parts.append(f"axis **{dim}** at indices **{ix}**")
            if parts:
                frags.append(
                    "A **tensor slice** step reduces the upstream tensor by fixing " + ", ".join(parts) + ", producing scalars for the next stage."
                )
            else:
                frags.append(
                    "A **tensor slice** step picks scalar entries from the upstream tensor along chosen axes before further processing."
                )
        elif t == "basic_calculator":
            d = _d(sn)
            latex = str(d.get("equationLatex") or "").strip().replace("`", "'")
            if latex and len(latex) <= 200:
                eq = _md_display_math(latex)
                binds = _basic_calculator_input_bindings_md(str(sn.get("id") or ""), nodes, edges)
                frags.append(
                    "A **basic calculator** combines those scalars with the rule:\n\n"
                    f"{eq}\n"
                )
                if binds:
                    frags.append(binds)
            else:
                frags.append(
                    "A **basic calculator** combines scalar inputs with a fixed algebraic recipe before logging."
                )
        elif t == "effective_rank":
            frags.append(
                "Upstream, **effective rank** turns the selected matrix into a single summary statistic at each log step."
            )
        elif t == "tensor_selector":
            frags.append(
                "A **tensor selector** stage chooses which internal activation is forwarded into the rest of the analysis chain."
            )
        if t in _UPSTREAM_WALK_STOP_TYPES:
            break
        cur = src
    if not frags:
        return ""
    return "\n\n".join(reversed(frags))


def _interpret_base_dashboard_row(
    row: dict[str, Any],
    nodes: dict[str, dict[str, Any]],
    edges: list[dict[str, Any]],
) -> str:
    kind = str(row.get("kind") or "")
    title = str(row.get("title") or "Panel").strip()
    if kind == "loss_series":
        metric = str(row.get("metric") or "loss")
        y_txt = "perplexity (exp of average loss)" if metric == "perplexity" else "training loss"
        base = f"**{title}** plots **{y_txt}** on the vertical axis versus **optimizer step** on the horizontal axis"
        tail = ", showing train and (when available) test curves."
        sid = str(row.get("source_node_id") or "")
        extra = _upstream_metric_math_fragments(sid, nodes, edges) if sid else ""
        if extra:
            return base + "; " + extra + tail
        return base + tail
    if kind == "obs_series":
        sid = str(row.get("source_node_id") or "")
        extra = _upstream_metric_math_fragments(sid, nodes, edges) if sid else ""
        base = f"**{title}** tracks a logged scalar (train and optional test) versus **optimizer step**."
        if extra:
            return base + " " + extra
        return base
    if kind == "raster":
        return f"**{title}** embeds a raster image captured from the training visualization (not a simple x/y curve)."
    return f"**{title}** summarizes an auxiliary signal saved during training."


def _base_training_composite_what_is_plotted_md(doc: dict[str, Any]) -> str:
    rows = _collect_base_dashboard_row_dicts(doc)
    if not rows:
        return ""
    nodes = _nodes_map(doc)
    edges = _edges(doc)
    parts: list[str] = [
        "**What this strip shows (left → right):** each panel is one signal we saved during training; "
        "the horizontal axis inside a panel is always the **training step** unless noted in the title."
    ]
    for i, r in enumerate(rows, start=1):
        parts.append(f"{i}. {_interpret_base_dashboard_row(r, nodes, edges)}")
    return "\n\n".join(parts)


def _build_base_canvas_composite_figure(
    doc: dict[str, Any],
    *,
    font_scale: float = 1.0,
) -> tuple[str, plt.Figure] | None:
    """One horizontal row of subplots: training viz (loss or raster), each ``observable_viz`` by canvas x, then orphan trainer series."""
    def fs(value: float) -> float:
        return round(float(value) * float(font_scale), 4)

    rows = _collect_base_dashboard_row_dicts(doc)
    if not rows:
        return None

    n_cols = len(rows)
    fig_w = min(36.0, 4.25 * n_cols + 1.25)
    fig_h = 4.05
    fig, axes = plt.subplots(1, n_cols, figsize=(fig_w, fig_h), squeeze=False, facecolor="white")
    _figure_face_white(fig)
    ax_row = axes[0]
    tfs = 8 if n_cols >= 6 else 10
    for i, r in enumerate(rows):
        ax = ax_row[i] if n_cols > 1 else ax_row[0]
        if r["kind"] == "loss_series":
            _plot_loss_pair(
                r["title"],
                r["steps"],
                r["train"],
                r["test"],
                r["metric"],
                ax,
                title_fontsize=tfs,
                font_scale=font_scale,
            )
        elif r["kind"] == "obs_series":
            _plot_observable_train_test(
                r["title"],
                r["steps"],
                r["train"],
                r["test"],
                ax,
                title_fontsize=tfs,
                font_scale=font_scale,
            )
        else:
            im = plt.imread(io.BytesIO(r["png_bytes"]))
            ax.imshow(im)
            ax.set_axis_off()
            ax.set_title(r["title"], fontsize=fs(tfs))

    fig.suptitle("Base experiment — training and observables", fontsize=fs(11.5), y=1.02)
    fig.tight_layout()
    return ("Base experiment — training and observables", fig)


def _collect_figure_pages(doc: dict[str, Any], *, font_scale: float = 1.0) -> list[tuple[str, plt.Figure]]:
    built = _build_base_canvas_composite_figure(doc, font_scale=font_scale)
    if built is None:
        return []
    _cap, fig = built
    return [(_cap, fig)]


def _write_paginated_text_pages(
    pdf: PdfPages,
    pages: list[list[str]],
    doc_title: str,
    *,
    first_page_number: int = 1,
    total_document_pages: int | None = None,
) -> int:
    """Render pre-paginated text (one list[str] per sheet). Returns page count."""
    n_here = len(pages)
    total = total_document_pages if total_document_pages is not None else n_here
    for i, page_lines in enumerate(pages):
        letter = plt.figure(figsize=(LETTER_W_IN, LETTER_H_IN), facecolor="white")
        _figure_face_white(letter)
        pnum = first_page_number + i
        _letter_header_and_page_line(letter, doc_title=doc_title, page_number=pnum, total_pages=total)
        _letter_draw_body_lines(letter, page_lines, y_top=PDF_TEXT_Y_TOP)
        _pdf_savefig(pdf, letter)
        plt.close(letter)
    return n_here


def _write_text_pages(
    pdf: PdfPages,
    lines: list[str],
    doc_title: str,
    chars: int | None = None,
    per_page: int | None = None,
    *,
    first_page_number: int = 1,
    total_document_pages: int | None = None,
) -> int:
    """Render text into ``pdf``. Returns the number of pages written."""
    cw = PDF_TEXT_WRAP_WIDTH if chars is None else chars
    lp = PDF_TEXT_LINES_PER_PAGE if per_page is None else per_page
    pages = _paginate(lines, cw, lp)
    return _write_paginated_text_pages(
        pdf,
        pages,
        doc_title,
        first_page_number=first_page_number,
        total_document_pages=total_document_pages,
    )


def write_experiment_pdf(doc: dict[str, Any], meta: dict[str, Any], output: Path, title: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    sections = _build_sections(doc, meta)
    fig_pages = _collect_figure_pages(doc)
    prose_plain = _strip_light_markdown_for_pdf(_base_training_composite_what_is_plotted_md(doc).strip())
    if prose_plain:
        fig_pages = [
            (f"{_strip_light_markdown_for_pdf(str(cap).strip())}\n\n{prose_plain}".strip(), fig) for cap, fig in fig_pages
        ]

    if not fig_pages:
        lines = _sections_to_lines(sections)
        lines.extend(
            [
                "RESULTS AND PLOTS",
                "",
                "This JSON snapshot does not yet contain embedded training curves (step ticks + loss history on the "
                "trainer or training visualization, or per-step observable histories). "
                "Train inside ComfyResearch with the graph open so the panels fill in, then save or export again so "
                "this report can auto-build the figures.",
            ]
        )
        with PdfPages(output) as pdf:
            _write_text_pages(pdf, lines, doc_title=title)
        return

    pre_secs, post_secs = _split_sections_before_analysis(sections)
    lines_pre = _sections_to_lines(pre_secs)
    lines_post = _sections_to_lines(post_secs) if post_secs else []
    pages_pre_list = _paginate(lines_pre, PDF_TEXT_WRAP_WIDTH, PDF_TEXT_LINES_PER_PAGE)
    pages_post_n = len(_paginate(lines_post, PDF_TEXT_WRAP_WIDTH, PDF_TEXT_LINES_PER_PAGE)) if lines_post else 0
    merge_first = bool(fig_pages) and _should_merge_last_text_page_with_first_figure(
        pages_pre_list, lines_per_page=PDF_TEXT_LINES_PER_PAGE
    )
    total_pages = len(pages_pre_list) + len(fig_pages) + pages_post_n - (1 if merge_first else 0)

    with PdfPages(output) as pdf:
        cur = 1
        if merge_first:
            cap0, fig0 = fig_pages[0]
            rest_pages = pages_pre_list[:-1]
            if rest_pages:
                cur += _write_paginated_text_pages(
                    pdf,
                    rest_pages,
                    doc_title=title,
                    first_page_number=cur,
                    total_document_pages=total_pages,
                )
            _pdf_savefig_letter_text_plus_first_figure(
                pdf,
                pages_pre_list[-1],
                fig0,
                doc_title=title,
                page_number=cur,
                total_pages=total_pages,
                figure_caption=cap0,
            )
            cur += 1
            for cap, fig in fig_pages[1:]:
                _pdf_savefig_on_letter_page(
                    pdf,
                    fig,
                    doc_title=title,
                    page_number=cur,
                    total_pages=total_pages,
                    figure_caption=cap,
                )
                cur += 1
        else:
            cur += _write_text_pages(
                pdf,
                lines_pre,
                doc_title=title,
                first_page_number=cur,
                total_document_pages=total_pages,
            )
            for cap, fig in fig_pages:
                _pdf_savefig_on_letter_page(
                    pdf,
                    fig,
                    doc_title=title,
                    page_number=cur,
                    total_pages=total_pages,
                    figure_caption=cap,
                )
                cur += 1
        if lines_post:
            _write_text_pages(
                pdf,
                lines_post,
                doc_title=title,
                first_page_number=cur,
                total_document_pages=total_pages,
            )


def main() -> None:
    p = argparse.ArgumentParser(description="Graph JSON → natural-language experiment PDF (ComfyResearch).")
    p.add_argument("--input", "-i", type=Path, required=True)
    p.add_argument("--output", "-o", type=Path, required=True)
    p.add_argument("--template-id", type=str, default=None)
    p.add_argument("--template-name", type=str, default=None)
    p.add_argument("--title", type=str, default=None)
    args = p.parse_args()

    if not args.input.is_file():
        raise SystemExit(f"Input not found: {args.input}")

    raw = _load_json(args.input)
    if isinstance(raw, list) and not (args.template_id or args.template_name):
        raise SystemExit("Input is a template array: pass --template-id or --template-name.")

    doc, meta = _resolve_document(raw, args.template_id, args.template_name)
    title = args.title or meta.get("name") or meta.get("id") or args.input.stem
    write_experiment_pdf(doc, meta, args.output, title=title)
    print(f"Wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
