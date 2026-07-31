#!/usr/bin/env python3
"""
Fetch a blog URL and write a ComfyResearch SavedGraphEntry JSON.

By default the graph includes:

- **comment** nodes: page title, URL, and a plain-text excerpt
- A **runnable** baseline: ``linear_dataset`` → ``mlp_model`` → ``mse_loss`` /
  ``adam_optimizer`` → ``trainer`` → ``model_checkpoint`` + ``training_visualization``

Use ``--comments-only`` for excerpt comments only (no trainer wiring).

Examples::

    python scripts/blog_to_template.py --url https://example.com/my-post

    python scripts/blog_to_template.py --url https://example.com/my-post \\
        --slug attention-residual

    python scripts/blog_to_template.py --url https://example.com/x --comments-only
"""

from __future__ import annotations

import argparse
import json
import re
import textwrap
import time
import uuid
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_LIB = _REPO_ROOT / "data" / "graph_library"
_TEMPLATES_DIR = _DEFAULT_LIB / "templates"

_MAX_EXCERPT = 12_000
_TIER = "medium"


class _StripHTMLParser(HTMLParser):
    """Collect visible text; skip script/style; insert newlines for block-ish tags."""

    _BLOCK = frozenset(
        {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "pre", "blockquote"}
    )

    def __init__(self) -> None:
        super().__init__()
        self._chunks: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        t = tag.lower()
        if t in ("script", "style", "noscript"):
            self._skip += 1
        if self._skip:
            return
        if t in self._BLOCK:
            self._chunks.append("\n")
        if t == "br":
            self._chunks.append("\n")

    def handle_endtag(self, tag: str) -> None:
        t = tag.lower()
        if t in ("script", "style", "noscript") and self._skip:
            self._skip -= 1
        if self._skip:
            return
        if t in self._BLOCK:
            self._chunks.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip:
            return
        if data.strip():
            self._chunks.append(data)

    def text(self) -> str:
        raw = "".join(self._chunks)
        raw = re.sub(r"[ \t\r\f\v]+", " ", raw)
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        return raw.strip()


def _fetch_html(url: str, timeout: int) -> tuple[str, str | None]:
    """Return (html, final_url_after_redirects_or_None)."""
    req = Request(
        url,
        headers={
            "User-Agent": "ComfyResearch-blog_to_template/1.0 (+https://github.com) Python-urllib",
            "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
        },
        method="GET",
    )
    with urlopen(req, timeout=timeout) as resp:  # noqa: S310 — user-supplied URL is intentional
        charset = resp.headers.get_content_charset() or "utf-8"
        body = resp.read().decode(charset, errors="replace")
        final = getattr(resp, "geturl", lambda: None)()
        return body, final


def _title_from_html(html: str) -> str | None:
    m = re.search(r"<title[^>]*>([^<]{1,500})</title>", html, re.I | re.DOTALL)
    if m:
        return re.sub(r"\s+", " ", m.group(1)).strip()
    m = re.search(
        r'<meta[^>]+property=["\']og:title["\'][^>]+content=["\']([^"\']+)["\']',
        html,
        re.I,
    )
    if m:
        return m.group(1).strip()
    m = re.search(
        r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+property=["\']og:title["\']',
        html,
        re.I,
    )
    if m:
        return m.group(1).strip()
    return None


def _slug_from_url(url: str) -> str:
    from urllib.parse import urlparse

    path = urlparse(url).path.strip("/").split("/")[-1] or "import"
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", path).strip("-").lower()
    return slug[:80] if slug else "blog-import"


def _default_train_graph(slug: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Linear MSE regression demo wired like ``default_research_graph.json`` (subset)."""
    p = f"{slug}_"
    nodes: list[dict[str, Any]] = [
        {
            "id": f"{p}linear_ds",
            "type": "linear_dataset",
            "position": {"x": 40.0, "y": 40.0},
            "data": {
                "inputDim": 10,
                "outputDim": 1,
                "inputDistribution": "standard_normal",
                "outputDistribution": "deterministic",
                "trainSize": 800,
                "testSize": 200,
                "noiseLevel": 0,
                "seed": 42,
            },
        },
        {
            "id": f"{p}mlp",
            "type": "mlp_model",
            "position": {"x": 40.0, "y": 260.0},
            "data": {
                "inputDim": 10,
                "outputDim": 1,
                "depth": 2,
                "width": 64,
                "activation": "relu",
                "seed": 42,
            },
        },
        {
            "id": f"{p}adam",
            "type": "adam_optimizer",
            "position": {"x": 40.0, "y": 500.0},
            "data": {
                "learningRate": 0.001,
                "beta1": 0.9,
                "beta2": 0.999,
                "epsilon": 1e-8,
            },
        },
        {
            "id": f"{p}mse",
            "type": "mse_loss",
            "position": {"x": 320.0, "y": 320.0},
            "data": {"lossScale": 1},
        },
        {
            "id": f"{p}trainer",
            "type": "trainer",
            "position": {"x": 520.0, "y": 120.0},
            "data": {"trainingSteps": 1000, "logFrequency": 20},
        },
        {
            "id": f"{p}ckpt",
            "type": "model_checkpoint",
            "position": {"x": 780.0, "y": 80.0},
            "data": {
                "checkpoint_b64": "",
                "memoryCheckpoint_b64": "",
                "checkpointSource": "memory",
                "checkpointFileName": "",
            },
        },
        {
            "id": f"{p}tviz",
            "type": "training_visualization",
            "position": {"x": 860.0, "y": 220.0},
            "data": {},
        },
    ]
    edges: list[dict[str, Any]] = [
        {
            "id": f"{p}e_ds_tr",
            "source": f"{p}linear_ds",
            "target": f"{p}trainer",
            "sourceHandle": "dataset",
            "targetHandle": "dataset",
        },
        {
            "id": f"{p}e_mlp_tr",
            "source": f"{p}mlp",
            "target": f"{p}trainer",
            "sourceHandle": "model",
            "targetHandle": "model",
        },
        {
            "id": f"{p}e_adam_tr",
            "source": f"{p}adam",
            "target": f"{p}trainer",
            "sourceHandle": "optimizer",
            "targetHandle": "optimizer",
        },
        {
            "id": f"{p}e_mse_tr",
            "source": f"{p}mse",
            "target": f"{p}trainer",
            "sourceHandle": "loss",
            "targetHandle": "loss",
        },
        {
            "id": f"{p}e_tr_ckpt",
            "source": f"{p}trainer",
            "target": f"{p}ckpt",
            "sourceHandle": "checkpoint",
            "targetHandle": "model_checkpoint",
        },
        {
            "id": f"{p}e_tr_tviz",
            "source": f"{p}trainer",
            "target": f"{p}tviz",
            "sourceHandle": "loss_results",
            "targetHandle": "plot",
        },
    ]
    return nodes, edges


def _atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _build_entry(
    url: str,
    slug: str,
    html: str,
    final_url: str | None,
    name_override: str | None,
    *,
    full_graph: bool,
) -> dict[str, Any]:
    title = _title_from_html(html) or "Imported blog"
    parser = _StripHTMLParser()
    try:
        parser.feed(html)
        parser.close()
    except Exception:
        body_text = ""
    else:
        body_text = parser.text()

    if len(body_text) > _MAX_EXCERPT:
        body_text = body_text[: _MAX_EXCERPT].rstrip() + "\n\n… (truncated)"

    canonical = final_url or url
    excerpt = body_text if body_text else "(No body text extracted; edit this comment or add nodes.)"
    intro = f"# {title}\n\n**URL:** {canonical}\n\n---\n\n{excerpt}".strip()

    if full_graph:
        howto = textwrap.dedent(
            f"""\
            ### Default training graph (included)

            A runnable **linear → MLP → MSE → Adam → trainer → checkpoint → loss viz** chain is wired (ids prefixed with ``{slug}_``).

            Tune **inputDim / outputDim** on the dataset and MLP, **depth / width / activation**, **trainingSteps**, and **seed** to follow the blog. Add branches (e.g. second dataset, observables) in the app or by editing the generated template JSON.
            """
        ).strip()
    else:
        howto = textwrap.dedent(
            """\
            ### Next steps

            Generated with ``--comments-only`` (no trainer). Add nodes and edges in the app or edit the generated JSON manually.
            """
        ).strip()

    entry_id = str(uuid.uuid4())
    saved_at = float(int(time.time() * 1000))
    display_name = name_override or f"blog: {slug}"

    comment_nodes = [
        {
            "id": f"{slug}_blog_source",
            "type": "comment",
            "position": {"x": -120.0, "y": -280.0},
            "data": {"text": intro},
        },
        {
            "id": f"{slug}_blog_howto",
            "type": "comment",
            "position": {"x": 380.0, "y": -260.0},
            "data": {"text": howto},
        },
    ]
    if full_graph:
        train_nodes, train_edges = _default_train_graph(slug)
        all_nodes = comment_nodes + train_nodes
        all_edges = train_edges
    else:
        all_nodes = comment_nodes
        all_edges = []

    document = {
        "version": 1,
        "nodes": all_nodes,
        "edges": all_edges,
        "viewport": None,
    }

    return {
        "id": entry_id,
        "name": display_name,
        "tier": _TIER,
        "document": document,
        "savedAt": saved_at,
    }


def _write_template_library_entry(entry: dict[str, Any]) -> Path:
    """Write ``entry`` to ``data/graph_library/templates/<id>.json`` (same layout as the app API)."""
    eid = str(entry["id"])
    if "/" in eid or "\\" in eid or eid.startswith("."):
        raise SystemExit(f"Invalid entry id for filename: {eid!r}")
    _TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
    path = _TEMPLATES_DIR / f"{eid}.json"
    _atomic_write_json(path, entry)
    return path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", required=True, help="Blog or article URL to fetch.")
    p.add_argument(
        "--slug",
        help="Short id prefix for comment node ids and default filename (default: derived from URL path).",
    )
    p.add_argument(
        "--output",
        type=Path,
        help=f"Output JSON path (default: {_TEMPLATES_DIR}/<entry id>.json).",
    )
    p.add_argument(
        "--name",
        help="Template display name (default: blog: <slug>).",
    )
    p.add_argument(
        "--merge-templates",
        action="store_true",
        help=f"Also write this entry to {_TEMPLATES_DIR}/<id>.json (same as default output; kept for scripts).",
    )
    p.add_argument("--timeout", type=int, default=30, help="HTTP timeout seconds (default: 30).")
    p.add_argument(
        "--comments-only",
        action="store_true",
        help="Only comment nodes (URL + excerpt); no dataset/model/trainer graph.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print JSON to stdout only; do not write files.",
    )
    args = p.parse_args()

    slug = args.slug or _slug_from_url(args.url)

    try:
        html, final_url = _fetch_html(args.url, args.timeout)
    except HTTPError as e:
        raise SystemExit(f"HTTP error fetching URL: {e}") from e
    except URLError as e:
        raise SystemExit(f"Network error fetching URL: {e}") from e

    entry = _build_entry(
        args.url,
        slug,
        html,
        final_url,
        args.name,
        full_graph=not args.comments_only,
    )

    if args.dry_run:
        print(json.dumps(entry, indent=2, ensure_ascii=False))
        return

    lib_path = _TEMPLATES_DIR / f"{entry['id']}.json"
    if args.output is not None:
        _atomic_write_json(args.output, entry)
        print(f"Wrote {args.output}")
        if args.merge_templates and args.output.resolve() != lib_path.resolve():
            _write_template_library_entry(entry)
            print(f"Also wrote library copy {lib_path}")
    else:
        _write_template_library_entry(entry)
        print(f"Wrote {lib_path}")


if __name__ == "__main__":
    main()
