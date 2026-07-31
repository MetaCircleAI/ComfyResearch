#!/usr/bin/env python3
"""Repair truncated graph library JSON (e.g. interrupted writes).

Handles the common case: the file ends after a complete value plus a trailing
comma, with missing ``]`` / ``}`` closers. Builds a bracket stack while
ignoring ``{`` ``[`` ``"`` inside JSON strings, then strips trailing commas and
appends the required closers.

Does not recover unterminated strings, invalid mid-file syntax, or truncated
numbers/keywords mid-token.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


def _eof_inside_string(s: str) -> bool:
    in_string = False
    escape = False
    for c in s:
        if in_string:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_string = False
        else:
            if c == '"':
                in_string = True
    return in_string


def bracket_stack_outside_strings(s: str) -> list[str]:
    """Return closers still needed at end of *s* (each ``{`` -> ``}``, ``[`` -> ``]``)."""
    stack: list[str] = []
    in_string = False
    escape = False
    for c in s:
        if in_string:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_string = False
        else:
            if c == '"':
                in_string = True
            elif c == "{":
                stack.append("}")
            elif c == "[":
                stack.append("]")
            elif c in "}]":
                if not stack:
                    raise ValueError(f"Unbalanced JSON: extra closer {c!r}")
                want = stack.pop()
                if want != c:
                    raise ValueError(f"Unbalanced JSON: expected {want!r}, got {c!r}")
    return stack


def repair_truncated_json_text(text: str) -> tuple[str, dict]:
    """Return repaired text and a small report dict."""
    raw_len = len(text)
    s = text.rstrip("\n\r\t ")
    report: dict[str, object] = {
        "bytes_input": raw_len,
        "stripped_trailing_ws_chars": raw_len - len(s),
    }

    removed_commas = 0
    while True:
        t = s.rstrip("\n\r\t ")
        if not t.endswith(","):
            s = t
            break
        s = t[:-1].rstrip("\n\r\t ")
        removed_commas += 1
    report["removed_trailing_commas"] = removed_commas

    if not s:
        raise ValueError("Nothing left after stripping trailing whitespace/commas.")

    if _eof_inside_string(s):
        raise ValueError("Truncation appears inside an unterminated JSON string; auto-repair not supported.")

    stack = bracket_stack_outside_strings(s)
    closers = "".join(reversed(stack))
    report["synthesized_closer_chars"] = closers
    repaired = s + closers
    parsed = json.loads(repaired)
    report["json_top_level"] = type(parsed).__name__
    if isinstance(parsed, list):
        report["list_len"] = len(parsed)
    return repaired, report


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "path",
        nargs="?",
        default="data/graph_library/templates.json",
        help="Library JSON path (legacy array file workflows/assets; templates are usually per-file under data/graph_library/templates/)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate repair only; do not write or create a backup.",
    )
    p.add_argument(
        "--canonicalize",
        action="store_true",
        help="After repair, re-serialize with json.dumps(indent=2) (large rewrite; alters float text).",
    )
    args = p.parse_args()
    path = Path(args.path)
    if not path.is_file():
        print(f"Not a file: {path}", file=sys.stderr)
        return 1

    text = path.read_text(encoding="utf-8")
    try:
        repaired, report = repair_truncated_json_text(text)
    except (ValueError, json.JSONDecodeError) as e:
        print(f"Repair failed: {e}", file=sys.stderr)
        return 1

    data = json.loads(repaired)
    if isinstance(data, list):
        report["list_len"] = len(data)
    print("Repair report:", report)

    if args.dry_run:
        print("Dry run OK (no files written).")
        return 0

    bak = path.with_suffix(path.suffix + ".bak")
    shutil.copy2(path, bak)
    out = json.dumps(data, indent=2, ensure_ascii=False) + "\n" if args.canonicalize else repaired
    path.write_text(out, encoding="utf-8")
    print(f"Wrote {path} ({len(out)} bytes). Backup: {bak}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
