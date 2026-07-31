"""Stdin/stdout JSON protocol for a persistent notebook kernel (spawned by notebook_execute)."""

from __future__ import annotations

import base64
import contextlib
import io
import json
import sys
import traceback
from typing import Any

_NS: dict[str, Any] = {"__name__": "__main__"}

# Figures captured during the current cell (base64 PNG); set to [] before each exec.
_cell_images: list[str] | None = None
_mpl_show_hook_installed = False


def _capture_open_figures(into: list[str]) -> None:
    """Serialize all open matplotlib figures as PNG base64 and close them."""
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    for num in list(plt.get_fignums()):
        fig = plt.figure(num)
        if not fig.axes:
            plt.close(fig)
            continue
        buf = io.BytesIO()
        try:
            fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
        except Exception:
            plt.close(fig)
            continue
        into.append(base64.standard_b64encode(buf.getvalue()).decode("ascii"))
        plt.close(fig)


def _ensure_matplotlib_inline() -> None:
    """Use non-interactive backend and make plt.show() capture figures instead of opening a window."""
    global _mpl_show_hook_installed
    if _mpl_show_hook_installed:
        return
    try:
        import matplotlib

        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt
    except Exception:
        return

    _real_show = plt.show

    def _show(*args: Any, **kwargs: Any) -> Any:
        if _cell_images is not None:
            _capture_open_figures(_cell_images)
        return None

    plt.show = _show
    _mpl_show_hook_installed = True
    # Stash so user code can still access real show if needed
    _NS["__cr_plt_show_original"] = _real_show


def main() -> None:
    global _cell_images
    _ensure_matplotlib_inline()
    stdin = sys.stdin
    out = sys.stdout
    while True:
        line = stdin.readline()
        if line == "":
            break
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            out.write(
                json.dumps(
                    {
                        "ok": False,
                        "exit_code": 1,
                        "stdout": "",
                        "stderr": "kernel: invalid json input line\n",
                        "images": [],
                    }
                )
                + "\n"
            )
            out.flush()
            continue
        code = msg.get("code", "")
        if not isinstance(code, str):
            code = ""
        buf_out = io.StringIO()
        buf_err = io.StringIO()
        exit_code = 0
        images: list[str] = []
        _cell_images = images
        _ensure_matplotlib_inline()
        try:
            with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
                exec(compile(code or "pass", "<cell>", "exec"), _NS, _NS)
        except BaseException:
            exit_code = 1
            buf_err.write(traceback.format_exc())
        finally:
            # Jupyter-style: capture any figures left open without plt.show()
            try:
                _capture_open_figures(images)
            except Exception:
                pass
            _cell_images = None

        payload = {
            "ok": exit_code == 0,
            "exit_code": exit_code,
            "stdout": buf_out.getvalue(),
            "stderr": buf_err.getvalue(),
            "images": images,
        }
        out.write(json.dumps(payload) + "\n")
        out.flush()


if __name__ == "__main__":
    main()
