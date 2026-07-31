"""
ComfyResearch entrypoint: `python app.py` starts the API server (Uvicorn).

Venv (once): python3 -m venv comfy-research && source comfy-research/bin/activate && pip install -r requirements.txt
Dev UI: cd frontend && npm install && npm run dev  (proxies /api — set COMFYRESEARCH_PORT to match this server's port, e.g. COMFYRESEARCH_PORT=8042 npm run dev)
Browser: python app.py --open  (or COMFYRESEARCH_OPEN_BROWSER=1) opens http://127.0.0.1:<port>/ after startup.
"""

from __future__ import annotations

import argparse
import os
import threading
import webbrowser

import uvicorn


def _schedule_open_browser(host: str, port: int, delay_s: float = 1.5) -> None:
    """Open the default browser after the server has time to bind (best-effort)."""
    display_host = "127.0.0.1" if host in ("0.0.0.0", "::", "[::]") else host
    url = f"http://{display_host}:{port}/"

    def _open() -> None:
        webbrowser.open(url)

    threading.Timer(delay_s, _open).start()


def main() -> None:
    parser = argparse.ArgumentParser(description="ComfyResearch server")
    parser.add_argument("--host", default=os.environ.get("COMFYRESEARCH_HOST", "0.0.0.0"))
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("COMFYRESEARCH_PORT", "8000")),
    )
    env_reload = os.environ.get("COMFYRESEARCH_RELOAD", "").lower()
    default_reload = env_reload in ("1", "true", "yes") if env_reload else False
    parser.add_argument(
        "--reload",
        action=argparse.BooleanOptionalAction,
        default=default_reload,
        help="Auto-reload on code changes (default: off; set COMFYRESEARCH_RELOAD=1 or pass --reload)",
    )
    open_default = os.environ.get("COMFYRESEARCH_OPEN_BROWSER", "").lower() in (
        "1",
        "true",
        "yes",
    )
    parser.add_argument(
        "--open",
        action=argparse.BooleanOptionalAction,
        default=open_default,
        help="Open default browser to this app (default: on if COMFYRESEARCH_OPEN_BROWSER=1)",
    )
    args = parser.parse_args()

    if args.open:
        _schedule_open_browser(args.host, args.port)

    uvicorn.run(
        "comfy_research.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
