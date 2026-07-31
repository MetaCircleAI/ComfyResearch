"""Run notebook-style Python cells on the server with a persistent namespace per session."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api", tags=["notebook"])

_MAX_CODE_CHARS = 400_000
_MAX_STREAM_CHARS = 200_000
_MAX_SESSION_ID_CHARS = 256
_DEFAULT_TIMEOUT_S = 90
_MAX_NOTEBOOK_IMAGES = 16
_MAX_BASE64_PER_IMAGE = 1_200_000

_KERNEL_CHILD = Path(__file__).with_name("_notebook_kernel_child.py")


class NotebookExecuteBody(BaseModel):
    code: str = Field(..., max_length=_MAX_CODE_CHARS)
    session_id: str = Field(..., min_length=1, max_length=_MAX_SESSION_ID_CHARS)


class NotebookExecuteResult(BaseModel):
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False
    images: list[str] = Field(
        default_factory=list,
        description="Base64-encoded PNG strings from matplotlib (inline notebook figures).",
    )


def _truncate(s: str, max_chars: int) -> str:
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 80] + "\n… [truncated; output exceeded limit]\n"


def _sanitize_images(raw: object) -> list[str]:
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    for item in raw[:_MAX_NOTEBOOK_IMAGES]:
        if not isinstance(item, str) or not item.strip():
            continue
        s = item.strip()
        if len(s) > _MAX_BASE64_PER_IMAGE:
            s = s[: max(0, _MAX_BASE64_PER_IMAGE - 80)] + "\n… [truncated image]\n"
        out.append(s)
    return out


def _readline_timeout(proc: subprocess.Popen, timeout_s: float) -> str:
    """Read one line from proc.stdout, or kill proc on timeout."""
    out: list[str | BaseException] = []

    def _read() -> None:
        assert proc.stdout is not None
        try:
            out.append(proc.stdout.readline())
        except Exception as e:
            out.append(e)

    th = threading.Thread(target=_read, daemon=True)
    th.start()
    th.join(timeout_s)
    if th.is_alive():
        try:
            proc.kill()
        except OSError:
            pass
        raise TimeoutError from None
    if not out:
        return ""
    item = out[0]
    if isinstance(item, BaseException):
        raise item
    return item if isinstance(item, str) else ""


class NotebookKernelSession:
    """One long-lived Python subprocess; cells exec in a shared dict namespace."""

    def __init__(self) -> None:
        if not _KERNEL_CHILD.is_file():
            raise FileNotFoundError(f"Missing kernel child script: {_KERNEL_CHILD}")
        self.proc = subprocess.Popen(
            [sys.executable, "-u", str(_KERNEL_CHILD)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            cwd=os.getcwd(),
            env={**os.environ, "PYTHONUNBUFFERED": "1", "MPLBACKEND": "Agg"},
        )
        self.lock = threading.Lock()

    def close(self) -> None:
        if self.proc.poll() is None:
            try:
                self.proc.kill()
            except OSError:
                pass
        for stream in (self.proc.stdin, self.proc.stdout, self.proc.stderr):
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass

    def execute(self, code: str, timeout_s: float) -> NotebookExecuteResult:
        with self.lock:
            if self.proc.poll() is not None:
                err = ""
                if self.proc.stderr:
                    try:
                        err = self.proc.stderr.read() or ""
                    except OSError:
                        pass
                raise BrokenPipeError(err or "kernel process has exited")

            assert self.proc.stdin is not None
            payload = json.dumps({"code": code}, ensure_ascii=True) + "\n"
            self.proc.stdin.write(payload)
            self.proc.stdin.flush()

            raw = _readline_timeout(self.proc, timeout_s)
            if not raw.strip():
                raise BrokenPipeError("kernel produced no response (process may have crashed)")

            try:
                data = json.loads(raw)
            except json.JSONDecodeError as e:
                raise BrokenPipeError(f"invalid kernel response: {e}") from e

            exit_code = int(data.get("exit_code", 1 if not data.get("ok") else 0))
            stdout = _truncate(str(data.get("stdout", "")), _MAX_STREAM_CHARS)
            stderr = _truncate(str(data.get("stderr", "")), _MAX_STREAM_CHARS)
            images = _sanitize_images(data.get("images"))
            return NotebookExecuteResult(
                exit_code=exit_code,
                stdout=stdout,
                stderr=stderr,
                timed_out=False,
                images=images,
            )


_sessions: dict[str, NotebookKernelSession] = {}
_sessions_lock = threading.Lock()


def _dispose_session(session_id: str) -> None:
    with _sessions_lock:
        s = _sessions.pop(session_id, None)
    if s is not None:
        s.close()


def _get_session(session_id: str) -> NotebookKernelSession:
    with _sessions_lock:
        existing = _sessions.get(session_id)
        if existing is not None and existing.proc.poll() is None:
            return existing
        if existing is not None:
            existing.close()
        s = NotebookKernelSession()
        _sessions[session_id] = s
        return s


@router.post("/notebook/execute", response_model=NotebookExecuteResult)
def notebook_execute(body: NotebookExecuteBody) -> NotebookExecuteResult:
    """
    Execute Python in a **persistent** interpreter for ``session_id`` (shared globals across cells).

    ``session_id`` should be stable for one logical notebook (e.g. canvas id). Use
    ``DELETE /api/notebook/session`` to drop the kernel and its namespace.

    Do not expose this service on the public internet.
    """
    code = body.code
    if not code.strip():
        raise HTTPException(status_code=400, detail="Empty code.")

    sid = body.session_id.strip()
    if not sid:
        raise HTTPException(status_code=400, detail="Empty session_id.")

    try:
        session = _get_session(sid)
        return session.execute(code, _DEFAULT_TIMEOUT_S)
    except TimeoutError:
        _dispose_session(sid)
        return NotebookExecuteResult(
            exit_code=-1,
            stdout="",
            stderr=f"Timed out after {_DEFAULT_TIMEOUT_S}s",
            timed_out=True,
            images=[],
        )
    except BrokenPipeError as e:
        _dispose_session(sid)
        raise HTTPException(
            status_code=503,
            detail=f"Notebook kernel stopped or crashed; try Run again. ({e})",
        ) from e
    except OSError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.delete("/notebook/session")
def notebook_session_delete(session_id: str = Query(..., min_length=1, max_length=_MAX_SESSION_ID_CHARS)) -> dict[str, str]:
    """Stop the kernel for ``session_id`` (clears Python globals for that notebook)."""
    _dispose_session(session_id.strip())
    return {"status": "ok"}
