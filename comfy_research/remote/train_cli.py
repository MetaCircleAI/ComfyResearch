"""stdin JSON (TrainRequest) -> stdout NDJSON training events (same schema as POST /api/train).

Used by SSH delegation from the local API; see docs/remote-gpu-lambda.md.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys

from fastapi import HTTPException

from comfy_research.remote.session_ipc import cleanup_remote_session, create_remote_session
from comfy_research.remote.execution_artifacts import (
    cache_checkpoint_artifacts,
    load_remote_execution_json,
    remote_artifact_root,
)
from comfy_research.engine.runs.train_control import bind_control_file, unbind_control_file
from comfy_research.engine.runs.train_phase import bind_train_phase_emitter
from comfy_research.schemas.train_request import TrainRequest, sanitize_train_ndjson_value
from comfy_research.engine.runs.trainer_run import iter_trainer_events_from_context, prepare_trainer_run


class _RemoteTrainAborted(BaseException):
    pass


def _abort_on_sigterm(_signum: int, _frame: object) -> None:
    raise _RemoteTrainAborted


def _ndjson_line(event: dict) -> bytes:
    safe = sanitize_train_ndjson_value(event)
    return (json.dumps(safe, separators=(",", ":")) + "\n").encode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream trainer NDJSON to stdout from stdin JSON.")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Run prepare_trainer_run only; exit 0 on success (no stdout stream).",
    )
    args = parser.parse_args()

    raw = sys.stdin.read()
    if not raw.strip():
        print("empty stdin", file=sys.stderr)
        sys.exit(2)
    try:
        body = TrainRequest.model_validate(load_remote_execution_json(raw))
    except Exception as e:
        print(f"invalid TrainRequest JSON: {e}", file=sys.stderr)
        sys.exit(2)

    buf = getattr(sys.stdout, "buffer", None)
    out_write = buf.write if buf is not None else lambda b: sys.stdout.write(b.decode("utf-8"))

    def _flush_stdout() -> None:
        if buf is not None:
            buf.flush()
        else:
            sys.stdout.flush()

    def _emit_phase_ndjson(payload: dict) -> None:
        # Dataset download/extract during remote prepare must reach NDJSON stdout so
        # the local API can relay phase status to the UI. Flush per line: prepare can
        # block for minutes and the SSH pipe is not line-buffered.
        out_write(_ndjson_line({"type": "phase", **payload}))
        _flush_stdout()

    remote_session = None
    previous_sigterm_handler = None
    try:
        # Publish the session before prepare so Abort can terminate dataset download,
        # extraction, and other blocking setup instead of waiting for the first step.
        if not args.validate_only:
            bind_train_phase_emitter(_emit_phase_ndjson)
            remote_session = create_remote_session(body.trainer_node_id)
            bind_control_file(body.trainer_node_id, str(remote_session.control_path))
            previous_sigterm_handler = signal.signal(signal.SIGTERM, _abort_on_sigterm)
            out_write(
                _ndjson_line(
                    {
                        "type": "remote_session",
                        "session_id": remote_session.session_id,
                        "trainer_node_id": body.trainer_node_id,
                        "pid": remote_session.pid,
                    }
                )
            )
            _flush_stdout()

        ctx = prepare_trainer_run(
            body.nodes,
            body.edges,
            body.trainer_node_id,
            resume=body.resume,
            hessian_oversized_policy=body.hessian_oversized_policy,
            validate_only=args.validate_only,
        )
        if args.validate_only:
            return

        for event in iter_trainer_events_from_context(ctx):
            out_write(_ndjson_line(event))
            if buf is not None:
                buf.flush()
            else:
                sys.stdout.flush()
            if event.get("type") in {"complete", "paused"}:
                try:
                    cache_checkpoint_artifacts(event, remote_artifact_root())
                except Exception as e:
                    print(f"checkpoint cache warning: {type(e).__name__}: {e}", file=sys.stderr)
    except _RemoteTrainAborted:
        try:
            out_write(_ndjson_line({"type": "aborted"}))
            _flush_stdout()
        except BrokenPipeError:
            pass
    except HTTPException as e:
        detail = e.detail
        msg = json.dumps(detail, ensure_ascii=False) if isinstance(detail, dict) else str(detail)
        print(msg, file=sys.stderr)
        sys.exit(1)
    except BrokenPipeError:
        sys.exit(0)
    except Exception as e:
        print(f"{type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        if not args.validate_only:
            if previous_sigterm_handler is not None:
                signal.signal(signal.SIGTERM, previous_sigterm_handler)
            bind_train_phase_emitter(None)
            unbind_control_file(body.trainer_node_id)
            if remote_session is not None:
                cleanup_remote_session(remote_session.session_id)


if __name__ == "__main__":
    main()
