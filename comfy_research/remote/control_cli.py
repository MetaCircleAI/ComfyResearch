"""Write pause/abort into a remote train session control file (second SSH invocation)."""

from __future__ import annotations

import argparse
import os
import signal
import sys

from comfy_research.remote.session_ipc import (
    ControlAction,
    load_remote_session_manifest,
    resolve_session_control_path,
    write_remote_control_action,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Signal pause/abort for a remote train session.")
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--action", required=True, choices=["pause", "abort"])
    args = parser.parse_args()

    try:
        manifest = load_remote_session_manifest(args.session_id.strip())
        control_path = manifest.get("control_path")
        if not isinstance(control_path, str) or not control_path.strip():
            print("session manifest missing control_path", file=sys.stderr)
            sys.exit(1)
        resolved = resolve_session_control_path(args.session_id.strip(), control_path.strip())
        write_remote_control_action(resolved, args.action)  # type: ignore[arg-type]
        if args.action == "abort":
            pid = manifest.get("pid")
            if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 1:
                raise ValueError("session manifest missing valid pid")
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"{type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
