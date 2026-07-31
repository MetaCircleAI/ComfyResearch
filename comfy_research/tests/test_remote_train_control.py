from __future__ import annotations

import io
import json
import signal
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import ANY, patch

from comfy_research.api.train import TrainControlRequest, post_train_control
from comfy_research.remote import control_cli, train_cli
from comfy_research.remote.session_ipc import (
    REMOTE_SESSION_DIR,
    RemoteTrainSession,
    cleanup_remote_session,
    create_remote_session,
    load_remote_session_manifest,
    resolve_session_control_path,
    write_remote_control_action,
)
from comfy_research.engine.runs.train_control import (
    bind_control_file,
    get_control,
    register_trainer,
    unregister_trainer,
)
from comfy_research.remote.session_registry import LocalRemoteTrainSession


class RemoteTrainControlFileTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        import comfy_research.remote.session_ipc as ipc

        self._orig_dir = ipc.REMOTE_SESSION_DIR
        ipc.REMOTE_SESSION_DIR = Path(self._tmpdir.name)

    def tearDown(self) -> None:
        import comfy_research.remote.session_ipc as ipc

        ipc.REMOTE_SESSION_DIR = self._orig_dir
        self._tmpdir.cleanup()

    def test_control_file_abort_reaches_train_loop(self) -> None:
        session = create_remote_session("trainer-a")
        register_trainer("trainer-a")
        bind_control_file("trainer-a", str(session.control_path))
        try:
            write_remote_control_action(session.control_path, "abort")
            ctrl = get_control("trainer-a")
            self.assertIsNotNone(ctrl)
            assert ctrl is not None
            self.assertTrue(ctrl.abort_requested)
            ctrl2 = get_control("trainer-a")
            self.assertIsNotNone(ctrl2)
            assert ctrl2 is not None
            self.assertTrue(ctrl2.abort_requested)
        finally:
            unregister_trainer("trainer-a")

    def test_control_file_pause_reaches_train_loop(self) -> None:
        session = create_remote_session("trainer-pause")
        register_trainer("trainer-pause")
        bind_control_file("trainer-pause", str(session.control_path))
        try:
            write_remote_control_action(session.control_path, "pause")
            ctrl = get_control("trainer-pause")
            self.assertIsNotNone(ctrl)
            assert ctrl is not None
            self.assertTrue(ctrl.pause_requested)
        finally:
            unregister_trainer("trainer-pause")

    def test_control_cli_manifest_roundtrip(self) -> None:
        session = create_remote_session("trainer-b")
        manifest = load_remote_session_manifest(session.session_id)
        self.assertEqual(manifest["trainer_node_id"], "trainer-b")
        self.assertEqual(manifest["session_id"], session.session_id)
        cleanup_remote_session(session.session_id)

    @patch("os.kill")
    def test_control_cli_abort_signals_remote_process(self, mock_kill: object) -> None:
        session = create_remote_session("trainer-abort")
        with patch.object(
            sys,
            "argv",
            ["remote_train_control_cli", "--session-id", session.session_id, "--action", "abort"],
        ):
            control_cli.main()

        mock_kill.assert_called_once_with(session.pid, signal.SIGTERM)

    def test_resolve_session_control_path_rejects_outside_session_dir(self) -> None:
        session = create_remote_session("trainer-c")
        outside = Path(self._tmpdir.name) / "evil.control.json"
        outside.write_text("{}", encoding="utf-8")
        with self.assertRaises(ValueError):
            resolve_session_control_path(session.session_id, str(outside))

    def test_resolve_session_control_path_accepts_manifest_path(self) -> None:
        session = create_remote_session("trainer-d")
        resolved = resolve_session_control_path(session.session_id, str(session.control_path))
        self.assertEqual(resolved, session.control_path.resolve())


class PostTrainControlRemoteFallbackTests(unittest.TestCase):
    @patch("comfy_research.api.train.run_remote_train_control", return_value=True)
    @patch("comfy_research.api.train.get_remote_train_session")
    @patch("comfy_research.api.train.request_abort", return_value=False)
    def test_abort_falls_back_to_remote_ssh_control(
        self,
        _mock_local_abort: object,
        mock_get_session: object,
        mock_remote_control: object,
    ) -> None:
        mock_get_session.return_value = LocalRemoteTrainSession("trainer-remote", "sess-abc")
        out = post_train_control(TrainControlRequest(trainer_node_id="trainer-remote", action="abort"))
        self.assertTrue(out["ok"])
        mock_remote_control.assert_called_once_with(
            "sess-abc",
            "abort",
            config=ANY,
        )

    @patch("comfy_research.api.train.run_remote_train_control", return_value=True)
    @patch("comfy_research.api.train.get_remote_train_session")
    @patch("comfy_research.api.train.request_pause", return_value=False)
    def test_pause_falls_back_to_remote_ssh_control(
        self,
        _mock_local_pause: object,
        mock_get_session: object,
        mock_remote_control: object,
    ) -> None:
        mock_get_session.return_value = LocalRemoteTrainSession("trainer-remote", "sess-pause")
        out = post_train_control(TrainControlRequest(trainer_node_id="trainer-remote", action="pause"))
        self.assertTrue(out["ok"])
        mock_remote_control.assert_called_once_with(
            "sess-pause",
            "pause",
            config=ANY,
        )


class RemoteTrainCliStartupTests(unittest.TestCase):
    def test_remote_session_is_published_before_prepare(self) -> None:
        order: list[str] = []
        stdout = io.StringIO()
        session = RemoteTrainSession(
            session_id="session-before-prepare",
            trainer_node_id="trainer-a",
            pid=1234,
            control_path=Path("/tmp/session.control.json"),
            manifest_path=Path("/tmp/session.json"),
        )

        def create_session(_trainer_node_id: str) -> RemoteTrainSession:
            order.append("session")
            return session

        def prepare(*_args: object, **_kwargs: object) -> object:
            order.append("prepare")
            return object()

        with (
            patch.object(sys, "argv", ["remote_train_cli"]),
            patch.object(sys, "stdin", io.StringIO('{"trainer_node_id":"trainer-a"}')),
            patch.object(sys, "stdout", stdout),
            patch.object(sys, "stderr", io.StringIO()),
            patch.object(train_cli, "create_remote_session", side_effect=create_session),
            patch.object(train_cli, "prepare_trainer_run", side_effect=prepare),
            patch.object(train_cli, "iter_trainer_events_from_context", return_value=iter(())),
            patch.object(train_cli, "bind_train_phase_emitter"),
            patch.object(train_cli, "bind_control_file"),
            patch.object(train_cli, "unbind_control_file"),
            patch.object(train_cli, "cleanup_remote_session"),
        ):
            train_cli.main()

        self.assertEqual(order, ["session", "prepare"])
        self.assertIn('"type":"remote_session"', stdout.getvalue())

    def test_sigterm_during_prepare_emits_aborted_and_cleans_session(self) -> None:
        stdout = io.StringIO()

        def prepare(*_args: object, **_kwargs: object) -> object:
            handler = signal.getsignal(signal.SIGTERM)
            self.assertTrue(callable(handler))
            assert callable(handler)
            try:
                handler(signal.SIGTERM, None)
            except Exception:
                return object()
            self.fail("SIGTERM handler returned")

        previous_handler = signal.signal(signal.SIGTERM, signal.SIG_DFL)
        self.addCleanup(signal.signal, signal.SIGTERM, previous_handler)
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch("comfy_research.remote.session_ipc.REMOTE_SESSION_DIR", Path(tmpdir)),
            patch.object(sys, "argv", ["remote_train_cli"]),
            patch.object(sys, "stdin", io.StringIO('{"trainer_node_id":"trainer-abort"}')),
            patch.object(sys, "stdout", stdout),
            patch.object(sys, "stderr", io.StringIO()),
            patch.object(train_cli, "prepare_trainer_run", side_effect=prepare),
            patch.object(train_cli, "iter_trainer_events_from_context", return_value=iter(())),
        ):
            train_cli.main()

            session_event = json.loads(stdout.getvalue().splitlines()[0])
            session_id = session_event["session_id"]
            self.assertEqual(stdout.getvalue().splitlines()[-1], '{"type":"aborted"}')
            self.assertFalse((Path(tmpdir) / f"{session_id}.json").exists())
            self.assertFalse((Path(tmpdir) / f"{session_id}.control.json").exists())

    def test_checkpoint_cache_failure_does_not_hide_complete_event(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()
        complete = {"type": "complete", "checkpoint_b64": "dHJhaW5lZA=="}

        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch("comfy_research.remote.session_ipc.REMOTE_SESSION_DIR", Path(tmpdir)),
            patch.object(sys, "argv", ["remote_train_cli"]),
            patch.object(sys, "stdin", io.StringIO('{"trainer_node_id":"trainer-cache"}')),
            patch.object(sys, "stdout", stdout),
            patch.object(sys, "stderr", stderr),
            patch.object(train_cli, "prepare_trainer_run", return_value=object()),
            patch.object(
                train_cli,
                "iter_trainer_events_from_context",
                return_value=iter((complete,)),
            ),
            patch.object(
                train_cli,
                "cache_checkpoint_artifacts",
                side_effect=OSError("disk full"),
            ),
        ):
            train_cli.main()

        self.assertEqual(json.loads(stdout.getvalue().splitlines()[-1]), complete)
        self.assertIn("disk full", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
