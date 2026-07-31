from __future__ import annotations

import os
import shlex
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from comfy_research.config.remote_train_config import RemoteTrainConfig, _parse_saved
from comfy_research.remote import ssh as remote_ssh
from comfy_research.remote.ssh import _local_training_bundle_digest, _ssh_shell_command


class RemoteTrainBootstrapDigestTests(unittest.TestCase):
    def test_bundle_digest_is_stable_hex(self) -> None:
        digest = _local_training_bundle_digest()
        self.assertEqual(len(digest), 64)
        self.assertEqual(digest, _local_training_bundle_digest())

    def test_bundle_digest_uses_mtime_cache(self) -> None:
        first = _local_training_bundle_digest()
        second = _local_training_bundle_digest()
        self.assertEqual(first, second)

    def test_bundle_digest_changes_when_an_older_file_is_added(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            newest = root / "newest.py"
            older = root / "older.bin"
            newest.write_bytes(b"newest")
            older.write_bytes(b"older")
            os.utime(newest, (200, 200))
            os.utime(older, (100, 100))

            with (
                patch.object(remote_ssh, "_BUNDLE_DIGEST_CACHE", None),
                patch.object(
                    remote_ssh,
                    "_iter_training_bundle_files",
                    side_effect=[
                        [("newest.py", newest)],
                        [("newest.py", newest), ("older.bin", older)],
                    ],
                ),
            ):
                before = _local_training_bundle_digest()
                after = _local_training_bundle_digest()

        self.assertNotEqual(before, after)

    def test_training_bundle_excludes_local_dataset_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "comfy_research").mkdir()
            (root / "comfy_research" / "__init__.py").write_text("")
            (root / "requirements.txt").write_text("")
            ib_input = root / "data" / "reproduction_inputs" / "information_bottleneck" / "var_u.mat"
            ib_input.parent.mkdir(parents=True)
            ib_input.write_bytes(b"paper input")
            batches = root / "data" / "cifar10" / "cifar-10-batches-py"
            batches.mkdir(parents=True)
            (batches / "data_batch_1").write_bytes(b"local cache")

            with patch.object(Path, "cwd", return_value=root):
                files = remote_ssh._iter_training_bundle_files()

        self.assertFalse(any(rel.startswith("data/cifar10/") for rel, _path in files))

    def test_training_bundle_includes_local_dataset_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "comfy_research").mkdir()
            (root / "comfy_research" / "__init__.py").write_text("")
            (root / "requirements.txt").write_text("")
            ib_input = root / "data" / "reproduction_inputs" / "information_bottleneck" / "var_u.mat"
            ib_input.parent.mkdir(parents=True)
            ib_input.write_bytes(b"paper input")
            batches = root / "data" / "cifar10" / "cifar-10-batches-py"
            batches.mkdir(parents=True)
            (batches / "data_batch_1").write_bytes(b"local cache")

            with patch.object(Path, "cwd", return_value=root):
                files = remote_ssh._iter_training_bundle_files(include_dataset=True)

        self.assertTrue(any(rel.endswith("data_batch_1") for rel, _path in files))

    def test_remote_config_upload_dataset_round_trip(self) -> None:
        from comfy_research.api.train import RemoteTrainConfigBody, _cfg_from_body, _cfg_response

        self.assertFalse(RemoteTrainConfig().upload_dataset)
        self.assertTrue(_parse_saved({"upload_dataset": True}).upload_dataset)
        response = _cfg_response(_cfg_from_body(RemoteTrainConfigBody(upload_dataset=True)))
        self.assertTrue(response.upload_dataset)

    def test_bootstrap_uses_remote_config_dataset_upload_choice(self) -> None:
        cfg = RemoteTrainConfig(
            host="example.com",
            remote_path="/root/ComfyResearch",
            upload_dataset=True,
            enabled=True,
        )
        with (
            patch.object(remote_ssh, "_local_training_bundle_digest", return_value="bundle") as digest,
            patch.object(remote_ssh, "_local_requirements_digest", return_value="requirements"),
            patch.object(remote_ssh, "_sync_local_training_bundle", return_value=True) as sync,
            patch.object(remote_ssh, "_setup_remote_train_environment", return_value={"ok": True}),
        ):
            list(remote_ssh.iter_remote_bootstrap_events(cfg))

        digest.assert_called_once_with(include_dataset=True)
        self.assertTrue(sync.call_args.kwargs["include_dataset"])

    @patch("comfy_research.remote.ssh._ssh_shell_command")
    @patch("comfy_research.remote.ssh.subprocess.run")
    def test_remote_bundle_is_current_checks_init_py(self, mock_run: object, mock_ssh: object) -> None:
        from comfy_research.config.remote_train_config import RemoteTrainConfig
        from comfy_research.remote.ssh import _remote_bundle_is_current

        mock_ssh.return_value = (["ssh"], None)
        mock_run.return_value = type(
            "R",
            (),
            {"returncode": 0, "stdout": b"BUNDLE_UP_TO_DATE=1\n", "stderr": b""},
        )()
        cfg = RemoteTrainConfig(
            host="h",
            user="root",
            remote_path="/root/ComfyResearch",
            password="x",
            enabled=True,
        )
        self.assertTrue(_remote_bundle_is_current(cfg, "abc123"))
        script = mock_ssh.call_args[0][1]
        self.assertIn("__init__.py", script)
        self.assertIn("requirements.txt", script)


class RemoteSshCommandTests(unittest.TestCase):
    def test_remote_shell_script_stays_one_bash_c_argument(self) -> None:
        cfg = RemoteTrainConfig(
            host="example.com",
            user="root",
            remote_path="/root/ComfyResearch",
            enabled=True,
        )
        script = "set -euo pipefail\ntrap 'echo stopped' TERM\nprintf '%s\\n' \"$HOME\""

        cmd, _env = _ssh_shell_command(cfg, script)
        target_index = cmd.index("root@example.com")
        remote_command = " ".join(cmd[target_index + 1 :])

        self.assertEqual(shlex.split(remote_command), ["bash", "-lc", script])


if __name__ == "__main__":
    unittest.main()
