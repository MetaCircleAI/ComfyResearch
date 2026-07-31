from __future__ import annotations

import base64
import importlib
import inspect
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from comfy_research.config.remote_train_config import RemoteTrainConfig
from comfy_research.remote import ssh
from comfy_research.remote import parametric_path_cli, train_cli


def _artifact_module():
    try:
        return importlib.import_module("comfy_research.remote.execution_artifacts")
    except ModuleNotFoundError as exc:
        raise AssertionError("remote execution artifact protocol is missing") from exc


class RemoteExecutionArtifactTests(unittest.TestCase):
    def test_externalizes_checkpoint_fields_recursively_and_deduplicates_content(self) -> None:
        artifacts = _artifact_module()
        checkpoint = base64.standard_b64encode(b"checkpoint-bytes").decode("ascii")
        payload = {
            "nodes": [
                {
                    "id": "ck",
                    "data": {
                        "checkpoint_b64": checkpoint,
                        "memoryCheckpoint_b64": checkpoint,
                        "label": "keep-me",
                    },
                }
            ],
            "resume": {"checkpoint_b64": checkpoint},
        }

        encoded, blobs = artifacts.externalize_checkpoint_artifacts(payload)

        self.assertEqual(len(blobs), 1)
        self.assertEqual(next(iter(blobs.values())), b"checkpoint-bytes")
        node_data = encoded["nodes"][0]["data"]
        self.assertEqual(node_data["label"], "keep-me")
        self.assertEqual(node_data["checkpoint_b64"], node_data["memoryCheckpoint_b64"])
        self.assertEqual(node_data["checkpoint_b64"], encoded["resume"]["checkpoint_b64"])
        self.assertEqual(node_data["checkpoint_b64"]["$comfyArtifact"]["kind"], "checkpoint")
        self.assertEqual(payload["nodes"][0]["data"]["checkpoint_b64"], checkpoint)

    def test_materializes_checkpoint_references_before_schema_validation(self) -> None:
        artifacts = _artifact_module()
        checkpoint = base64.standard_b64encode(b"checkpoint-bytes").decode("ascii")
        encoded, blobs = artifacts.externalize_checkpoint_artifacts(
            {"nodes": [{"data": {"checkpoint_b64": checkpoint}}]}
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            artifacts.write_artifact_blobs(root, blobs)

            decoded = artifacts.materialize_checkpoint_artifacts(encoded, root)

        self.assertEqual(decoded["nodes"][0]["data"]["checkpoint_b64"], checkpoint)

    def test_repairs_a_partial_cached_artifact(self) -> None:
        artifacts = _artifact_module()
        checkpoint = base64.standard_b64encode(b"complete-checkpoint").decode("ascii")
        _encoded, blobs = artifacts.externalize_checkpoint_artifacts(
            {"checkpoint_b64": checkpoint}
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            rel = next(iter(blobs))
            partial = root / rel
            partial.parent.mkdir(parents=True)
            partial.write_bytes(b"partial")

            artifacts.write_artifact_blobs(root, blobs)

            self.assertEqual(partial.read_bytes(), b"complete-checkpoint")

    def test_caches_remote_train_checkpoint_for_later_remote_nodes(self) -> None:
        artifacts = _artifact_module()
        checkpoint = base64.standard_b64encode(b"trained-remotely").decode("ascii")
        event = {"type": "complete", "checkpoint_b64": checkpoint}
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            artifacts.cache_checkpoint_artifacts(event, root)
            _encoded, blobs = artifacts.externalize_checkpoint_artifacts(event)
            digest = next(iter(blobs))

            self.assertEqual((root / digest).read_bytes(), b"trained-remotely")

    def test_remote_clis_materialize_artifacts_through_one_json_decoder(self) -> None:
        artifacts = _artifact_module()
        load = getattr(artifacts, "load_remote_execution_json", None)
        self.assertTrue(callable(load), "remote CLIs must share one artifact-aware JSON decoder")
        checkpoint = base64.standard_b64encode(b"checkpoint-bytes").decode("ascii")
        encoded, blobs = artifacts.externalize_checkpoint_artifacts(
            {"nodes": [{"data": {"checkpoint_b64": checkpoint}}]}
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            artifacts.write_artifact_blobs(root, blobs)

            decoded = load(json.dumps(encoded), root=root)

        self.assertEqual(decoded["nodes"][0]["data"]["checkpoint_b64"], checkpoint)

    def test_every_remote_json_payload_uses_the_artifact_transport(self) -> None:
        prepare = getattr(ssh, "prepare_remote_execution_json", None)
        self.assertTrue(callable(prepare), "remote JSON transport must have one artifact-aware entry point")
        checkpoint = base64.standard_b64encode(b"large-checkpoint").decode("ascii")
        cfg = RemoteTrainConfig(host="gpu", remote_path="/repo", enabled=True)

        with patch.object(ssh, "_sync_remote_execution_artifacts") as sync:
            raw = prepare(
                {"checkpoint_b64": checkpoint},
                target_node_id="trainer",
                config=cfg,
                trainer_node_id="trainer",
            )

        self.assertNotIn(checkpoint.encode("ascii"), raw)
        synced = sync.call_args.args[1]
        self.assertEqual(list(synced.values()), [b"large-checkpoint"])

    def test_remote_transport_enforces_the_target_dependency_closure(self) -> None:
        cfg = RemoteTrainConfig(host="gpu", remote_path="/repo", enabled=True)
        payload = {
            "sampler_node_id": "sampler",
            "nodes": [
                {"id": "source", "type": "dataset", "data": {}},
                {"id": "trainer", "type": "trainer", "data": {"lossHistory": [1, 2, 3]}},
                {"id": "checkpoint", "type": "model_checkpoint", "data": {}},
                {"id": "sampler", "type": "parametric_path_sampler", "data": {}},
                {"id": "viz", "type": "curve_series_viz", "data": {}},
            ],
            "edges": [
                {"id": "source-trainer", "source": "source", "target": "trainer"},
                {"id": "trainer-checkpoint", "source": "trainer", "target": "checkpoint"},
                {"id": "checkpoint-sampler", "source": "checkpoint", "target": "sampler"},
                {"id": "sampler-viz", "source": "sampler", "target": "viz"},
            ],
        }

        with patch.object(ssh, "_sync_remote_execution_artifacts"):
            raw = ssh.prepare_remote_execution_json(
                payload,
                target_node_id="sampler",
                config=cfg,
            )

        encoded = json.loads(raw)
        self.assertEqual([node["id"] for node in encoded["nodes"]], ["checkpoint", "sampler"])
        self.assertEqual([edge["id"] for edge in encoded["edges"]], ["checkpoint-sampler"])

    def test_all_remote_execution_entry_points_use_the_shared_protocol(self) -> None:
        for fn in (
            ssh.validate_remote_train,
            ssh.iter_remote_train_stdout_lines,
            ssh.run_remote_parametric_path_sampler_json,
        ):
            self.assertIn("prepare_remote_execution_json(", inspect.getsource(fn), fn.__name__)
        for fn in (train_cli.main, parametric_path_cli.main):
            self.assertIn("load_remote_execution_json(", inspect.getsource(fn), fn.__module__)
        self.assertIn("cache_checkpoint_artifacts(", inspect.getsource(train_cli.main))


if __name__ == "__main__":
    unittest.main()
