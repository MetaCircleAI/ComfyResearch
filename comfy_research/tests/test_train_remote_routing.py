from __future__ import annotations

import unittest

from comfy_research.api.train import _trainer_prefers_remote_gpu
from comfy_research.schemas.graph import Node
from comfy_research.schemas.train_request import TrainRequest


def _train_request_for_device(compute_device: str | None, **extra_data: object) -> TrainRequest:
    data: dict[str, object] = dict(extra_data)
    if compute_device is not None:
        data["computeDevice"] = compute_device
    return TrainRequest(
        trainer_node_id="trainer-1",
        nodes=[
            Node(
                id="trainer-1",
                type="trainer",
                data=data,
            )
        ],
        edges=[],
    )


class TrainerRemoteRoutingTests(unittest.TestCase):
    def test_local_cpu_and_mps_do_not_request_remote_gpu(self) -> None:
        for spec in (None, "", "cpu", "CPU", "mps", "MPS", "auto"):
            with self.subTest(spec=spec):
                self.assertFalse(_trainer_prefers_remote_gpu(_train_request_for_device(spec)))

    def test_cuda_without_explicit_remote_gpu_stays_local(self) -> None:
        """remote routing now requires an explicit remoteGpu=True opt-in
        (bare cuda / cuda:N = local CUDA device)."""
        for spec in ("cuda", "CUDA", "cuda:0", "cuda:1"):
            with self.subTest(spec=spec):
                self.assertFalse(_trainer_prefers_remote_gpu(_train_request_for_device(spec)))

    def test_cuda_with_remote_gpu_true_requests_remote_for_all_cuda_specs(self) -> None:
        for spec in ("cuda", "CUDA", "cuda:0", "cuda:1"):
            with self.subTest(spec=spec):
                self.assertTrue(
                    _trainer_prefers_remote_gpu(_train_request_for_device(spec, remoteGpu=True))
                )

    def test_cuda_with_remote_gpu_false_stays_local(self) -> None:
        req = _train_request_for_device("cuda", remoteGpu=False)
        self.assertFalse(_trainer_prefers_remote_gpu(req))

    def test_cuda_with_remote_gpu_true_requests_remote(self) -> None:
        req = _train_request_for_device("cuda", remoteGpu=True)
        self.assertTrue(_trainer_prefers_remote_gpu(req))


if __name__ == "__main__":
    unittest.main()
