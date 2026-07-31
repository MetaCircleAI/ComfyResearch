from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from comfy_research.engine.runs.cuda_devices import list_local_cuda_devices


class ListLocalCudaDevicesTests(unittest.TestCase):
    @patch("comfy_research.engine.runs.cuda_devices.torch.cuda.is_available", return_value=False)
    def test_returns_empty_when_cuda_unavailable(self, _mock: MagicMock) -> None:
        self.assertEqual(list_local_cuda_devices(), [])

    @patch("comfy_research.engine.runs.cuda_devices.torch.cuda.get_device_properties")
    @patch("comfy_research.engine.runs.cuda_devices.torch.cuda.device_count", return_value=2)
    @patch("comfy_research.engine.runs.cuda_devices.torch.cuda.is_available", return_value=True)
    def test_lists_each_device(
        self,
        _avail: MagicMock,
        _count: MagicMock,
        get_props: MagicMock,
    ) -> None:
        def _props(i: int) -> MagicMock:
            p = MagicMock()
            p.name = f"GPU-{i}"
            p.total_memory = (8 + i) * 1024**3
            return p

        get_props.side_effect = _props
        out = list_local_cuda_devices()
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0]["index"], 0)
        self.assertEqual(out[0]["name"], "GPU-0")
        self.assertEqual(out[0]["totalMemoryMb"], 8 * 1024)
        self.assertEqual(out[1]["index"], 1)


if __name__ == "__main__":
    unittest.main()
