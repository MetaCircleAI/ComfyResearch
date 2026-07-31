from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

# Allow running this script directly from repo root without installing the package.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from comfy_research.api.dataset_tensor import DatasetTensorRequest, post_dataset_tensor


def _decode_tensor_response(resp) -> tuple[list[int], np.ndarray]:
    shape_hdr = resp.headers.get("X-Tensor-Shape")
    if not shape_hdr:
        raise RuntimeError("Missing X-Tensor-Shape header from dataset_tensor response.")
    shape = [int(x) for x in json.loads(shape_hdr)]
    values = np.frombuffer(resp.body, dtype=np.float32)
    expected = int(np.prod(shape, dtype=np.int64)) if shape else 0
    if expected != int(values.size):
        raise RuntimeError(f"Shape/values mismatch: shape={shape}, expected={expected}, got={values.size}.")
    return shape, values


def _rows_to_coords(flat: np.ndarray, row_count: int, row_width: int) -> np.ndarray:
    if row_width < 2 or flat.size < row_count * row_width:
        return np.zeros((0, 3), dtype=np.float32)
    reshaped = flat[: row_count * row_width].reshape(row_count, row_width)
    if row_width >= 3:
        return reshaped[:, :3].astype(np.float32)
    out = np.zeros((row_count, 3), dtype=np.float32)
    out[:, :2] = reshaped[:, :2]
    return out


def _extract_structure_coords(shape: list[int], values: np.ndarray, sample_index: int) -> np.ndarray:
    if not shape:
        return np.zeros((0, 3), dtype=np.float32)
    dims = [int(x) for x in shape if int(x) > 0]
    if not dims:
        return np.zeros((0, 3), dtype=np.float32)

    if len(dims) == 1:
        l = dims[0]
        if l % 3 == 0:
            return _rows_to_coords(values, l // 3, 3)
        if l % 2 == 0:
            return _rows_to_coords(values, l // 2, 2)
        return np.zeros((0, 3), dtype=np.float32)

    if len(dims) == 2:
        a, b = dims
        if b in (2, 3):
            return _rows_to_coords(values, a, b)
        # [batch, flat] where flat encodes xyzxyz... or xyxy...
        if b % 3 == 0 or b % 2 == 0:
            sample_count = a
            s = min(max(int(sample_index), 0), sample_count - 1)
            off = s * b
            chunk = values[off : off + b]
            if b % 3 == 0:
                return _rows_to_coords(chunk, b // 3, 3)
            return _rows_to_coords(chunk, b // 2, 2)
        return np.zeros((0, 3), dtype=np.float32)

    # [batch, n, c] where c is 2 or 3
    b, n, c = dims[0], dims[1], dims[2]
    if c not in (2, 3):
        return np.zeros((0, 3), dtype=np.float32)
    s = min(max(int(sample_index), 0), b - 1)
    sample_len = n * c
    off = s * sample_len
    chunk = values[off : off + sample_len]
    return _rows_to_coords(chunk, n, c)


def _verify_dataset_kind(kind: str, extra: dict[str, object] | None = None) -> None:
    data = {
        "inputDim": 64,
        "outputDim": 48,
        "trainSize": 32,
        "testSize": 8,
        "seed": 0,
    }
    if extra:
        data.update(extra)

    req = DatasetTensorRequest(
        dataset_node_id=f"{kind}-0",
        dataset_node_type=kind,
        dataset_data=data,
        split="train",
        tensor_key="output",
    )
    resp = post_dataset_tensor(req)
    shape, values = _decode_tensor_response(resp)
    if len(shape) != 2:
        raise AssertionError(f"{kind}: expected rank-2 output tensor [batch, feat], got {shape}.")
    if shape[0] < 2:
        raise AssertionError(f"{kind}: need at least 2 train samples for sample-index switching, got shape={shape}.")

    s0 = _extract_structure_coords(shape, values, sample_index=0)
    s1 = _extract_structure_coords(shape, values, sample_index=1)

    if s0.shape[0] == 0 or s1.shape[0] == 0:
        raise AssertionError(f"{kind}: failed to extract structure coordinates from shape={shape}.")
    if s0.shape != s1.shape:
        raise AssertionError(f"{kind}: extracted shape mismatch between sample 0 and 1: {s0.shape} vs {s1.shape}.")
    if np.allclose(s0, s1):
        raise AssertionError(f"{kind}: sample index did not change extracted structure (sample0 ~= sample1).")

    print(f"[OK] {kind}: tensor shape={shape}, extracted points={s0.shape[0]}, sample 0 != sample 1")


def main() -> None:
    checks: Iterable[tuple[str, dict[str, object] | None]] = [
        ("alphafold1_toy_dataset", {"localMix": 0.35}),
        ("alphafold2_toy_dataset", {"recycleSteps": 2}),
        ("alphafold3_toy_dataset", {"diffusionNoiseScale": 0.4, "diffusionTimesteps": 32}),
    ]
    for kind, extra in checks:
        _verify_dataset_kind(kind, extra)
    print("All AlphaFold toy dataset structure-panel checks passed.")


if __name__ == "__main__":
    main()

