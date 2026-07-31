"""Verified local input table for the information-bottleneck dataset node.

The bundled MATLAB asset arrived with PR #161. Its claimed upstream source and
licence were not recorded in the PR, so this module deliberately makes no claim
that it is an official release. The SHA-256 check prevents a silently replaced
table from being used as the reproducibility input.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import struct
import zlib

import numpy as np


VAR_U_PATH = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "reproduction_inputs"
    / "information_bottleneck"
    / "var_u.mat"
)
VAR_U_SHA256 = "0ba9551878a855396a8de0cbaae620788ede1e7b1eb2b8373bbe017d5ea02036"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mat_element(payload: bytes, offset: int, *, aligned: bool = True) -> tuple[int, bytes, int]:
    """Read one little-endian MAT v5 element, including compact tag form."""
    if offset + 4 > len(payload):
        raise ValueError("truncated MATLAB element")
    first = struct.unpack_from("<I", payload, offset)[0]
    compact_size = first >> 16
    if compact_size:
        return first & 0xFFFF, payload[offset + 4 : offset + 4 + compact_size], offset + 8
    if offset + 8 > len(payload):
        raise ValueError("truncated MATLAB element tag")
    size = struct.unpack_from("<I", payload, offset + 4)[0]
    start = offset + 8
    end = start + size
    if end > len(payload):
        raise ValueError("truncated MATLAB element payload")
    return first, payload[start:end], end + (-size % 8) if aligned else end


def _load_mat_v5_uint8_arrays(source: Path) -> dict[str, np.ndarray]:
    """Read this asset's compressed uint8 MAT v5 matrices without SciPy.

    This deliberately supports only the two-dimensional numeric matrices stored
    in this versioned asset. Unknown element classes fail closed rather than
    being interpreted as a potentially different dataset.
    """
    raw = source.read_bytes()
    if len(raw) < 128 or raw[126:128] != b"IM":
        raise ValueError("information-bottleneck asset is not little-endian MAT v5")
    offset, arrays = 128, {}
    while offset < len(raw):
        # This asset packs compressed top-level elements back-to-back.
        element_type, compressed, offset = _mat_element(raw, offset, aligned=False)
        if element_type != 15:  # miCOMPRESSED
            raise ValueError("information-bottleneck asset contains an unsupported MAT element")
        try:
            matrix_type, matrix, _ = _mat_element(zlib.decompress(compressed), 0)
        except zlib.error as exc:
            raise ValueError("information-bottleneck asset has invalid compressed MATLAB data") from exc
        if matrix_type != 14:  # miMATRIX
            raise ValueError("information-bottleneck asset contains a non-matrix value")
        _flags_type, _flags, cursor = _mat_element(matrix, 0)
        dims_type, dims_raw, cursor = _mat_element(matrix, cursor)
        name_type, name_raw, cursor = _mat_element(matrix, cursor)
        value_type, values_raw, _ = _mat_element(matrix, cursor)
        if dims_type != 5 or name_type != 1 or value_type != 2:  # miINT32, miINT8, miUINT8
            raise ValueError("information-bottleneck asset uses unsupported MATLAB matrix encoding")
        dims = np.frombuffer(dims_raw, dtype="<i4")
        if dims.size != 2 or np.any(dims < 1):
            raise ValueError("information-bottleneck asset has invalid matrix dimensions")
        expected = int(dims[0]) * int(dims[1])
        if len(values_raw) != expected:
            raise ValueError("information-bottleneck asset matrix byte count does not match dimensions")
        name = name_raw.decode("ascii")
        arrays[name] = np.frombuffer(values_raw, dtype=np.uint8).reshape(tuple(dims), order="F")
    return arrays


def load_var_u(path: str | Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Load the bundled binary input/label table after structural/hash checks."""
    source = Path(path) if path is not None else VAR_U_PATH
    if not source.is_file():
        raise FileNotFoundError(f"information-bottleneck input asset is missing: {source}")
    actual = _sha256(source)
    if actual != VAR_U_SHA256:
        raise ValueError(
            "information-bottleneck input asset SHA-256 mismatch: "
            f"expected {VAR_U_SHA256}, got {actual}"
        )
    payload = _load_mat_v5_uint8_arrays(source)
    if "F" not in payload or "y" not in payload:
        raise ValueError("information-bottleneck input asset must contain F and y")
    x = np.asarray(payload["F"])
    y = np.asarray(payload["y"]).reshape(-1)
    if x.shape != (4096, 12) or y.shape != (4096,):
        raise ValueError(f"information-bottleneck asset must have F[4096,12] and y[4096]; got {x.shape} and {y.shape}")
    if not np.isin(x, (0, 1)).all() or not np.isin(y, (0, 1)).all():
        raise ValueError("information-bottleneck input asset must contain binary F and y")
    return x.astype(np.float32, copy=False), y.astype(np.int64, copy=False)


def build_information_bottleneck_arrays(
    rng: np.random.Generator,
    *,
    train_size: int,
    test_size: int,
    input_dim: int = 12,
    asset_path: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Sample deterministic train/test arrays from the verified local table."""
    if input_dim != 12:
        raise ValueError("information-bottleneck dataset fixes input_dim=12")
    x_all, y_all = load_var_u(asset_path)
    total = len(x_all)
    count = min(max(1, int(train_size)), total)
    # Match the formal reproduction engine exactly; NumPy choice without
    # replacement is not guaranteed to equal permutation[:count].
    train_idx = rng.choice(total, size=count, replace=False)
    if test_size <= 0:
        return x_all[train_idx].copy(), y_all[train_idx].copy(), None, None
    if test_size >= total:
        return x_all[train_idx].copy(), y_all[train_idx].copy(), x_all.copy(), y_all.copy()
    test_idx = rng.choice(total, size=max(1, int(test_size)), replace=False)
    return x_all[train_idx].copy(), y_all[train_idx].copy(), x_all[test_idx].copy(), y_all[test_idx].copy()
