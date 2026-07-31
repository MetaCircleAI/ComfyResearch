"""Vision datasets for trainer_run (numpy → torch): MNIST (IDX URL), Gaussian blobs, shape world, hole counting."""

from __future__ import annotations

import gzip
import os
import pickle
import shutil
import struct
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import HTTPException

from comfy_research.schemas.graph import NodeKind
from comfy_research.engine.runs.train_phase import (
    CIFAR10_DATASET_NAME,
    MNIST_DATASET_NAME,
    emit_dataset_download,
    emit_dataset_extract,
    emit_dataset_load,
    emit_train_phase,
    remote_phase_message,
)


def _scalar_int(v: Any, default: int) -> int:
    if isinstance(v, list) and v:
        v = v[0]
    try:
        return int(v)
    except (TypeError, ValueError):
        return int(default)


def _scalar_str(v: Any, default: str) -> str:
    if isinstance(v, list) and v:
        v = v[0]
    s = str(v or "").strip()
    return s if s else default


def _scalar_float(v: Any, default: float) -> float:
    if isinstance(v, list) and v:
        v = v[0]
    try:
        return float(v)
    except (TypeError, ValueError):
        return float(default)


def _vision_flatten_feature_matrix(x_nchw: np.ndarray) -> np.ndarray:
    if x_nchw.ndim != 4:
        raise HTTPException(status_code=500, detail="Internal: expected NCHW vision tensors before flatten.")
    n = int(x_nchw.shape[0])
    rest = int(np.prod(np.asarray(x_nchw.shape[1:], dtype=np.int64)))
    return np.ascontiguousarray(x_nchw.reshape(n, rest), dtype=np.float32)


def vision_num_classes(ds_kind: NodeKind, dd: dict[str, Any]) -> int:
    if ds_kind in (NodeKind.mnist_dataset, NodeKind.cifar10_dataset):
        return 10
    if ds_kind == NodeKind.gaussian_blob_dataset:
        return max(2, min(64, _scalar_int(dd.get("numClasses"), 10)))
    if ds_kind == NodeKind.shape_world_dataset:
        return 3
    if ds_kind == NodeKind.hole_counting_dataset:
        mh = _scalar_int(dd.get("maxHoles"), 3)
        return max(1, mh + 1)
    raise HTTPException(status_code=500, detail=f"Internal: unknown vision dataset {ds_kind}")


def _mnist_download_bytes(url: str, cache_dir: str) -> bytes:
    os.makedirs(cache_dir, exist_ok=True)
    name = url.rsplit("/", 1)[-1]
    path = os.path.join(cache_dir, name)

    def _download() -> None:
        tmp_path = f"{path}.part"
        try:
            urllib.request.urlretrieve(url, tmp_path)  # noqa: S310 — intentional dataset URL
            os.replace(tmp_path, path)
        except OSError as e:
            if os.path.isfile(tmp_path):
                os.unlink(tmp_path)
            raise HTTPException(
                status_code=502,
                detail=f"Could not download MNIST file {url}: {e}",
            ) from e

    if not os.path.isfile(path):
        _download()
    with open(path, "rb") as f:
        data = f.read()
    try:
        gzip.decompress(data)
    except OSError:
        if os.path.isfile(path):
            os.unlink(path)
        _download()
        with open(path, "rb") as f:
            data = f.read()
    return data


def _parse_mnist_images(buf: bytes) -> np.ndarray:
    magic, n, rows, cols = struct.unpack_from(">IIII", buf, 0)
    if magic != 2051:
        raise HTTPException(status_code=400, detail="Invalid MNIST image file magic.")
    data = np.frombuffer(buf, dtype=np.uint8, offset=16).reshape(n, rows * cols)
    return data.reshape(n, 1, rows, cols).astype(np.float32) / 255.0


def _parse_mnist_labels(buf: bytes) -> np.ndarray:
    magic, n = struct.unpack_from(">II", buf, 0)
    if magic != 2049:
        raise HTTPException(status_code=400, detail="Invalid MNIST label file magic.")
    return np.frombuffer(buf, dtype=np.uint8, offset=8).astype(np.int64)


def _load_mnist_official(
    *,
    train_n: int,
    test_n: int,
    rng: np.random.Generator,
    cache_dir: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    base = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    files = {
        "train_x": base + "train-images-idx3-ubyte.gz",
        "train_y": base + "train-labels-idx1-ubyte.gz",
        "test_x": base + "t10k-images-idx3-ubyte.gz",
        "test_y": base + "t10k-labels-idx1-ubyte.gz",
    }
    os.makedirs(cache_dir, exist_ok=True)
    if any(not os.path.isfile(os.path.join(cache_dir, url.rsplit("/", 1)[-1])) for url in files.values()):
        emit_dataset_download(MNIST_DATASET_NAME)
    x_tr_full = _parse_mnist_images(gzip.decompress(_mnist_download_bytes(files["train_x"], cache_dir)))
    y_tr_full = _parse_mnist_labels(gzip.decompress(_mnist_download_bytes(files["train_y"], cache_dir)))
    x_te_full = _parse_mnist_images(gzip.decompress(_mnist_download_bytes(files["test_x"], cache_dir)))
    y_te_full = _parse_mnist_labels(gzip.decompress(_mnist_download_bytes(files["test_y"], cache_dir)))

    n_tr = min(train_n, x_tr_full.shape[0])
    n_te = min(test_n, x_te_full.shape[0]) if test_n > 0 else 0
    idx_tr = rng.permutation(x_tr_full.shape[0])[:n_tr]
    x_train = x_tr_full[idx_tr].copy()
    y_train = y_tr_full[idx_tr].copy()
    if n_te <= 0:
        return x_train, y_train, None, None
    idx_te = rng.permutation(x_te_full.shape[0])[:n_te]
    return x_train, y_train, x_te_full[idx_te].copy(), y_te_full[idx_te].copy()


_CIFAR10_SJTU_MIRROR_URL = (
    "https://scidata.sjtu.edu.cn/records/p4t8m-rbe26/files/"
    "cifar-10-python.tar.gz?download=1"
)
_CIFAR10_GITHUB_MIRROR_URL = (
    "https://github.com/Digital-Media/cv_data/releases/download/cifar-10/cifar-10-python.tar.gz"
)
_CIFAR10_OFFICIAL_DIRECT_URL = "https://cave.cs.toronto.edu/kriz/cifar-10-python.tar.gz"
_CIFAR10_LEGACY_REDIRECT_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
# CIFAR-10 官方归档的 SHA-256(cifar-10-python.tar.gz;
# 下载物先验摘要再落盘/解包/unpickle。env 覆盖 URL 必须配套
# COMFYRESEARCH_CIFAR10_SHA256,否则拒绝——信任边界显式化)。
_CIFAR10_KNOWN_SHA256 = "6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce"


def _cifar10_expected_sha256() -> str:
    env = os.environ.get("COMFYRESEARCH_CIFAR10_SHA256", "").strip().lower()
    return env or _CIFAR10_KNOWN_SHA256


def _sha256_of_file(path: str) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


_CIFAR10_EXPECTED_BYTES = 170_498_071
_CIFAR10_DOWNLOAD_TIMEOUT_S = 120
_CIFAR10_SJTU_DOWNLOAD_TIMEOUT_S = 300
_CIFAR10_PROGRESS_EVERY_BYTES = 512 * 1024
_CIFAR10_MIN_BATCH_BYTES = 30_000_000
_CIFAR10_BATCH_FILES = (*(f"data_batch_{i}" for i in range(1, 6)), "test_batch")


def _repo_root() -> Path:
    # engine/datasets/ 层级(后):parents[3] = repo 根(repro 旧层级是 [2])。
    return Path(__file__).resolve().parents[3]


def _cifar10_batches_complete(root: str | Path) -> bool:
    path = Path(root)
    try:
        return all(
            (path / name).is_file()
            and (path / name).stat().st_size >= _CIFAR10_MIN_BATCH_BYTES
            for name in _CIFAR10_BATCH_FILES
        )
    except OSError:
        return False


def _bundled_cifar10_batches_root() -> str | None:
    candidates: list[Path] = []
    env = os.environ.get("COMFYRESEARCH_CIFAR10_BUNDLED", "").strip()
    if env:
        p = Path(env)
        candidates.append(p if p.name == "cifar-10-batches-py" else p / "cifar-10-batches-py")
    candidates.extend(
        [
            _repo_root() / "data" / "cifar10" / "cifar-10-batches-py",
            Path.cwd() / "data" / "cifar10" / "cifar-10-batches-py",
        ]
    )
    for c in candidates:
        if _cifar10_batches_complete(c):
            return str(c)
    return None


def _default_mnist_cache_dir() -> str:
    for p in (_repo_root() / "data" / "mnist", Path.cwd() / "data" / "mnist"):
        if p.is_dir():
            return str(p)
    return os.path.join(os.path.expanduser("~"), ".cache", "comfy_research_mnist")


def _default_cifar10_cache_dir() -> str:
    env = os.environ.get("COMFYRESEARCH_CIFAR10_CACHE", "").strip()
    if env:
        return env
    bundled_parent = _repo_root() / "data" / "cifar10"
    if (_bundled_cifar10_batches_root() is not None) or bundled_parent.is_dir():
        return str(bundled_parent)
    autodl_tmp = "/root/autodl-tmp/comfy_research_cifar10"
    if os.path.isdir("/root/autodl-tmp"):
        return autodl_tmp
    return os.path.join(os.path.expanduser("~"), ".cache", "comfy_research_cifar10")


def _cifar10_download_urls() -> list[str]:
    urls: list[str] = []
    env_url = os.environ.get("COMFYRESEARCH_CIFAR10_URL", "").strip()
    if env_url:
        urls.append(env_url)
    extra = os.environ.get("COMFYRESEARCH_CIFAR10_URLS", "").strip()
    if extra:
        urls.extend(u.strip() for u in extra.split(",") if u.strip())
    for candidate in (
        _CIFAR10_SJTU_MIRROR_URL,
        _CIFAR10_GITHUB_MIRROR_URL,
        _CIFAR10_OFFICIAL_DIRECT_URL,
        _CIFAR10_LEGACY_REDIRECT_URL,
    ):
        if candidate not in urls:
            urls.append(candidate)
    return urls


def _cifar10_download_timeout_s(url: str) -> int:
    if "scidata.sjtu.edu.cn" in url:
        return _CIFAR10_SJTU_DOWNLOAD_TIMEOUT_S
    return _CIFAR10_DOWNLOAD_TIMEOUT_S


def _verify_cifar10_tar_digest(tar_path: str) -> None:
    """extract 前恒验摘要——旧缓存/手放/历史恶意 tar 一律不豁免。"""
    digest = _sha256_of_file(tar_path)
    expected = _cifar10_expected_sha256()
    if digest != expected:
        raise OSError(f"CIFAR-10 archive sha256 mismatch: got {digest}, expected {expected}")


def _download_cifar10_tar(tar_path: str) -> None:
    tmp_path = f"{tar_path}.part"
    if os.path.isfile(tmp_path):
        os.unlink(tmp_path)
    last_report = {"bytes": 0}

    def _progress(blocknum: int, blocksize: int, totalsize: int) -> None:
        received = blocknum * blocksize
        if received - last_report["bytes"] < _CIFAR10_PROGRESS_EVERY_BYTES and (
            totalsize <= 0 or received < totalsize
        ):
            return
        last_report["bytes"] = received
        emit_train_phase(
            "dataset_download",
            remote_phase_message(f"downloading {CIFAR10_DATASET_NAME}"),
            meta={
                "receivedBytes": received,
                "totalBytes": totalsize if totalsize > 0 else _CIFAR10_EXPECTED_BYTES,
            },
        )

    errors: list[str] = []
    for url in _cifar10_download_urls():
        timeout_s = _cifar10_download_timeout_s(url)
        emit_dataset_download(
            CIFAR10_DATASET_NAME,
            meta={"url": url, "timeoutSec": timeout_s},
        )
        try:
            import socket

            old_timeout = socket.getdefaulttimeout()
            socket.setdefaulttimeout(timeout_s)
            try:
                urllib.request.urlretrieve(url, tmp_path, reporthook=_progress)  # noqa: S310
            finally:
                socket.setdefaulttimeout(old_timeout)
            size = os.path.getsize(tmp_path)
            if size < int(_CIFAR10_EXPECTED_BYTES * 0.95):
                raise OSError(
                    f"download too small ({size} bytes, expected ~{_CIFAR10_EXPECTED_BYTES})"
                )
            _verify_cifar10_tar_digest(tmp_path)
            os.replace(tmp_path, tar_path)
            emit_train_phase(
                "dataset_download_done",
                remote_phase_message(f"downloaded {CIFAR10_DATASET_NAME}"),
                meta={"url": url, "tarBytes": size},
            )
            return
        except (OSError, TimeoutError, urllib.error.URLError) as e:
            errors.append(f"{url}: {e}")
            if os.path.isfile(tmp_path):
                os.unlink(tmp_path)
    raise HTTPException(
        status_code=502,
        detail=(
            "Could not download CIFAR-10 archive. Tried: "
            + "; ".join(errors)
            + ". Run `python scripts/fetch_bundled_cifar10.py` locally, then re-train (bundled data syncs on remote bootstrap)."
        ),
    )


def _parse_cifar10_batch(buf: bytes) -> tuple[np.ndarray, np.ndarray]:
    obj = pickle.loads(buf, encoding="bytes")
    data = np.asarray(obj[b"data"], dtype=np.uint8)
    labels = np.asarray(obj[b"labels"], dtype=np.int64).reshape(-1)
    n = data.shape[0]
    x = data.reshape(n, 3, 32, 32).astype(np.float32) / 255.0
    return x, labels


def denormalize_cifar10_images(images: np.ndarray, normalize: str) -> np.ndarray:
    """Return RGB floats in [0, 1] for sampling previews and similarity metrics."""
    out = np.asarray(images, dtype=np.float32)
    if normalize == "minus_one_to_one":
        out = (out + 1.0) * 0.5
    return np.clip(out, 0.0, 1.0)


def _cifar10_subset_indices(
    labels: np.ndarray,
    size: int,
    *,
    seed: int,
    class_balanced: bool,
) -> np.ndarray:
    n = min(max(0, int(size)), int(labels.shape[0]))
    rng = np.random.default_rng(int(seed))
    if not class_balanced:
        return rng.permutation(labels.shape[0])[:n]
    classes = np.unique(labels)
    chunks: list[np.ndarray] = []
    base, remainder = divmod(n, len(classes))
    for index, label in enumerate(classes):
        want = base + (1 if index < remainder else 0)
        candidates = np.flatnonzero(labels == label)
        chunks.append(rng.permutation(candidates)[:want])
    return rng.permutation(np.concatenate(chunks)) if chunks else np.empty((0,), dtype=np.int64)


def cifar10_jastrzbski_split_indices(
    total: int,
    train_n: int,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """The successful Fig. 1 run used NumPy's legacy RandomState split."""
    permutation = np.random.RandomState(seed).permutation(total)
    n = min(max(0, int(train_n)), total)
    return permutation[:n], permutation[n:]


def cifar10_pixel_mean_global_std(x_train: np.ndarray) -> tuple[np.ndarray, float]:
    """Per-pixel mean image plus one scalar std, matching the Fig. 1 run."""
    pixel_sum = np.zeros(x_train.shape[1:], dtype=np.float64)
    squared_sum = 0.0
    for start in range(0, len(x_train), 1024):
        chunk = np.asarray(x_train[start : start + 1024], dtype=np.float64)
        pixel_sum += chunk.sum(axis=0)
        squared_sum += float(np.square(chunk).sum())
    mean = pixel_sum / len(x_train)
    element_count = len(x_train) * int(np.prod(x_train.shape[1:]))
    variance = squared_sum / element_count - float(np.mean(np.square(mean)))
    std = float(np.sqrt(max(float(variance), 0.0)))
    if std == 0:
        raise ValueError("CIFAR training data has zero standard deviation")
    return mean, std


def _safe_tar_members(tf: "tarfile.TarFile", cache_dir: str) -> list["tarfile.TarInfo"]:
    """拒绝逃逸 cache_dir 的成员(../、绝对路径、link)与
    意外顶层目录(官方归档恒为 cifar-10-batches-py/)。"""
    root = os.path.realpath(cache_dir)
    members = []
    for m in tf.getmembers():
        if not (m.isreg() or m.isdir()):
            raise HTTPException(status_code=502, detail=f"CIFAR-10 archive has non-file member: {m.name}")
        top = m.name.split("/", 1)[0]
        if top != "cifar-10-batches-py":
            raise HTTPException(status_code=502, detail=f"CIFAR-10 archive has unexpected top-level path: {m.name}")
        dest = os.path.realpath(os.path.join(cache_dir, m.name))
        if dest != root and not dest.startswith(root + os.sep):
            raise HTTPException(status_code=502, detail=f"CIFAR-10 archive member escapes cache dir: {m.name}")
        members.append(m)
    return members


def _extract_cifar10_tar(tar_path: str, cache_dir: str) -> None:
    emit_dataset_extract(CIFAR10_DATASET_NAME)
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(path=cache_dir, members=_safe_tar_members(tf, cache_dir))


def _ensure_cifar10_extracted(cache_dir: str) -> tuple[str, dict[str, object]]:
    bundled = _bundled_cifar10_batches_root()
    if bundled:
        return bundled, {
            "cacheDir": bundled,
            "batchesCached": True,
            "source": "bundled",
            "downloaded": False,
            "extracted": False,
        }

    root = os.path.join(cache_dir, "cifar-10-batches-py")
    tar_path = os.path.join(cache_dir, "cifar-10-python.tar.gz")
    info: dict[str, object] = {
        "cacheDir": cache_dir,
        "batchesCached": _cifar10_batches_complete(root),
        "tarCached": os.path.isfile(tar_path),
        "tarBytes": os.path.getsize(tar_path) if os.path.isfile(tar_path) else 0,
        "downloaded": False,
        "extracted": False,
    }
    if info["batchesCached"]:
        return root, info
    os.makedirs(cache_dir, exist_ok=True)

    def _wipe_cifar_cache() -> None:
        if os.path.isfile(tar_path):
            os.unlink(tar_path)
        shutil.rmtree(root, ignore_errors=True)

    if os.path.isdir(root) and not info["batchesCached"]:
        shutil.rmtree(root, ignore_errors=True)

    if not os.path.isfile(tar_path):
        t0 = time.monotonic()
        _download_cifar10_tar(tar_path)
        info["downloaded"] = True
        info["downloadMs"] = round((time.monotonic() - t0) * 1000)
        info["tarBytes"] = os.path.getsize(tar_path)
    else:
        # 缓存 tar 同样过摘要门;不匹配则清缓存重下(重下路径内部再验)。
        try:
            _verify_cifar10_tar_digest(tar_path)
        except OSError:
            _wipe_cifar_cache()
            t0 = time.monotonic()
            _download_cifar10_tar(tar_path)
            info["downloaded"] = True
            info["downloadMs"] = round((time.monotonic() - t0) * 1000)
            info["tarBytes"] = os.path.getsize(tar_path)
    try:
        t1 = time.monotonic()
        _extract_cifar10_tar(tar_path, cache_dir)
        if not _cifar10_batches_complete(root):
            raise OSError("CIFAR-10 extracted batch files are incomplete")
        info["extracted"] = True
        info["extractMs"] = round((time.monotonic() - t1) * 1000)
    except (tarfile.ReadError, EOFError, OSError):
        _wipe_cifar_cache()
        try:
            t0 = time.monotonic()
            _download_cifar10_tar(tar_path)
            info["downloaded"] = True
            info["downloadMs"] = round((time.monotonic() - t0) * 1000)
            t1 = time.monotonic()
            _extract_cifar10_tar(tar_path, cache_dir)
            if not _cifar10_batches_complete(root):
                raise OSError("CIFAR-10 extracted batch files are incomplete")
            info["extracted"] = True
            info["extractMs"] = round((time.monotonic() - t1) * 1000)
        except (tarfile.ReadError, EOFError, OSError) as e2:
            raise HTTPException(
                status_code=502,
                detail=f"CIFAR-10 archive corrupt or incomplete after retry: {e2}",
            ) from e2
    if not _cifar10_batches_complete(root):
        raise HTTPException(status_code=500, detail="CIFAR-10 archive extracted but batch dir missing.")
    return root, info


def _load_cifar10_official(
    *,
    train_n: int,
    test_n: int,
    rng: np.random.Generator,
    cache_dir: str,
    split_seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    t0 = time.monotonic()
    root, cache_info = _ensure_cifar10_extracted(cache_dir)
    emit_dataset_load(
        CIFAR10_DATASET_NAME,
        meta={**cache_info, "trainN": train_n, "testN": test_n},
    )
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for i in range(1, 6):
        path = os.path.join(root, f"data_batch_{i}")
        with open(path, "rb") as f:
            xb, yb = _parse_cifar10_batch(f.read())
        xs.append(xb)
        ys.append(yb)
    x_tr_full = np.concatenate(xs, axis=0)
    y_tr_full = np.concatenate(ys, axis=0)
    with open(os.path.join(root, "test_batch"), "rb") as f:
        x_te_full, y_te_full = _parse_cifar10_batch(f.read())

    n_tr = min(train_n, x_tr_full.shape[0])
    n_te = min(test_n, x_te_full.shape[0]) if test_n > 0 else 0
    if split_seed is None:
        idx_tr = rng.permutation(x_tr_full.shape[0])[:n_tr]
    else:
        idx_tr, _ = cifar10_jastrzbski_split_indices(
            x_tr_full.shape[0], n_tr, seed=split_seed
        )
    x_train = x_tr_full[idx_tr].copy()
    y_train = y_tr_full[idx_tr].copy()
    x_te_out: np.ndarray | None = None
    y_te_out: np.ndarray | None = None
    if n_te > 0:
        idx_te = (
            rng.permutation(x_te_full.shape[0])[:n_te]
            if split_seed is None
            else np.arange(n_te)
        )
        x_te_out = x_te_full[idx_te].copy()
        y_te_out = y_te_full[idx_te].copy()
    emit_train_phase(
        "dataset_load_done",
        remote_phase_message(f"loaded {CIFAR10_DATASET_NAME}"),
        meta={
            "loadMs": round((time.monotonic() - t0) * 1000),
            "trainShape": list(x_train.shape),
            "testShape": list(x_te_out.shape) if x_te_out is not None else None,
        },
    )
    return x_train, y_train, x_te_out, y_te_out


def build_cifar10_arrays(
    dd: dict[str, Any],
    train_n: int,
    test_n: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Build CIFAR-10 arrays using a fixed, reproducible transformation order.

    Official source pixels start in ``[0, 1]``. The pipeline is: sample rows,
    perturb input pixels, apply any paper preprocessing, apply diffusion scaling,
    then corrupt training labels only. This keeps both reproduction protocols
    explicit and prevents scaling from silently changing white-normalized data.
    """
    cache = _scalar_str(dd.get("downloadCacheDir"), "").strip() or _default_cifar10_cache_dir()
    recipe = _scalar_str(dd.get("trainingRecipe"), "standard").strip().lower()
    split_seed = (
        _scalar_int(dd.get("initSeed", dd.get("seed")), 0)
        if recipe == "jastrzbski_fig1"
        else None
    )
    x_train, y_train, x_test, y_test = _load_cifar10_official(
        train_n=train_n,
        test_n=test_n,
        rng=rng,
        cache_dir=cache,
        split_seed=split_seed,
    )
    if recipe != "jastrzbski_fig1":
        subset_seed = _scalar_int(dd.get("subsetSeed", dd.get("initSeed", 0)), 0)
        class_balanced = bool(dd.get("classBalanced", True))
        root, _ = _ensure_cifar10_extracted(cache)
        xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        for i in range(1, 6):
            with open(os.path.join(root, f"data_batch_{i}"), "rb") as f:
                xb, yb = _parse_cifar10_batch(f.read())
            xs.append(xb)
            ys.append(yb)
        full_x = np.concatenate(xs, axis=0)
        full_y = np.concatenate(ys, axis=0)
        indices = _cifar10_subset_indices(full_y, train_n, seed=subset_seed, class_balanced=class_balanced)
        x_train, y_train = full_x[indices].copy(), full_y[indices].copy()

    preprocessing = _scalar_str(dd.get("preprocessing"), "none").strip()
    normalize = _scalar_str(dd.get("normalize"), "zero_one").strip().lower()
    if preprocessing != "none" and normalize == "minus_one_to_one":
        raise HTTPException(
            status_code=400,
            detail="CIFAR per-image whitening cannot be combined with normalize=minus_one_to_one. Use normalize=zero_one.",
        )
    randomization_seed = _scalar_int(dd.get("initSeed", dd.get("seed")), 0)
    x_train, y_train, x_test, y_test = _apply_cifar10_randomization(
        x_train,
        y_train,
        x_test,
        y_test,
        input_transform=_scalar_str(dd.get("inputTransform"), "none").strip(),
        label_corruption=_scalar_float(dd.get("labelCorruption"), 0.0),
        rng=np.random.default_rng(randomization_seed),
        preprocessing=preprocessing,
    )
    return _apply_cifar10_normalization(x_train, y_train, x_test, y_test, normalize=normalize)


def _apply_cifar10_normalization(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray | None,
    y_test: np.ndarray | None,
    *,
    normalize: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Apply diffusion scaling after all image-space processing."""
    if normalize == "zero_one":
        return x_train, y_train, x_test, y_test
    if normalize != "minus_one_to_one":
        raise HTTPException(status_code=400, detail="CIFAR normalize must be zero_one or minus_one_to_one.")
    x_train = np.asarray(x_train * 2.0 - 1.0, dtype=np.float32)
    if x_test is not None:
        x_test = np.asarray(x_test * 2.0 - 1.0, dtype=np.float32)
    return x_train, y_train, x_test, y_test


def _apply_cifar10_randomization(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray | None,
    y_test: np.ndarray | None,
    *,
    input_transform: str,
    label_corruption: float,
    rng: np.random.Generator,
    preprocessing: str = "none",
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Apply fixed-across-epochs pixel changes before paper preprocessing."""
    mode = input_transform or "none"
    valid = {"none", "shuffled_pixels", "random_pixels", "gaussian"}
    if mode not in valid:
        raise HTTPException(status_code=400, detail=f"Unknown CIFAR-10 input transform: {mode}")
    x_tr = np.asarray(x_train, dtype=np.float32).copy()
    x_te = None if x_test is None else np.asarray(x_test, dtype=np.float32).copy()

    def _flat_apply(x: np.ndarray, permutation: np.ndarray) -> np.ndarray:
        rows = x.reshape(x.shape[0], -1)
        return rows[:, permutation].reshape(x.shape).astype(np.float32, copy=False)

    if mode == "shuffled_pixels":
        permutation = rng.permutation(int(np.prod(x_tr.shape[1:])))
        x_tr = _flat_apply(x_tr, permutation)
        if x_te is not None:
            x_te = _flat_apply(x_te, permutation)
    elif mode == "random_pixels":
        def _independent_permutations(x: np.ndarray) -> np.ndarray:
            rows = x.reshape(x.shape[0], -1)
            out = np.empty_like(rows)
            for i in range(rows.shape[0]):
                out[i] = rows[i, rng.permutation(rows.shape[1])]
            return out.reshape(x.shape)
        x_tr = _independent_permutations(x_tr)
        if x_te is not None:
            x_te = _independent_permutations(x_te)
    elif mode == "gaussian":
        mean = float(np.mean(x_tr))
        std = max(float(np.std(x_tr)), 1e-6)
        x_tr = np.clip(rng.normal(mean, std, size=x_tr.shape), 0.0, 1.0).astype(np.float32)
        if x_te is not None:
            x_te = np.clip(rng.normal(mean, std, size=x_te.shape), 0.0, 1.0).astype(np.float32)

    x_tr = _apply_cifar10_preprocessing(x_tr, preprocessing)
    if x_te is not None:
        x_te = _apply_cifar10_preprocessing(x_te, preprocessing)
    ratio = min(1.0, max(0.0, float(label_corruption)))
    y_tr = np.asarray(y_train, dtype=np.int64).copy()
    if ratio > 0.0 and y_tr.size > 0:
        mask = rng.random(y_tr.shape[0]) < ratio
        y_tr[mask] = rng.integers(0, 10, size=int(mask.sum()), dtype=np.int64)
    y_te = None if y_test is None else np.asarray(y_test, dtype=np.int64).copy()
    return x_tr, y_tr, x_te, y_te


def _apply_cifar10_preprocessing(x: np.ndarray, preprocessing: str) -> np.ndarray:
    """Apply a deterministic CIFAR preprocessing pipeline to NCHW images."""
    mode = preprocessing or "none"
    if mode == "none":
        return np.asarray(x, dtype=np.float32)
    if mode != "center_crop_28_per_image_whiten":
        raise HTTPException(status_code=400, detail=f"Unknown CIFAR-10 preprocessing: {mode}")
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim != 4 or arr.shape[-2] < 28 or arr.shape[-1] < 28:
        raise HTTPException(status_code=400, detail="center_crop_28_per_image_whiten expects NCHW images at least 28x28.")
    top = (arr.shape[-2] - 28) // 2
    left = (arr.shape[-1] - 28) // 2
    cropped = np.ascontiguousarray(arr[..., top : top + 28, left : left + 28])
    rows = cropped.reshape(cropped.shape[0], -1)
    means = rows.mean(axis=1, keepdims=True)
    stddevs = rows.std(axis=1, keepdims=True)
    adjusted = np.maximum(stddevs, 1.0 / np.sqrt(float(rows.shape[1])))
    whitened = (rows - means) / adjusted
    return np.ascontiguousarray(whitened.reshape(cropped.shape), dtype=np.float32)

def build_gaussian_blob_arrays(
    dd: dict[str, Any],
    train_n: int,
    test_n: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Class-specific 2D Gaussian blobs on a grid + configurable per-pixel noise (toy classification)."""
    size = max(8, min(64, _scalar_int(dd.get("imageSize"), 28)))
    noise_level = max(0.0, _scalar_float(dd.get("noiseLevel"), 0.15))
    proto_std = min(0.1, 0.2 * noise_level)
    c = max(2, min(64, _scalar_int(dd.get("numClasses"), 10)))
    protos = np.zeros((c, 1, size, size), dtype=np.float32)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    g = max(1, int(np.ceil(np.sqrt(float(c)))))
    margin = float(size) * 0.12
    span = float(size) - 2.0 * margin
    cell = span / float(g) if g > 0 else float(size)
    sigma = float(size) * 0.18
    for k in range(c):
        row, col = divmod(k, g)
        cx = margin + (float(col) + 0.5) * cell
        cy = margin + (float(row) + 0.5) * cell
        protos[k, 0] = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma**2))
    if proto_std > 0:
        protos += proto_std * rng.standard_normal(protos.shape).astype(np.float32)

    def sample(n: int) -> tuple[np.ndarray, np.ndarray]:
        y = rng.integers(0, c, size=(n,), dtype=np.int64)
        x = protos[y].copy()
        if noise_level > 0:
            x += noise_level * rng.standard_normal(x.shape).astype(np.float32)
        x = np.clip(x, 0.0, 1.0)
        return x, y

    x_train, y_train = sample(train_n)
    if test_n <= 0:
        return x_train, y_train, None, None
    x_test, y_test = sample(test_n)
    return x_train, y_train, x_test, y_test


def build_mnist_arrays(
    dd: dict[str, Any],
    train_n: int,
    test_n: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Official MNIST IDX (28×28); prefers data/mnist, else HTTPS cache."""
    cache = _scalar_str(dd.get("downloadCacheDir"), "").strip() or _default_mnist_cache_dir()
    return _load_mnist_official(train_n=train_n, test_n=test_n, rng=rng, cache_dir=cache)


def _draw_disk(img: np.ndarray, cx: float, cy: float, r: float, val: float) -> None:
    h, w = img.shape[-2:]
    yy, xx = np.ogrid[0:h, 0:w]
    m = (xx - cx) ** 2 + (yy - cy) ** 2 <= r**2
    img[..., m] = val


def _draw_triangle(img: np.ndarray, pts: list[tuple[float, float]], fill: float) -> None:
    from matplotlib.path import Path as MplPath

    h, w = img.shape[-2:]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    flat = np.column_stack([xx.ravel(), yy.ravel()])
    p = MplPath(pts)
    inside = p.contains_points(flat).reshape(h, w)
    img[..., inside] = fill


def build_shape_world_arrays(
    dd: dict[str, Any],
    train_n: int,
    test_n: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """3 classes: 0 square, 1 triangle, 2 circle on gray background."""
    size = max(16, min(96, _scalar_int(dd.get("imageSize"), 32)))
    noise_level = max(0.0, _scalar_float(dd.get("noiseLevel"), 0.04))
    bg = 0.15
    fg = 0.95

    def one() -> tuple[np.ndarray, int]:
        x = np.full((1, size, size), bg, dtype=np.float32)
        cls = int(rng.integers(0, 3))
        margin = size * 0.15
        if cls == 0:
            # Filled axis-aligned square in the H×W plane (index as x[0, row, col]).
            inner = max(1.0, float(size) - 2.0 * margin)
            s = float(rng.uniform(inner * 0.18, inner * 0.48))
            x0 = float(rng.uniform(margin, float(size) - margin - s))
            y0 = float(rng.uniform(margin, float(size) - margin - s))
            r0 = max(0, int(np.floor(y0)))
            r1 = min(size, int(np.ceil(y0 + s)))
            c0 = max(0, int(np.floor(x0)))
            c1 = min(size, int(np.ceil(x0 + s)))
            if r1 > r0 and c1 > c0:
                x[0, r0:r1, c0:c1] = fg
        elif cls == 1:
            cx = rng.uniform(margin + size * 0.2, size - margin - size * 0.2)
            cy = rng.uniform(margin + size * 0.2, size - margin - size * 0.2)
            r = rng.uniform(size * 0.15, size * 0.28)
            pts = [
                (cx, cy - r),
                (cx - 0.866 * r, cy + 0.5 * r),
                (cx + 0.866 * r, cy + 0.5 * r),
            ]
            _draw_triangle(x[0], pts, fg)
        else:
            cx = rng.uniform(margin + size * 0.2, size - margin - size * 0.2)
            cy = rng.uniform(margin + size * 0.2, size - margin - size * 0.2)
            r = rng.uniform(size * 0.12, size * 0.25)
            _draw_disk(x[0], cx, cy, r, fg)
        if noise_level > 0:
            x += noise_level * rng.standard_normal(x.shape).astype(np.float32)
        return np.clip(x, 0.0, 1.0), cls

    xs = []
    ys = []
    for _ in range(train_n):
        im, c = one()
        xs.append(im)
        ys.append(c)
    x_train = np.stack(xs, axis=0).astype(np.float32)
    y_train = np.array(ys, dtype=np.int64)
    if test_n <= 0:
        return x_train, y_train, None, None
    xs_te = []
    ys_te = []
    for _ in range(test_n):
        im, c = one()
        xs_te.append(im)
        ys_te.append(c)
    return x_train, y_train, np.stack(xs_te, axis=0).astype(np.float32), np.array(ys_te, dtype=np.int64)


def build_hole_counting_arrays(
    dd: dict[str, Any],
    train_n: int,
    test_n: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Binary foreground blob with k circular holes; label = k (0..maxHoles)."""
    size = max(24, min(96, _scalar_int(dd.get("imageSize"), 48)))
    max_holes = max(0, min(8, _scalar_int(dd.get("maxHoles"), 3)))
    n_cls = max_holes + 1
    bg = 0.1
    fg = 0.9

    def one() -> tuple[np.ndarray, int]:
        x = np.full((1, size, size), bg, dtype=np.float32)
        k = int(rng.integers(0, n_cls))
        cx = size * 0.5 + rng.uniform(-size * 0.12, size * 0.12)
        cy = size * 0.5 + rng.uniform(-size * 0.12, size * 0.12)
        outer_r = rng.uniform(size * 0.28, size * 0.38)
        yy, xx = np.ogrid[0:size, 0:size]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= outer_r**2
        x[0, mask] = fg
        for _i in range(k):
            hx = cx + rng.uniform(-outer_r * 0.55, outer_r * 0.55)
            hy = cy + rng.uniform(-outer_r * 0.55, outer_r * 0.55)
            hr = rng.uniform(outer_r * 0.12, outer_r * 0.22)
            hole = (xx - hx) ** 2 + (yy - hy) ** 2 <= hr**2
            x[0, hole] = bg
        x += 0.03 * rng.standard_normal(x.shape).astype(np.float32)
        return np.clip(x, 0.0, 1.0), k

    xs = []
    ys = []
    for _ in range(train_n):
        im, c = one()
        xs.append(im)
        ys.append(c)
    x_train = np.stack(xs, axis=0).astype(np.float32)
    y_train = np.array(ys, dtype=np.int64)
    if test_n <= 0:
        return x_train, y_train, None, None
    xs_te = []
    ys_te = []
    for _ in range(test_n):
        im, c = one()
        xs_te.append(im)
        ys_te.append(c)
    return x_train, y_train, np.stack(xs_te, axis=0).astype(np.float32), np.array(ys_te, dtype=np.int64)


def build_vision_numpy_arrays(
    ds_kind: NodeKind,
    dd: dict[str, Any],
    train_n: int,
    test_n: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    if ds_kind == NodeKind.mnist_dataset:
        return build_mnist_arrays(dd, train_n, test_n, rng)
    if ds_kind == NodeKind.cifar10_dataset:
        return build_cifar10_arrays(dd, train_n, test_n, rng)
    if ds_kind == NodeKind.gaussian_blob_dataset:
        return build_gaussian_blob_arrays(dd, train_n, test_n, rng)
    if ds_kind == NodeKind.shape_world_dataset:
        return build_shape_world_arrays(dd, train_n, test_n, rng)
    if ds_kind == NodeKind.hole_counting_dataset:
        return build_hole_counting_arrays(dd, train_n, test_n, rng)
    raise HTTPException(status_code=400, detail=f"Unsupported vision dataset node: {ds_kind}")
