/**
 * Self-contained Python cells for vision dataset nodes (Code tab / blog export).
 * Logic mirrors ``comfy_research.engine.vision_datasets_runtime`` (same formulas, seeds, dtypes).
 */
import {
  defaultVisionDatasetData,
  type VisionDatasetKind,
  type VisionDatasetNodeData,
} from "../components/nodes/visionDatasetDefaults";

function firstScalar<T>(v: T | T[] | undefined | null, fallback: T): T {
  if (v === undefined || v === null) return fallback;
  if (Array.isArray(v)) return (v.length ? v[0]! : fallback) as T;
  return v as T;
}

function buildCifar10Loaders(
  pySym: string,
  title: string,
  trainN: number,
  testN: number,
  seed: number,
  flatten: boolean,
  downloadCacheDir: string,
): string {
  return `# === ${title} (cifar10_dataset) ===
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from comfy_research.engine.vision_datasets_runtime import build_cifar10_arrays

def fn_${pySym}_loaders(batch_size: int = 64, device: str | torch.device = "cpu"):
    rng = np.random.default_rng(int(${JSON.stringify(seed)}))
    data = {"downloadCacheDir": ${JSON.stringify(downloadCacheDir)}}
    x_tr, y_tr, x_te, y_te = build_cifar10_arrays(
        data, int(${JSON.stringify(trainN)}), int(${JSON.stringify(testN)}), rng
    )
    if ${flatten ? "True" : "False"}:
        x_tr = x_tr.reshape(int(x_tr.shape[0]), -1).astype(np.float32)
        if x_te is not None:
            x_te = x_te.reshape(int(x_te.shape[0]), -1).astype(np.float32)
    x_tr_t = torch.from_numpy(x_tr)
    y_tr_t = torch.from_numpy(y_tr).long()
    train_loader = DataLoader(TensorDataset(x_tr_t, y_tr_t), batch_size=int(batch_size), shuffle=True)
    if x_te is None or y_te is None:
        return train_loader, None
    x_te_t = torch.from_numpy(x_te)
    y_te_t = torch.from_numpy(y_te).long()
    test_loader = DataLoader(TensorDataset(x_te_t, y_te_t), batch_size=int(batch_size), shuffle=False)
    return train_loader, test_loader
`;
}

function buildGaussianBlobLoaders(
  pySym: string,
  title: string,
  trainN: number,
  testN: number,
  seed: number,
  flatten: boolean,
  imageSize: number,
  noiseLevel: number,
  numClasses: number,
): string {
  return `# === ${title} (gaussian_blob_dataset) ===
# Multi-class toy images: class prototypes are 2D Gaussian blobs on a grid + per-pixel noise (matches engine).
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def fn_${pySym}_loaders(batch_size: int = 64, device: str | torch.device = "cpu"):
    rng = np.random.default_rng(int(${JSON.stringify(seed)}))
    train_n = int(${JSON.stringify(trainN)})
    test_n = int(${JSON.stringify(testN)})
    size = int(max(8, min(64, ${JSON.stringify(imageSize)})))
    noise_level = float(max(0.0, ${JSON.stringify(noiseLevel)}))
    proto_std = float(min(0.1, 0.2 * noise_level))
    c = int(max(2, min(64, ${JSON.stringify(numClasses)})))
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
        x_test = y_test = None
    else:
        x_test, y_test = sample(test_n)

    flat = bool(${flatten ? "True" : "False"})

    def pack_x(x: np.ndarray | None) -> torch.Tensor | None:
        if x is None:
            return None
        t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to(device)
        if flat:
            t = t.reshape(t.shape[0], -1)
        return t

    xt = pack_x(x_train)
    yt = torch.from_numpy(y_train).to(device).long()
    if x_test is not None and y_test is not None:
        xte = pack_x(x_test)
        yte = torch.from_numpy(y_test).to(device).long()
        test_ds = TensorDataset(xte, yte)
    else:
        test_ds = None
    train_ds = TensorDataset(xt, yt)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size) if test_ds else None
    return train_loader, test_loader
`;
}

function buildMnistLoaders(
  pySym: string,
  title: string,
  trainN: number,
  testN: number,
  seed: number,
  flatten: boolean,
  downloadCacheDir: string,
): string {
  return `# === ${title} (mnist_dataset) ===
# Official MNIST IDX (28×28) via HTTPS; cached locally (same URLs and layout as engine).
import gzip
import os
import struct
import urllib.request

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def fn_${pySym}_loaders(batch_size: int = 64, device: str | torch.device = "cpu"):
    rng = np.random.default_rng(int(${JSON.stringify(seed)}))
    train_n = int(${JSON.stringify(trainN)})
    test_n = int(${JSON.stringify(testN)})
    cache_dir = (${JSON.stringify(downloadCacheDir)}).strip() or os.path.join(
        os.path.expanduser("~"), ".cache", "comfy_research_mnist"
    )
    os.makedirs(cache_dir, exist_ok=True)
    base = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    files = {
        "train_x": base + "train-images-idx3-ubyte.gz",
        "train_y": base + "train-labels-idx1-ubyte.gz",
        "test_x": base + "t10k-images-idx3-ubyte.gz",
        "test_y": base + "t10k-labels-idx1-ubyte.gz",
    }

    def _download(url: str) -> bytes:
        name = url.rsplit("/", 1)[-1]
        path = os.path.join(cache_dir, name)
        if not os.path.isfile(path):
            urllib.request.urlretrieve(url, path)
        with open(path, "rb") as f:
            return f.read()

    def _parse_images(buf: bytes) -> np.ndarray:
        magic, n, rows, cols = struct.unpack_from(">IIII", buf, 0)
        if magic != 2051:
            raise ValueError("Invalid MNIST image magic")
        data = np.frombuffer(buf, dtype=np.uint8, offset=16).reshape(n, rows * cols)
        return data.reshape(n, 1, rows, cols).astype(np.float32) / 255.0

    def _parse_labels(buf: bytes) -> np.ndarray:
        magic, n = struct.unpack_from(">II", buf, 0)
        if magic != 2049:
            raise ValueError("Invalid MNIST label magic")
        return np.frombuffer(buf, dtype=np.uint8, offset=8).astype(np.int64)

    x_tr_full = _parse_images(gzip.decompress(_download(files["train_x"])))
    y_tr_full = _parse_labels(gzip.decompress(_download(files["train_y"])))
    x_te_full = _parse_images(gzip.decompress(_download(files["test_x"])))
    y_te_full = _parse_labels(gzip.decompress(_download(files["test_y"])))

    n_tr = min(train_n, x_tr_full.shape[0])
    n_te = min(test_n, x_te_full.shape[0]) if test_n > 0 else 0
    idx_tr = rng.permutation(x_tr_full.shape[0])[:n_tr]
    x_train = x_tr_full[idx_tr].copy()
    y_train = y_tr_full[idx_tr].copy()
    if n_te <= 0:
        x_test = y_test = None
    else:
        idx_te = rng.permutation(x_te_full.shape[0])[:n_te]
        x_test = x_te_full[idx_te].copy()
        y_test = y_te_full[idx_te].copy()

    flat = bool(${flatten ? "True" : "False"})

    def pack_x(x: np.ndarray | None) -> torch.Tensor | None:
        if x is None:
            return None
        t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to(device)
        if flat:
            t = t.reshape(t.shape[0], -1)
        return t

    xt = pack_x(x_train)
    yt = torch.from_numpy(y_train).to(device).long()
    if x_test is not None and y_test is not None:
        xte = pack_x(x_test)
        yte = torch.from_numpy(y_test).to(device).long()
        test_ds = TensorDataset(xte, yte)
    else:
        test_ds = None
    train_ds = TensorDataset(xt, yt)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size) if test_ds else None
    return train_loader, test_loader
`;
}

function buildShapeWorldLoaders(
  pySym: string,
  title: string,
  trainN: number,
  testN: number,
  seed: number,
  flatten: boolean,
  imageSize: number,
  noiseLevel: number,
): string {
  return `# === ${title} (shape_world_dataset) ===
# 3-class synthetic shapes (square / triangle / circle) on gray background (matches engine).
from matplotlib.path import Path as MplPath

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def fn_${pySym}_loaders(batch_size: int = 64, device: str | torch.device = "cpu"):
    rng = np.random.default_rng(int(${JSON.stringify(seed)}))
    train_n = int(${JSON.stringify(trainN)})
    test_n = int(${JSON.stringify(testN)})
    size = int(max(16, min(96, ${JSON.stringify(imageSize)})))
    noise_level = float(max(0.0, ${JSON.stringify(noiseLevel)}))
    bg = 0.15
    fg = 0.95

    def _draw_disk(img: np.ndarray, cx: float, cy: float, r: float, val: float) -> None:
        h, w = img.shape[-2:]
        yy, xx = np.ogrid[0:h, 0:w]
        m = (xx - cx) ** 2 + (yy - cy) ** 2 <= r**2
        img[..., m] = val

    def _draw_triangle(img: np.ndarray, pts: list[tuple[float, float]], fill: float) -> None:
        h, w = img.shape[-2:]
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        flat = np.column_stack([xx.ravel(), yy.ravel()])
        p = MplPath(pts)
        inside = p.contains_points(flat).reshape(h, w)
        img[..., inside] = fill

    def one() -> tuple[np.ndarray, int]:
        x = np.full((1, size, size), bg, dtype=np.float32)
        cls = int(rng.integers(0, 3))
        margin = size * 0.15
        if cls == 0:
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

    pairs = [one() for _ in range(train_n)]
    x_train = np.stack([p[0] for p in pairs], axis=0).astype(np.float32)
    y_train = np.array([p[1] for p in pairs], dtype=np.int64)
    if test_n <= 0:
        x_test = y_test = None
    else:
        pairs_te = [one() for _ in range(test_n)]
        x_test = np.stack([p[0] for p in pairs_te], axis=0).astype(np.float32)
        y_test = np.array([p[1] for p in pairs_te], dtype=np.int64)

    flat = bool(${flatten ? "True" : "False"})

    def pack_x(x: np.ndarray | None) -> torch.Tensor | None:
        if x is None:
            return None
        t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to(device)
        if flat:
            t = t.reshape(t.shape[0], -1)
        return t

    xt = pack_x(x_train)
    yt = torch.from_numpy(y_train).to(device).long()
    if x_test is not None and y_test is not None:
        xte = pack_x(x_test)
        yte = torch.from_numpy(y_test).to(device).long()
        test_ds = TensorDataset(xte, yte)
    else:
        test_ds = None
    train_ds = TensorDataset(xt, yt)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size) if test_ds else None
    return train_loader, test_loader
`;
}

function buildHoleCountingLoaders(
  pySym: string,
  title: string,
  trainN: number,
  testN: number,
  seed: number,
  flatten: boolean,
  imageSize: number,
  maxHoles: number,
): string {
  return `# === ${title} (hole_counting_dataset) ===
# Binary blob with k circular holes; label k ∈ {0,…,maxHoles} (matches engine).
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def fn_${pySym}_loaders(batch_size: int = 64, device: str | torch.device = "cpu"):
    rng = np.random.default_rng(int(${JSON.stringify(seed)}))
    train_n = int(${JSON.stringify(trainN)})
    test_n = int(${JSON.stringify(testN)})
    size = int(max(24, min(96, ${JSON.stringify(imageSize)})))
    max_holes = int(max(0, min(8, ${JSON.stringify(maxHoles)})))
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

    pairs = [one() for _ in range(train_n)]
    x_train = np.stack([p[0] for p in pairs], axis=0).astype(np.float32)
    y_train = np.array([p[1] for p in pairs], dtype=np.int64)
    if test_n <= 0:
        x_test = y_test = None
    else:
        pairs_te = [one() for _ in range(test_n)]
        x_test = np.stack([p[0] for p in pairs_te], axis=0).astype(np.float32)
        y_test = np.array([p[1] for p in pairs_te], dtype=np.int64)

    flat = bool(${flatten ? "True" : "False"})

    def pack_x(x: np.ndarray | None) -> torch.Tensor | None:
        if x is None:
            return None
        t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to(device)
        if flat:
            t = t.reshape(t.shape[0], -1)
        return t

    xt = pack_x(x_train)
    yt = torch.from_numpy(y_train).to(device).long()
    if x_test is not None and y_test is not None:
        xte = pack_x(x_test)
        yte = torch.from_numpy(y_test).to(device).long()
        test_ds = TensorDataset(xte, yte)
    else:
        test_ds = None
    train_ds = TensorDataset(xt, yt)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size) if test_ds else None
    return train_loader, test_loader
`;
}

/** Runnable loaders mirroring ``vision_datasets_runtime.build_*_arrays`` + optional flatten. */
export function buildVisionDatasetTorch(
  pySym: string,
  title: string,
  kind: VisionDatasetKind,
  raw: Record<string, unknown>,
): string {
  const defs = defaultVisionDatasetData(kind);
  const d = { ...defs, ...(raw as Partial<VisionDatasetNodeData>) } as VisionDatasetNodeData;
  const trainN = Math.max(0, Math.floor(firstScalar(d.trainSize, defs.trainSize as number)));
  const testN = Math.max(0, Math.floor(firstScalar(d.testSize, defs.testSize as number)));
  const seed = Math.floor(firstScalar(d.initSeed ?? d.seed, 0));
  const flatRaw = d.flattenOutput ?? defs.flattenOutput ?? false;
  const flatten =
    typeof flatRaw === "boolean"
      ? flatRaw
      : Array.isArray(flatRaw)
        ? Boolean(flatRaw[0])
        : Boolean(flatRaw);

  if (kind === "gaussian_blob_dataset") {
    const imageSize = Math.floor(firstScalar(d.imageSize, 28));
    const noiseLevel = Number(firstScalar(d.noiseLevel, 0.15));
    const numClasses = Math.floor(firstScalar(d.numClasses, 10));
    return buildGaussianBlobLoaders(pySym, title, trainN, testN, seed, flatten, imageSize, noiseLevel, numClasses);
  }
  if (kind === "mnist_dataset") {
    const cache = String(firstScalar(d.downloadCacheDir, "") ?? "").trim();
    return buildMnistLoaders(pySym, title, trainN, testN, seed, flatten, cache);
  }
  if (kind === "cifar10_dataset") {
    const cache = String(firstScalar(d.downloadCacheDir, "") ?? "").trim();
    return buildCifar10Loaders(pySym, title, trainN, testN, seed, flatten, cache);
  }
  if (kind === "shape_world_dataset") {
    const imageSize = Math.floor(firstScalar(d.imageSize, 32));
    const noiseLevel = Number(firstScalar(d.noiseLevel, 0.04));
    return buildShapeWorldLoaders(pySym, title, trainN, testN, seed, flatten, imageSize, noiseLevel);
  }
  const imageSize = Math.floor(firstScalar(d.imageSize, 48));
  const maxHoles = Math.floor(firstScalar(d.maxHoles, 3));
  return buildHoleCountingLoaders(pySym, title, trainN, testN, seed, flatten, imageSize, maxHoles);
}
