"""Pure synthetic-dataset array builders extracted from trainer_run.

Numpy-focused; these raise ``fastapi.HTTPException`` on invalid config and one
builder calls ``sample_inputs``. Stateless free functions, no torch, no
trainer runtime. This module MUST NOT import ``trainer_run`` (one-way edge).
"""
from __future__ import annotations

from typing import Any, Literal

import numpy as np
from fastapi import HTTPException

from .random_input_distribution_runtime import sample_inputs


def _scalar_int(x: Any, default: int = 0) -> int:
    if isinstance(x, list):
        if not x:
            return default
        try:
            return int(x[0])
        except (TypeError, ValueError):
            return default
    try:
        return int(x)
    except (TypeError, ValueError):
        return default


def _scalar_float(x: Any, default: float = 0.0) -> float:
    if isinstance(x, list):
        if not x:
            return default
        try:
            return float(x[0])
        except (TypeError, ValueError):
            return default
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _scalar_str(x: Any, default: str = "") -> str:
    if isinstance(x, list):
        if not x:
            return default
        return str(x[0])
    return str(x) if x is not None else default


def _memorization_class_probs(output_dim: int, out_dist: str, alpha: float) -> np.ndarray:
    if output_dim < 2:
        raise HTTPException(
            status_code=400,
            detail=(
                "memorization_a_dataset with cross_entropy_loss needs output dim (num classes) >= 2 "
                f"(got output_dim={output_dim})."
            ),
        )
    ranks = np.arange(1, output_dim + 1, dtype=np.float64)
    if out_dist == "power_law_class_probs":
        a = max(float(alpha), 1e-8)
        un = np.power(ranks, -a)
    elif out_dist == "exponential_class_probs":
        a = max(float(alpha), 1e-8)
        un = np.exp(-a * ranks)
    else:
        # Default and explicit "uniform_class_probs".
        un = np.ones((output_dim,), dtype=np.float64)
    z = float(np.sum(un))
    if not np.isfinite(z) or z <= 0:
        raise HTTPException(status_code=400, detail="Invalid memorization class distribution.")
    return (un / z).astype(np.float64)


def _unigram_token_probs(vocab_size: int, out_dist: str, alpha: float) -> np.ndarray:
    """Token probabilities over ids ``0 .. V-1`` from ranks ``1 .. V`` (same shapes as memorization class priors)."""
    ranks = np.arange(1, vocab_size + 1, dtype=np.float64)
    if out_dist == "power_law_class_probs":
        a = max(float(alpha), 1e-8)
        un = np.power(ranks, -a)
    elif out_dist == "exponential_class_probs":
        a = max(float(alpha), 1e-8)
        un = np.exp(-a * ranks)
    elif out_dist == "uniform_class_probs":
        un = np.ones((vocab_size,), dtype=np.float64)
    else:
        raise HTTPException(
            status_code=400,
            detail=(
                "Unigram dataset outputDistribution must be one of "
                "uniform_class_probs, power_law_class_probs, exponential_class_probs."
            ),
        )
    z = float(np.sum(un))
    if not np.isfinite(z) or z <= 0:
        raise HTTPException(status_code=400, detail="Invalid unigram token distribution.")
    return (un / z).astype(np.float64)


def _sample_memorization_labels(
    rng: np.random.Generator, n: int, output_dim: int, out_dist: str, alpha: float
) -> np.ndarray:
    probs = _memorization_class_probs(output_dim, out_dist, alpha)
    return rng.choice(output_dim, size=(int(n),), p=probs).astype(np.int64)


def _memorization_b_vocab_size(data: dict[str, Any], default: int = 40) -> int:
    """Unified vocabulary size for memorization B (shared by input and output classes)."""
    v = _scalar_int(data.get("vocabSize"), 0)
    if v >= 2:
        return v
    d_in = _scalar_int(data.get("inputDim"), default)
    d_out = _scalar_int(data.get("outputDim"), d_in)
    if d_in != d_out:
        raise HTTPException(
            status_code=400,
            detail=(
                "memorization_b_dataset uses one vocabulary size for both endpoints; "
                f"got inputDim={d_in}, outputDim={d_out}. Set vocabSize (recommended) or make them equal."
            ),
        )
    if d_in < 2:
        raise HTTPException(
            status_code=400,
            detail=f"memorization_b_dataset vocabSize must be >= 2 (got {d_in}).",
        )
    return int(d_in)


def _memorization_b_xy_numpy(
    rng: np.random.Generator,
    n: int,
    vocab_size: int,
    class_dist: str,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Memorization B: random input class (one-hot rows) and independent random output class labels."""
    if vocab_size < 2:
        raise HTTPException(
            status_code=400,
            detail=(
                "memorization_b_dataset needs vocab size (number of classes) >= 2 "
                f"(got vocab_size={vocab_size})."
            ),
        )
    in_probs = _memorization_class_probs(vocab_size, class_dist, alpha)
    x_idx = rng.choice(vocab_size, size=(int(n),), p=in_probs).astype(np.int64)
    x_np = np.zeros((int(n), int(vocab_size)), dtype=np.float32)
    if int(n) > 0:
        x_np[np.arange(int(n), dtype=np.int64), x_idx] = 1.0
    y_np = _sample_memorization_labels(rng, int(n), vocab_size, class_dist, alpha)
    return x_np, y_np


def _bigram_spectrum_alpha(data: dict[str, Any], *, fallback: float | None = None) -> float:
    """Spectrum weight scale ``alpha``; legacy ``rankDecay`` if ``alpha`` absent; optional ``fallback`` (e.g. train) for test split."""
    if "alpha" in data:
        return _scalar_float(data.get("alpha"), 0.0)
    if "rankDecay" in data:
        return _scalar_float(data.get("rankDecay"), 0.0)
    if fallback is not None:
        return float(fallback)
    return 0.0


def _bigram_spectrum_decay_type(data: dict[str, Any]) -> Literal["power_law", "exponential"]:
    raw = data.get("decayType")
    if raw is None:
        return "power_law"
    s = str(_scalar_str(raw, "")).strip().lower().replace("-", "_")
    if s in ("power_law", "powerlaw"):
        return "power_law"
    if s in ("exponential", "exp"):
        return "exponential"
    raise HTTPException(
        status_code=400,
        detail="Bigram low-rank dataset decayType must be 'power_law' or 'exponential'.",
    )


def _bigram_spectrum_lamb(rank: int, alpha: float, decay_type: str) -> np.ndarray:
    """Per-rank weights ``λ_n`` for columns of ``A`` in ``logits = (A * λ) @ B`` with ``n = 1..R``."""
    if alpha == 0.0:
        return np.ones((rank,), dtype=np.float64)
    n = np.arange(1, rank + 1, dtype=np.float64)
    if decay_type == "exponential":
        return np.exp(-alpha * n)
    return np.power(n, -alpha)


def _build_bigram_low_rank_arrays(
    train_data: dict[str, Any],
    test_data: dict[str, Any],
    train_size: int,
    test_size: int,
    *,
    train_pair_rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, int]:
    vocab = _scalar_int(train_data.get("vocabSize"), 100)
    rank = _scalar_int(train_data.get("rank"), 20)
    logit_scale = _scalar_float(train_data.get("logitScale"), 1.0)
    corrupt_ratio = _scalar_float(train_data.get("corruptRatio"), 0.0)
    corrupt_scale = _scalar_float(train_data.get("corruptScale"), 5.0)
    alpha = _bigram_spectrum_alpha(train_data)
    decay_type = _bigram_spectrum_decay_type(train_data)
    init_seed = _scalar_int(train_data.get("initSeed"), 0)
    if vocab < 2:
        raise HTTPException(status_code=400, detail="Bigram low-rank dataset vocabSize must be >= 2.")
    if rank < 1 or rank > vocab:
        raise HTTPException(status_code=400, detail="Bigram low-rank dataset rank must be in [1, vocabSize].")
    if alpha < 0:
        raise HTTPException(status_code=400, detail="Bigram low-rank dataset alpha must be >= 0.")
    if corrupt_ratio < 0.0 or corrupt_ratio > 1.0:
        raise HTTPException(status_code=400, detail="Bigram low-rank dataset corruptRatio must be in [0, 1].")
    if corrupt_scale < 0.0:
        raise HTTPException(status_code=400, detail="Bigram low-rank dataset corruptScale must be >= 0.")

    vocab_te = _scalar_int(test_data.get("vocabSize"), vocab)
    rank_te = _scalar_int(test_data.get("rank"), rank)
    logit_scale_te = _scalar_float(test_data.get("logitScale"), logit_scale)
    corrupt_ratio_te = _scalar_float(test_data.get("corruptRatio"), corrupt_ratio)
    corrupt_scale_te = _scalar_float(test_data.get("corruptScale"), corrupt_scale)
    alpha_te = _bigram_spectrum_alpha(test_data, fallback=alpha)
    raw_decay_te = test_data.get("decayType")
    if raw_decay_te is None:
        decay_type_te = decay_type
    else:
        decay_type_te = _bigram_spectrum_decay_type(test_data)
    init_seed_te = _scalar_int(test_data.get("initSeed"), init_seed)
    if (
        vocab_te != vocab
        or rank_te != rank
        or logit_scale_te != logit_scale
        or corrupt_ratio_te != corrupt_ratio
        or corrupt_scale_te != corrupt_scale
        or alpha_te != alpha
        or decay_type_te != decay_type
        or init_seed_te != init_seed
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "Test bigram low-rank dataset must match train for vocabSize, rank, "
                "logitScale, corruptRatio, corruptScale, alpha, decayType, and initSeed."
            ),
        )

    init_rng = np.random.default_rng(init_seed)
    A = init_rng.standard_normal((vocab, rank)).astype(np.float64)
    B = init_rng.standard_normal((rank, vocab)).astype(np.float64)
    lamb = _bigram_spectrum_lamb(rank, alpha, decay_type)
    logits = (A * lamb[None, :]) @ B
    logits = (logit_scale * logits) / np.sqrt(float(max(rank, 1)))
    row_max = np.max(logits, axis=1, keepdims=True)
    probs = np.exp(logits - row_max)
    probs = probs / np.sum(probs, axis=1, keepdims=True)

    pi = np.full((vocab,), 1.0 / float(vocab), dtype=np.float64)
    for _ in range(256):
        pi = pi @ probs
        s = float(np.sum(pi))
        if s <= 0 or not np.isfinite(s):
            raise HTTPException(status_code=400, detail="Invalid stationary distribution for bigram low-rank dataset.")
        pi = pi / s

    def _sample_pairs(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        x = rng.choice(vocab, size=(n,), p=pi).astype(np.int64)
        y = np.empty((n,), dtype=np.int64)
        for i, token in enumerate(x):
            p = probs[int(token)]
            if corrupt_ratio > 0.0 and rng.random() < corrupt_ratio:
                noisy_logits = rng.standard_normal((vocab,)).astype(np.float64) * corrupt_scale
                noisy_logits = noisy_logits - np.max(noisy_logits)
                noisy = np.exp(noisy_logits)
                noisy = noisy / np.sum(noisy)
                p = noisy
            y[i] = rng.choice(vocab, p=p)
        return x[:, None], y

    seed = _scalar_int(train_data.get("seed"), 0)
    sample_rng = train_pair_rng if train_pair_rng is not None else np.random.default_rng(seed)
    x_train, y_train = _sample_pairs(train_size, sample_rng)
    x_test: np.ndarray | None = None
    y_test: np.ndarray | None = None
    if test_size > 0:
        x_test, y_test = _sample_pairs(test_size, sample_rng)
    return x_train, y_train, x_test, y_test, vocab


def _build_circular_motion_arrays(
    train_data: dict[str, Any],
    test_data: dict[str, Any],
    train_size: int,
    test_size: int,
    *,
    train_sample_rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, int, int]:
    vocab = _scalar_int(train_data.get("vocabSize"), 128)
    ctx_len = _scalar_int(train_data.get("contextLength"), 20)
    r_min = _scalar_float(train_data.get("radiusMin"), 0.15)
    r_max = _scalar_float(train_data.get("radiusMax"), 0.35)
    omega = _scalar_float(train_data.get("angularVelocity"), 0.5)
    if vocab < 2:
        raise HTTPException(status_code=400, detail="Circular motion dataset vocabSize must be >= 2.")
    if ctx_len < 1:
        raise HTTPException(status_code=400, detail="Circular motion dataset contextLength must be >= 1.")
    if r_min < 0.0 or r_max <= 0.0 or r_min > r_max or r_max >= 1.0:
        raise HTTPException(status_code=400, detail="Circular motion dataset requires 0 <= radiusMin <= radiusMax < 1.")

    vocab_te = _scalar_int(test_data.get("vocabSize"), vocab)
    ctx_te = _scalar_int(test_data.get("contextLength"), ctx_len)
    rmin_te = _scalar_float(test_data.get("radiusMin"), r_min)
    rmax_te = _scalar_float(test_data.get("radiusMax"), r_max)
    omega_te = _scalar_float(test_data.get("angularVelocity"), omega)
    if vocab_te != vocab or ctx_te != ctx_len or rmin_te != r_min or rmax_te != r_max or omega_te != omega:
        raise HTTPException(
            status_code=400,
            detail=(
                "Test circular motion dataset must match train for vocabSize, contextLength, "
                "radiusMin, radiusMax, and angularVelocity."
            ),
        )

    seed = _scalar_int(train_data.get("seed"), 0)
    sample_rng = train_sample_rng if train_sample_rng is not None else np.random.default_rng(seed)

    def _coord_to_token(coord: np.ndarray) -> np.ndarray:
        u = (np.clip(coord, -1.0, 1.0 - 1e-8) + 1.0) * 0.5
        return np.floor(vocab * u).astype(np.int64)

    def _sample_rows(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        if n <= 0:
            return np.zeros((0, ctx_len, 2), dtype=np.int64), np.zeros((0, 2), dtype=np.int64)
        radii = rng.uniform(r_min, r_max, size=(n,)).astype(np.float64)
        phase0 = rng.uniform(0.0, 2.0 * np.pi, size=(n,)).astype(np.float64)
        x_ctx = np.empty((n, ctx_len), dtype=np.float64)
        y_ctx = np.empty((n, ctx_len), dtype=np.float64)
        x_nxt = np.empty((n,), dtype=np.float64)
        y_nxt = np.empty((n,), dtype=np.float64)
        for t in range(ctx_len):
            phi = phase0 + omega * t
            x_ctx[:, t] = radii * np.cos(phi)
            y_ctx[:, t] = radii * np.sin(phi)
        phi_n = phase0 + omega * ctx_len
        x_nxt[:] = radii * np.cos(phi_n)
        y_nxt[:] = radii * np.sin(phi_n)
        pairs = np.stack([_coord_to_token(x_ctx), _coord_to_token(y_ctx)], axis=-1)
        y_pair = np.stack([_coord_to_token(x_nxt), _coord_to_token(y_nxt)], axis=-1)
        return pairs, y_pair

    x_train, y_train = _sample_rows(train_size, sample_rng)
    x_test: np.ndarray | None = None
    y_test: np.ndarray | None = None
    if test_size > 0:
        x_test, y_test = _sample_rows(test_size, sample_rng)
    return x_train, y_train, x_test, y_test, vocab, ctx_len


def _random_linear_weights(
    rng: np.random.Generator,
    input_dim: int,
    output_dim: int,
) -> np.ndarray:
    """W with shape (output_dim, input_dim), W_ij ~ N(0,1) / sqrt(input_dim)."""
    if output_dim < 1:
        raise HTTPException(status_code=400, detail="Output dimension must be >= 1")
    scale = 1.0 / np.sqrt(float(input_dim))
    return (scale * rng.standard_normal((output_dim, input_dim))).astype(np.float32)


def _linear_regression_weight_rng(dd: dict[str, Any]) -> np.random.Generator:
    """RNG for linear_dataset oracle weights only, independent of the trainer stream.

    Train and test splits must share the same W (standalone trainer path reuses ``w`` for the
    test split). dataset_mixer_b used to resample W per split and also consumed branch-B RNG
    even when lambda=1, which desynchronized curves from wiring dataset A alone.
    """
    w_seed = _scalar_int(dd.get("seed"), 0)
    return np.random.default_rng(w_seed)


def _apply_linear_map(
    x: np.ndarray,
    w: np.ndarray,
    noise_sigma: float,
    additive_noise: bool,
    rng: np.random.Generator,
) -> np.ndarray:
    """y = x @ W.T + sigma * epsilon, epsilon ~ N(0, I)."""
    n, d_in = x.shape
    o, d_w = w.shape
    if d_in != d_w or o < 1:
        raise HTTPException(status_code=400, detail="Weight matrix shape does not match inputs")
    y = (x @ w.T).astype(np.float32)
    if additive_noise and noise_sigma > 0:
        y = y + noise_sigma * rng.standard_normal((n, o)).astype(np.float32)
    return y


def _linear_like_targets_from_x(
    dd: dict[str, Any],
    x_np: np.ndarray,
    input_dim: int,
    output_dim: int,
    *,
    sigma: float,
    additive: bool,
    rng: np.random.Generator,
) -> np.ndarray:
    target_kind = _scalar_str(dd.get("targetKind"), "linear")
    if target_kind == "xor2d":
        if input_dim != 2 or output_dim != 1:
            raise HTTPException(
                status_code=400,
                detail="linear_dataset targetKind xor2d requires inputDim=2 and outputDim=1.",
            )
        y_np = ((x_np[:, 0] > 0.0) ^ (x_np[:, 1] > 0.0)).astype(np.float32).reshape(-1, 1)
        if additive and sigma > 0:
            y_np = y_np + sigma * rng.standard_normal(y_np.shape).astype(np.float32)
        return y_np
    w = _random_linear_weights(_linear_regression_weight_rng(dd), input_dim, output_dim)
    return _apply_linear_map(x_np, w, sigma, additive, rng)


def _build_uniform_linear_motion_arrays(
    rng: np.random.Generator,
    dd_train: dict[str, Any],
    dd_test: dict[str, Any],
    train_size: int,
    test_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    pos_dim = _scalar_int(dd_train.get("positionDim"), 1)
    if pos_dim < 1:
        raise HTTPException(status_code=400, detail="Uniform linear motion dataset positionDim must be >= 1.")
    ctx = max(1, _scalar_int(dd_train.get("contextLength"), 2))
    pos_dist = _scalar_str(dd_train.get("positionDistribution"), "standard_normal")
    vel_dist = _scalar_str(
        dd_train.get("velocityDistribution"),
        _scalar_str(dd_train.get("x1Distribution"), "standard_normal"),
    )
    vel_scale = _scalar_float(dd_train.get("velocityScale"), 1.0)
    out_dist = _scalar_str(dd_train.get("outputDistribution"), "deterministic")
    noise = _scalar_float(dd_train.get("noiseLevel"), 0.0)
    additive = out_dist == "additive_gaussian"

    def _sample_one(
        n: int, dist_x0: str, dist_v: str, v_scale: float, add_noise: bool, sigma: float
    ) -> tuple[np.ndarray, np.ndarray]:
        if n <= 0:
            z = np.zeros((0, ctx, pos_dim), dtype=np.float32)
            return z, z
        x0 = sample_inputs(rng, n, pos_dim, dist_x0).astype(np.float32)
        v_raw = sample_inputs(rng, n, pos_dim, dist_v).astype(np.float32)
        v = (float(v_scale) * v_raw).astype(np.float32)
        idx = np.arange(ctx + 1, dtype=np.float32)
        traj = x0[:, np.newaxis, :] + v[:, np.newaxis, :] * idx[np.newaxis, :, np.newaxis]
        x_in = traj[:, :ctx, :].astype(np.float32)
        y_out = traj[:, 1 : ctx + 1, :].astype(np.float32)
        if add_noise and sigma > 0:
            y_out = y_out.copy()
            y_out[:, -1, :] = y_out[:, -1, :] + sigma * rng.standard_normal((n, pos_dim)).astype(np.float32)
        return x_in, y_out

    x_np, y_np = _sample_one(train_size, pos_dist, vel_dist, vel_scale, additive, noise if additive else 0.0)
    x_test_np: np.ndarray | None = None
    y_test_np: np.ndarray | None = None
    if test_size > 0:
        pos_dist_te = _scalar_str(dd_test.get("positionDistribution"), pos_dist)
        vel_dist_te = _scalar_str(
            dd_test.get("velocityDistribution"),
            _scalar_str(dd_test.get("x1Distribution"), vel_dist),
        )
        vel_scale_te = _scalar_float(dd_test.get("velocityScale"), vel_scale)
        ctx_te = max(1, _scalar_int(dd_test.get("contextLength"), ctx))
        if ctx_te != ctx:
            raise HTTPException(
                status_code=400,
                detail="Train/test uniform_linear_motion_dataset contextLength must match.",
            )
        out_dist_te = _scalar_str(dd_test.get("outputDistribution"), out_dist)
        noise_te = _scalar_float(dd_test.get("noiseLevel"), noise)
        add_te = out_dist_te == "additive_gaussian"
        x_test_np, y_test_np = _sample_one(
            test_size, pos_dist_te, vel_dist_te, vel_scale_te, add_te, noise_te if add_te else 0.0
        )
    return x_np, y_np, x_test_np, y_test_np


def _build_kepler_2d_arrays(
    rng: np.random.Generator,
    dd_train: dict[str, Any],
    dd_test: dict[str, Any],
    train_size: int,
    test_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    ctx = max(1, _scalar_int(dd_train.get("contextLength"), 8))
    a_min = _scalar_float(dd_train.get("semiMajorAxisMin"), 0.7)
    a_max = _scalar_float(dd_train.get("semiMajorAxisMax"), 1.3)
    e_min = _scalar_float(dd_train.get("eccentricityMin"), 0.0)
    e_max = _scalar_float(dd_train.get("eccentricityMax"), 0.55)
    mean_motion = _scalar_float(dd_train.get("meanMotion"), 0.4)
    out_dist = _scalar_str(dd_train.get("outputDistribution"), "deterministic")
    noise = _scalar_float(dd_train.get("noiseLevel"), 0.0)
    additive = out_dist == "additive_gaussian"
    if not (a_min > 0 and a_max >= a_min):
        raise HTTPException(status_code=400, detail="kepler_2d_dataset requires 0 < semiMajorAxisMin <= semiMajorAxisMax.")
    if e_min < 0 or e_max < e_min or e_max >= 0.999:
        raise HTTPException(status_code=400, detail="kepler_2d_dataset requires 0 <= eccentricityMin <= eccentricityMax < 0.999.")
    if mean_motion <= 0:
        raise HTTPException(status_code=400, detail="kepler_2d_dataset meanMotion must be > 0.")

    def _solve_kepler(mean_anomaly: np.ndarray, ecc: np.ndarray, n_iter: int = 7) -> np.ndarray:
        e_anom = mean_anomaly.copy()
        for _ in range(n_iter):
            f = e_anom - ecc * np.sin(e_anom) - mean_anomaly
            fp = 1.0 - ecc * np.cos(e_anom)
            e_anom = e_anom - f / np.clip(fp, 1e-6, None)
        return e_anom

    def _sample_rows(
        n: int,
        *,
        a_min_v: float,
        a_max_v: float,
        e_min_v: float,
        e_max_v: float,
        mean_motion_v: float,
        add_noise: bool,
        sigma: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        if n <= 0:
            z = np.zeros((0, ctx, 2), dtype=np.float32)
            return z, z
        a = rng.uniform(a_min_v, a_max_v, size=(n,)).astype(np.float64)
        e = rng.uniform(e_min_v, e_max_v, size=(n,)).astype(np.float64)
        phase = rng.uniform(0.0, 2.0 * np.pi, size=(n,)).astype(np.float64)
        m0 = rng.uniform(0.0, 2.0 * np.pi, size=(n,)).astype(np.float64)
        m_steps = m0[:, None] + float(mean_motion_v) * np.arange(ctx + 1, dtype=np.float64)[None, :]
        e_mat = np.repeat(e[:, None], ctx + 1, axis=1)
        e_anom = _solve_kepler(m_steps, e_mat)
        cos_e = np.cos(e_anom)
        sin_e = np.sin(e_anom)
        fac = np.sqrt(np.clip(1.0 - e_mat * e_mat, 1e-8, None))
        x_orb = a[:, None] * (cos_e - e_mat)
        y_orb = a[:, None] * fac * sin_e
        cp = np.cos(phase)[:, None]
        sp = np.sin(phase)[:, None]
        x_rot = cp * x_orb - sp * y_orb
        y_rot = sp * x_orb + cp * y_orb
        pts = np.stack([x_rot, y_rot], axis=-1).astype(np.float32)
        x_np = pts[:, :ctx, :]
        y_np = pts[:, 1 : ctx + 1, :]
        if add_noise and sigma > 0:
            y_np = y_np.copy()
            y_np[:, -1, :] = y_np[:, -1, :] + sigma * rng.standard_normal((n, 2)).astype(np.float32)
        return x_np, y_np

    x_np, y_np = _sample_rows(
        train_size,
        a_min_v=a_min,
        a_max_v=a_max,
        e_min_v=e_min,
        e_max_v=e_max,
        mean_motion_v=mean_motion,
        add_noise=additive,
        sigma=noise if additive else 0.0,
    )
    x_test_np: np.ndarray | None = None
    y_test_np: np.ndarray | None = None
    if test_size > 0:
        ctx_te = max(1, _scalar_int(dd_test.get("contextLength"), ctx))
        if ctx_te != ctx:
            raise HTTPException(status_code=400, detail="Train/test kepler_2d_dataset contextLength must match.")
        a_min_te = _scalar_float(dd_test.get("semiMajorAxisMin"), a_min)
        a_max_te = _scalar_float(dd_test.get("semiMajorAxisMax"), a_max)
        e_min_te = _scalar_float(dd_test.get("eccentricityMin"), e_min)
        e_max_te = _scalar_float(dd_test.get("eccentricityMax"), e_max)
        mean_motion_te = _scalar_float(dd_test.get("meanMotion"), mean_motion)
        out_dist_te = _scalar_str(dd_test.get("outputDistribution"), out_dist)
        noise_te = _scalar_float(dd_test.get("noiseLevel"), noise)
        add_te = out_dist_te == "additive_gaussian"
        x_test_np, y_test_np = _sample_rows(
            test_size,
            a_min_v=a_min_te,
            a_max_v=a_max_te,
            e_min_v=e_min_te,
            e_max_v=e_max_te,
            mean_motion_v=mean_motion_te,
            add_noise=add_te,
            sigma=noise_te if add_te else 0.0,
        )
    return x_np, y_np, x_test_np, y_test_np
