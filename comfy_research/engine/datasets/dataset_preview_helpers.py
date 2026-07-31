"""Dataset preview 共享 builder(自 api/dataset_tensor.py 原体移入)。

移层动机:NodeDef preview provider(nodes 包)绝不 import
api 代码;api 与 providers 都从这里取同一实现。函数体与被移前逐字节一致
(_scalar_* 用 engine.trainer.scalar 等价版;_memorization_class_probs 原体
随迁——api 本地版带 HTTPException 语义,与 synthetic_dataset_builders 版不混用)。
symbolic_func 迁走后，本模块是 direct-arrays 家族唯一的 preview 实现。
"""
from __future__ import annotations

from typing import Any

import numpy as np
from fastapi import HTTPException

from comfy_research.engine.datasets.random_input_distribution_runtime import sample_inputs
from comfy_research.engine.datasets.symbolic_dataset_compile import build_y_numpy_fn
from comfy_research.engine.trainer.scalar import _scalar_float, _scalar_int, _scalar_str
from comfy_research.engine.datasets.teacher_dataset_runtime import teacher_labels_numpy
from comfy_research.engine.runs.trainer_run import (
    _bigram_spectrum_alpha,
    _bigram_spectrum_decay_type,
    _bigram_spectrum_lamb,
    _incoming,
    _memorization_b_vocab_size,
    _memorization_b_xy_numpy,
    _node_map,
    _resolve_teacher_model_eval,
    _sample_teacher_input_tensor,
)
from comfy_research.schemas.graph import Edge, Node, NodeKind


def _memorization_class_probs(output_dim: int, out_dist: str, alpha: float) -> np.ndarray:
    if output_dim < 2:
        raise HTTPException(status_code=400, detail="memorization_a_dataset outputDim must be >= 2.")
    ranks = np.arange(1, output_dim + 1, dtype=np.float64)
    if out_dist == "power_law_class_probs":
        a = max(float(alpha), 1e-8)
        un = np.power(ranks, -a)
    elif out_dist == "exponential_class_probs":
        a = max(float(alpha), 1e-8)
        un = np.exp(-a * ranks)
    else:
        un = np.ones((output_dim,), dtype=np.float64)
    z = float(np.sum(un))
    if not np.isfinite(z) or z <= 0:
        raise HTTPException(status_code=400, detail="Invalid memorization class distribution.")
    return (un / z).astype(np.float64)


def build_linear_or_symbolic_arrays(dataset_type: str, data: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    input_dim = _scalar_int(data.get("inputDim"), 10)
    output_dim = _scalar_int(data.get("outputDim"), 1)
    train_size = max(0, _scalar_int(data.get("trainSize"), 800))
    test_size = max(0, _scalar_int(data.get("testSize"), 200))
    seed = _scalar_int(data.get("seed"), 0)
    input_dist = _scalar_str(data.get("inputDistribution"), "standard_normal")
    input_dist_test = _scalar_str(data.get("inputDistribution"), input_dist)
    out_dist = _scalar_str(data.get("outputDistribution"), "additive_gaussian")
    out_dist_test = _scalar_str(data.get("outputDistribution"), out_dist)
    noise_level = _scalar_float(data.get("noiseLevel"), 0.25)
    noise_level_test = _scalar_float(data.get("noiseLevel"), noise_level)
    additive = out_dist == "additive_gaussian"
    additive_test = out_dist_test == "additive_gaussian"
    sigma = float(noise_level if additive else 0.0)
    sigma_test = float(noise_level_test if additive_test else 0.0)

    rng = np.random.default_rng(seed)
    if dataset_type == "memorization_b_dataset":
        vocab = _memorization_b_vocab_size(data)
        alpha = _scalar_float(data.get("alpha"), 1.0)
        alpha_test = _scalar_float(data.get("alpha"), alpha)
        x_train, y_train = _memorization_b_xy_numpy(rng, train_size, vocab, out_dist, alpha)
        y_train = y_train.astype(np.float32)
        if test_size > 0:
            x_test, y_test = _memorization_b_xy_numpy(rng, test_size, vocab, out_dist_test, alpha_test)
            y_test = y_test.astype(np.float32)
        else:
            x_test = np.zeros((0, vocab), dtype=np.float32)
            y_test = np.zeros((0,), dtype=np.float32)
        return x_train, y_train, x_test, y_test

    x_train = sample_inputs(rng, train_size, input_dim, input_dist)
    x_test = sample_inputs(rng, test_size, input_dim, input_dist_test) if test_size > 0 else np.zeros((0, input_dim), dtype=np.float32)

    if dataset_type == "linear_dataset":
        w = (1.0 / np.sqrt(float(max(input_dim, 1)))) * rng.standard_normal((output_dim, input_dim))
        w = w.astype(np.float32)
        y_train = (x_train @ w.T).astype(np.float32)
        if additive and sigma > 0:
            y_train = y_train + sigma * rng.standard_normal(y_train.shape).astype(np.float32)
        y_test = (x_test @ w.T).astype(np.float32)
        if additive_test and sigma_test > 0 and y_test.size > 0:
            y_test = y_test + sigma_test * rng.standard_normal(y_test.shape).astype(np.float32)
    elif dataset_type == "random_noise_dataset":
        y_train = sample_inputs(rng, train_size, output_dim, input_dist)
        y_test = sample_inputs(rng, test_size, output_dim, input_dist_test) if test_size > 0 else np.zeros((0, output_dim), dtype=np.float32)
    elif dataset_type == "memorization_a_dataset":
        alpha = _scalar_float(data.get("alpha"), 1.0)
        probs = _memorization_class_probs(output_dim, out_dist, alpha)
        y_train = rng.choice(output_dim, size=(train_size,), p=probs).astype(np.float32)
        y_test = rng.choice(output_dim, size=(test_size,), p=probs).astype(np.float32)
    elif dataset_type == "symbolic_func_dataset":
        y_fn = build_y_numpy_fn(data)
        y_train = y_fn(x_train).astype(np.float32)
        if additive and sigma > 0:
            y_train = y_train + sigma * rng.standard_normal(y_train.shape).astype(np.float32)
        y_test = y_fn(x_test).astype(np.float32) if x_test.size > 0 else np.zeros((0, output_dim), dtype=np.float32)
        if additive_test and sigma_test > 0 and y_test.size > 0:
            y_test = y_test + sigma_test * rng.standard_normal(y_test.shape).astype(np.float32)
    else:
        raise HTTPException(
            status_code=400,
            detail=f"dataset_tensor linear/symbolic path does not support dataset_type {dataset_type!r}.",
        )
    return x_train, y_train, x_test, y_test


# ---- token/motion singleton preview（自 api/dataset_tensor.py 移入；
# providers 不 import api,api 与 providers 共用本模块)----
def _build_token_arrays(data: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    vocab = max(2, _scalar_int(data.get("vocabSize"), 4))
    ctx_len = max(1, _scalar_int(data.get("contextLength"), 4))
    retrieval_mode = _scalar_str(data.get("retrievalMode"), "position").strip().lower()
    which = _scalar_int(data.get("whichToken"), -1)
    train_size = max(0, _scalar_int(data.get("trainSize"), 800))
    test_size = max(0, _scalar_int(data.get("testSize"), 200))
    seed = _scalar_int(data.get("seed"), 0)
    if retrieval_mode not in {"position", "content"}:
        raise HTTPException(status_code=400, detail="retrievalMode must be 'position' or 'content'.")
    if retrieval_mode == "position" and not (-ctx_len <= which <= ctx_len - 1):
        raise HTTPException(status_code=400, detail=f"whichToken out of range for contextLength={ctx_len}.")
    if retrieval_mode == "content" and ctx_len < 2:
        raise HTTPException(status_code=400, detail="content retrieval mode requires contextLength >= 2.")
    rng = np.random.default_rng(seed)
    x_train = rng.integers(0, vocab, size=(train_size, ctx_len), dtype=np.int64)
    x_test = rng.integers(0, vocab, size=(test_size, ctx_len), dtype=np.int64) if test_size > 0 else np.zeros((0, ctx_len), dtype=np.int64)
    if retrieval_mode == "content":
        cur_train = x_train[:, -1][:, None]
        prev_train = x_train[:, :-1]
        dist_train = np.abs(prev_train - cur_train)
        idx_train = np.argmin(dist_train, axis=1)
        y_train = prev_train[np.arange(prev_train.shape[0]), idx_train].copy()
        if x_test.size > 0:
            cur_test = x_test[:, -1][:, None]
            prev_test = x_test[:, :-1]
            dist_test = np.abs(prev_test - cur_test)
            idx_test = np.argmin(dist_test, axis=1)
            y_test = prev_test[np.arange(prev_test.shape[0]), idx_test].copy()
        else:
            y_test = np.zeros((0,), dtype=np.int64)
    else:
        y_train = x_train[:, which].copy()
        y_test = x_test[:, which].copy() if x_test.size > 0 else np.zeros((0,), dtype=np.int64)
    return x_train, y_train, x_test, y_test


def _build_circle_random_walk_arrays(data: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    vocab = max(2, _scalar_int(data.get("vocabSize"), 10))
    ctx_len = max(1, _scalar_int(data.get("contextLength"), 1))
    p_right = _scalar_float(data.get("rightStepProb"), 0.5)
    train_size = max(0, _scalar_int(data.get("trainSize"), 800))
    test_size = max(0, _scalar_int(data.get("testSize"), 200))
    seed = _scalar_int(data.get("seed"), 0)
    if p_right < 0.0 or p_right > 1.0:
        raise HTTPException(status_code=400, detail="rightStepProb must be in [0, 1].")

    rng = np.random.default_rng(seed)

    def _sample(n: int) -> tuple[np.ndarray, np.ndarray]:
        if n <= 0:
            return np.zeros((0, ctx_len), dtype=np.int64), np.zeros((0,), dtype=np.int64)
        seq = np.empty((n, ctx_len + 1), dtype=np.int64)
        seq[:, 0] = rng.integers(0, vocab, size=(n,), dtype=np.int64)
        steps = np.where(rng.random((n, ctx_len)) < p_right, 1, -1).astype(np.int64)
        for t in range(ctx_len):
            seq[:, t + 1] = (seq[:, t] + steps[:, t]) % vocab
        return seq[:, :ctx_len].copy(), seq[:, ctx_len].copy()

    x_train, y_train = _sample(train_size)
    x_test, y_test = _sample(test_size)
    return x_train, y_train, x_test, y_test


def _build_circular_motion_arrays(data: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    vocab = max(2, _scalar_int(data.get("vocabSize"), 128))
    ctx_len = max(1, _scalar_int(data.get("contextLength"), 20))
    r_min = _scalar_float(data.get("radiusMin"), 0.15)
    r_max = _scalar_float(data.get("radiusMax"), 0.35)
    omega = _scalar_float(data.get("angularVelocity"), 0.5)
    train_size = max(0, _scalar_int(data.get("trainSize"), 4000))
    test_size = max(0, _scalar_int(data.get("testSize"), 1000))
    seed = _scalar_int(data.get("seed"), 0)
    if r_min < 0.0 or r_max <= 0.0 or r_min > r_max or r_max >= 1.0:
        raise HTTPException(status_code=400, detail="Circular motion dataset requires 0 <= radiusMin <= radiusMax < 1.")

    rng = np.random.default_rng(seed)

    def _coord_to_token(coord: np.ndarray) -> np.ndarray:
        u = (np.clip(coord, -1.0, 1.0 - 1e-8) + 1.0) * 0.5
        return np.floor(vocab * u).astype(np.int64)

    def _sample(n: int) -> tuple[np.ndarray, np.ndarray]:
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

    x_train, y_train = _sample(train_size)
    x_test, y_test = _sample(test_size)
    return x_train, y_train, x_test, y_test


def _build_bigram_low_rank_arrays(data: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    vocab = max(2, _scalar_int(data.get("vocabSize"), 100))
    rank = _scalar_int(data.get("rank"), 20)
    logit_scale = _scalar_float(data.get("logitScale"), 1.0)
    alpha = _bigram_spectrum_alpha(data)
    decay_type = _bigram_spectrum_decay_type(data)
    train_size = max(0, _scalar_int(data.get("trainSize"), 1200))
    test_size = max(0, _scalar_int(data.get("testSize"), 300))
    seed = _scalar_int(data.get("seed"), 0)
    init_seed = _scalar_int(data.get("initSeed"), 0)
    if rank < 1 or rank > vocab:
        raise HTTPException(status_code=400, detail="rank must be in [1, vocabSize].")
    if alpha < 0:
        raise HTTPException(status_code=400, detail="alpha must be >= 0.")

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
            raise HTTPException(status_code=400, detail="Invalid stationary distribution.")
        pi = pi / s

    rng = np.random.default_rng(seed)

    def _sample(n: int) -> tuple[np.ndarray, np.ndarray]:
        if n <= 0:
            return np.zeros((0, 1), dtype=np.int64), np.zeros((0,), dtype=np.int64)
        x = rng.choice(vocab, size=(n,), p=pi).astype(np.int64)
        y = np.empty((n,), dtype=np.int64)
        for i, token in enumerate(x):
            y[i] = rng.choice(vocab, p=probs[int(token)])
        return x[:, None], y

    x_train, y_train = _sample(train_size)
    x_test, y_test = _sample(test_size)
    return x_train, y_train, x_test, y_test


def _build_associative_recall_arrays(data: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    vocab = max(2, _scalar_int(data.get("vocabSize"), 64))
    num_pairs = max(1, _scalar_int(data.get("numPairs"), 32))
    in_context_repeat = max(1, _scalar_int(data.get("inContextRepeat"), 1))
    cross_repeat_prob = _scalar_float(data.get("crossSampleRepeatProb"), 0.0)
    repeated_token_count = _scalar_int(data.get("repeatedTokenCount"), 2)
    train_size = max(0, _scalar_int(data.get("trainSize"), 10_000))
    test_size = max(0, _scalar_int(data.get("testSize"), 2_000))
    seed = _scalar_int(data.get("seed"), 0)
    if cross_repeat_prob < 0.0 or cross_repeat_prob > 1.0:
        raise HTTPException(status_code=400, detail="crossSampleRepeatProb must be in [0, 1].")
    if repeated_token_count < 1 or repeated_token_count > vocab:
        raise HTTPException(status_code=400, detail="repeatedTokenCount must be in [1, vocabSize].")

    ctx_len = 2 * num_pairs + 1
    repeated_pool = np.arange(repeated_token_count, dtype=np.int64)
    rng = np.random.default_rng(seed)

    def _sample(n: int) -> tuple[np.ndarray, np.ndarray]:
        if n <= 0:
            return np.zeros((0, ctx_len), dtype=np.int64), np.zeros((0,), dtype=np.int64)
        x = np.empty((n, ctx_len), dtype=np.int64)
        y = np.empty((n,), dtype=np.int64)
        for i in range(n):
            keys = rng.permutation(vocab)[:num_pairs].astype(np.int64)
            vals = rng.permutation(vocab)[:num_pairs].astype(np.int64)
            query_slot = int(rng.integers(0, num_pairs))
            query = int(keys[query_slot])
            if cross_repeat_prob > 0 and float(rng.random()) < cross_repeat_prob and repeated_pool.size > 0:
                query = int(repeated_pool[int(rng.integers(0, repeated_pool.size))])
                hit = np.where(keys == query)[0]
                if hit.size == 0:
                    query_slot = int(rng.integers(0, num_pairs))
                    keys[query_slot] = query
                else:
                    query_slot = int(hit[0])
            if in_context_repeat > 1:
                candidates = np.arange(num_pairs, dtype=np.int64)
                candidates = candidates[candidates != query_slot]
                if candidates.size > 0:
                    rng.shuffle(candidates)
                    extra = min(in_context_repeat - 1, int(candidates.size))
                    keys[candidates[:extra]] = query
            seq = np.empty((ctx_len,), dtype=np.int64)
            seq[0::2][:num_pairs] = keys
            seq[1::2][:num_pairs] = vals
            seq[-1] = query
            hit_after = np.where(keys == query)[0]
            target = int(vals[int(hit_after[0])]) if hit_after.size > 0 else int(vals[0])
            x[i] = seq
            y[i] = target
        return x, y

    x_train, y_train = _sample(train_size)
    x_test, y_test = _sample(test_size)
    return x_train, y_train, x_test, y_test


def _build_uniform_linear_motion_arrays(data: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pos_dim = max(1, _scalar_int(data.get("positionDim"), 1))
    ctx = max(1, _scalar_int(data.get("contextLength"), 2))
    train_size = max(0, _scalar_int(data.get("trainSize"), 800))
    test_size = max(0, _scalar_int(data.get("testSize"), 200))
    seed = _scalar_int(data.get("seed"), 0)
    pos_dist = _scalar_str(data.get("positionDistribution"), "standard_normal")
    vel_dist = _scalar_str(
        data.get("velocityDistribution"),
        _scalar_str(data.get("x1Distribution"), "standard_normal"),
    )
    vel_scale = _scalar_float(data.get("velocityScale"), 1.0)
    out_dist = _scalar_str(data.get("outputDistribution"), "deterministic")
    noise = _scalar_float(data.get("noiseLevel"), 0.0)
    rng = np.random.default_rng(seed)

    def _sample(n: int) -> tuple[np.ndarray, np.ndarray]:
        if n <= 0:
            z = np.zeros((0, ctx, pos_dim), dtype=np.float32)
            return z, z
        x0 = sample_inputs(rng, n, pos_dim, pos_dist).astype(np.float32)
        v_raw = sample_inputs(rng, n, pos_dim, vel_dist).astype(np.float32)
        v = (float(vel_scale) * v_raw).astype(np.float32)
        idx = np.arange(ctx + 1, dtype=np.float32)
        traj = x0[:, np.newaxis, :] + v[:, np.newaxis, :] * idx[np.newaxis, :, np.newaxis]
        x_in = traj[:, :ctx, :].astype(np.float32)
        y_out = traj[:, 1 : ctx + 1, :].astype(np.float32)
        if out_dist == "additive_gaussian" and noise > 0:
            y_out = y_out.copy()
            y_out[:, -1, :] = y_out[:, -1, :] + noise * rng.standard_normal((n, pos_dim)).astype(np.float32)
        return x_in, y_out

    x_train, y_train = _sample(train_size)
    x_test, y_test = _sample(test_size)
    return x_train, y_train, x_test, y_test


def _build_modular_addition_arrays(data: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    modulus = max(2, _scalar_int(data.get("modulus"), 59))
    train_fraction = _scalar_float(data.get("trainFraction"), 0.3)
    seed = _scalar_int(data.get("seed"), 0)
    if train_fraction <= 0.0 or train_fraction >= 1.0:
        raise HTTPException(status_code=400, detail="trainFraction must be in (0, 1).")
    rng = np.random.default_rng(seed)
    a, b = np.meshgrid(np.arange(modulus, dtype=np.int64), np.arange(modulus, dtype=np.int64), indexing="ij")
    x_all = np.stack([a.reshape(-1), b.reshape(-1)], axis=1)
    y_all = ((x_all[:, 0] + x_all[:, 1]) % modulus).astype(np.int64)
    perm = rng.permutation(x_all.shape[0])
    x_all = x_all[perm]
    y_all = y_all[perm]
    train_size = int(round(train_fraction * x_all.shape[0]))
    train_size = min(max(train_size, 1), x_all.shape[0])
    x_train = x_all[:train_size].copy()
    y_train = y_all[:train_size].copy()
    test_size = int(x_all.shape[0] - train_size)
    if test_size <= 0:
        x_test = np.zeros((0, 2), dtype=np.int64)
        y_test = np.zeros((0,), dtype=np.int64)
    else:
        x_test = x_all[train_size:].copy()
        y_test = y_all[train_size:].copy()
    return x_train, y_train, x_test, y_test


# ---- teacher preview（自 api/dataset_tensor.py 移入）----
def _build_teacher_arrays(
    dataset_node_id: str,
    dataset_data: dict[str, Any],
    raw_nodes: list[dict[str, Any]],
    raw_edges: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(raw_nodes) == 0:
        raise HTTPException(
            status_code=400,
            detail="Teacher dataset tensor preview needs graph context. Reconnect the teacher dataset output and try again.",
        )
    try:
        nodes = [Node.model_validate(n) for n in raw_nodes]
        edges = [Edge.model_validate(e) for e in raw_edges]
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid graph context for teacher dataset preview: {exc}") from exc

    nmap = _node_map(nodes)
    teacher_ds = nmap.get(dataset_node_id)
    if teacher_ds is None:
        teacher_ds = Node(
            id=dataset_node_id,
            type=NodeKind.teacher_dataset,
            data=dataset_data or {},
        )
    if teacher_ds.type != NodeKind.teacher_dataset:
        raise HTTPException(status_code=400, detail="Dataset node is not a teacher_dataset.")

    dd_train: dict[str, Any] = teacher_ds.data or {}
    fallback_seed = _scalar_int(dd_train.get("seed"), 0)

    teacher_src = _incoming(edges, nmap, teacher_ds.id, "model")
    teacher_eval, teacher_in_dim, _ = _resolve_teacher_model_eval(teacher_src, nodes, edges, fallback_seed)
    x_train, train_sampler = _sample_teacher_input_tensor(edges, nmap, teacher_ds, "train_input", None)
    if x_train.shape[1] != teacher_in_dim:
        raise HTTPException(
            status_code=400,
            detail=f"Teacher dataset input dimension mismatch: train input dim={x_train.shape[1]}, teacher inputDim={teacher_in_dim}.",
        )
    y_train = teacher_labels_numpy(teacher_eval, x_train)
    x_test: np.ndarray | None = None
    y_test: np.ndarray | None = None
    test_sampler_wired = _incoming(edges, nmap, teacher_ds.id, "test_input") is not None
    legacy_rid_wired = _incoming(edges, nmap, teacher_ds.id, "input_distribution") is not None
    test_size = max(0, _scalar_int(dd_train.get("testSize"), 0))
    if test_sampler_wired or (legacy_rid_wired and test_size > 0):
        x_test, _ = _sample_teacher_input_tensor(edges, nmap, teacher_ds, "test_input", train_sampler)
        if x_test.shape[1] != teacher_in_dim:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Teacher dataset input dimension mismatch: "
                    f"test input dim={x_test.shape[1]}, teacher inputDim={teacher_in_dim}."
                ),
            )
        y_test = teacher_labels_numpy(teacher_eval, x_test)
    if x_test is None:
        x_test = np.zeros((0, teacher_in_dim), dtype=np.float32)
    if y_test is None:
        y_test = np.zeros((0, y_train.shape[1] if y_train.ndim >= 2 else 1), dtype=np.float32)
    return x_train, y_train, x_test, y_test
