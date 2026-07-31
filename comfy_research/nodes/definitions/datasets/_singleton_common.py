"""Singleton 数据集共享 provider。顶层仅依赖标准库。

- token 系（token_prediction、bigram、circle、circular、recall、modular）的
  materialize 路径继续由 trainer 处理；preview 直接调用 engine 层函数。
- regression 三型（kepler、uniform、symbolic）的 materializer 沿用 dense hook
  之后的实现，不添加任务条件。
- unigram 和 kepler 没有 preview；crl_env_config 没有 provider。
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from comfy_research.nodes.provider_types import DatasetMaterializeContext, DatasetPreviewRequest


def preview_engine_builder(req: "DatasetPreviewRequest"):
    """literal 分支原体:engine 移层的 _build_*_arrays(data) 按 effective_type 分派。"""
    from comfy_research.engine.datasets import dataset_preview_helpers as h

    fn = {
        "token_prediction_dataset": h._build_token_arrays,
        "bigram_low_rank_dataset": h._build_bigram_low_rank_arrays,
        "circle_random_walk_dataset": h._build_circle_random_walk_arrays,
        "circular_motion_dataset": h._build_circular_motion_arrays,
        "in_context_associative_recall_dataset": h._build_associative_recall_arrays,
        "modular_addition_dataset": h._build_modular_addition_arrays,
        "uniform_linear_motion_dataset": h._build_uniform_linear_motion_arrays,
    }[req.effective_type]
    return fn(req.data)


def materialize_kepler_2d(ctx: "DatasetMaterializeContext"):
    from comfy_research.engine.datasets.synthetic_dataset_builders import _build_kepler_2d_arrays
    from comfy_research.engine.trainer.dataset_materialize import _materialize_paired_split

    return _materialize_paired_split(
        lambda: _build_kepler_2d_arrays(ctx.rng, ctx.dd_train, ctx.dd_test, ctx.train_size, ctx.test_size),
        ctx.ds_type, ctx.rng, ctx.dd_train, ctx.dd_test, ctx.train_size, ctx.test_size,
        ctx.input_dim, ctx.output_dim, "Internal: Kepler 2D test tensors missing.",
    )


def materialize_uniform_linear_motion(ctx: "DatasetMaterializeContext"):
    from comfy_research.engine.datasets.synthetic_dataset_builders import _build_uniform_linear_motion_arrays
    from comfy_research.engine.trainer.dataset_materialize import _materialize_paired_split

    return _materialize_paired_split(
        lambda: _build_uniform_linear_motion_arrays(ctx.rng, ctx.dd_train, ctx.dd_test, ctx.train_size, ctx.test_size),
        ctx.ds_type, ctx.rng, ctx.dd_train, ctx.dd_test, ctx.train_size, ctx.test_size,
        ctx.input_dim, ctx.output_dim, "Internal: uniform motion test tensors missing.",
    )


def materialize_symbolic_func(ctx: "DatasetMaterializeContext"):
    """arm 原体 + 标量解析行原样(linear 先例)。"""
    from comfy_research.engine.trainer.dataset_materialize import _materialize_symbolic_func
    from comfy_research.engine.trainer.scalar import _scalar_float, _scalar_str

    dd_train, dd_test = ctx.dd_train, ctx.dd_test
    input_dist = _scalar_str(dd_train.get("inputDistribution"), "standard_normal")
    out_dist = _scalar_str(dd_train.get("outputDistribution"), "additive_gaussian")
    noise_level = _scalar_float(dd_train.get("noiseLevel"), 0.25)
    additive = out_dist == "additive_gaussian"
    sigma = noise_level if additive else 0.0
    input_dist_test = _scalar_str(dd_test.get("inputDistribution"), input_dist)
    out_dist_test = _scalar_str(dd_test.get("outputDistribution"), out_dist)
    noise_test = _scalar_float(dd_test.get("noiseLevel"), noise_level)
    additive_test = out_dist_test == "additive_gaussian"
    sigma_test = noise_test if additive_test else 0.0
    return _materialize_symbolic_func(
        ctx.rng, dd_train, dd_test, ctx.ds_test_raw, ctx.train_size, ctx.test_size,
        ctx.input_dim, ctx.output_dim,
        input_dist, sigma, additive, input_dist_test, sigma_test, additive_test,
    )
