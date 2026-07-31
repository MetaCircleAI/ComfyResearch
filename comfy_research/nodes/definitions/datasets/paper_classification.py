"""Dense synthetic classification data for Rank Collapse and Slingshot.

This node intentionally exposes no modular-addition or Grokking mode.
"""
from __future__ import annotations

from comfy_research.nodes.registry import dataset_def, dataset_materializer_for
from comfy_research.nodes.schema import DatasetDef, EnumField, FloatField, FrontendSpec, IntField

EXPERIMENT_MODES: tuple[str, ...] = (
    "rank_figure5_main",
    "rank_figure5_uniform",
    "rank_figure5_mse",
    "rank_orthogonal_scaleup_ce",
    "rank_orthogonal_scaleup_mse",
    # Historical V=16 scale-up modes remain loadable for old graphs, but are
    # not used by the formal Figure-5 template or its evidence contract.
    "rank_collapse_skewed",
    "rank_uniform",
    "rank_mse_control",
    "slingshot_hypercube",
)

FAMILY: tuple[str, ...] = (
    "dense_classification_dataset",
    "canvas_dataset_source",
    "canvas_trainer_autoconnect_dataset",
)

PAPER_CLASSIFICATION = dataset_def(
    DatasetDef(
        type="paper_classification_dataset",
        label="Paper classification dataset",
        hint=(
            "Dense toy tasks for Rank Collapse (fixed Figure-5 samples, scalable X=I_N "
            "orthogonal CE/MSE protocols, plus legacy repeated-prototype controls) and "
            "Slingshot (8-class noisy hypercube); no Grokking task is used."
        ),
        family=FAMILY,
        fields=(
            EnumField(
                key="experimentMode",
                label="Experiment Mode",
                default="rank_collapse_skewed",
                options=EXPERIMENT_MODES,
            ),
            IntField(key="inputDim", label="Input Dim", default=16, min=3),
            IntField(key="outputDim", label="Output Dim", default=16, min=2),
            IntField(key="trainSize", label="Train Size", default=1029, min=1),
            IntField(key="testSize", label="Test Size", default=0, min=0),
            FloatField(key="frequencyRatio", label="Frequency Ratio", default=2.0, min=1.0),
            FloatField(key="classSeparation", label="Class Separation", default=1.0, min=0.0),
            IntField(key="seed", label="Seed", default=0),
            EnumField(
                key="samplingMode",
                label="Sampling Mode",
                default="fixed",
                options=("fixed",),
            ),
        ),
        frontend=FrontendSpec(
            component_key="GenericDatasetNode",
            codegen_key="paper_classification_dataset",
        ),
    )
)


@dataset_materializer_for(PAPER_CLASSIFICATION)
def materialize_paper_classification(ctx):
    import numpy as np
    from fastapi import HTTPException

    from comfy_research.engine.datasets.paper_classification import (
        rank_classification_arrays,
        rank_figure5_arrays,
        rank_orthogonal_scaleup_arrays,
        slingshot_hypercube_splits,
    )
    from comfy_research.engine.trainer.dataset_arrays import DatasetArrays
    from comfy_research.engine.trainer.scalar import _scalar_float, _scalar_int, _scalar_str

    mode = _scalar_str(ctx.dd_train.get("experimentMode"), "rank_collapse_skewed")
    try:
        sampling_mode = _scalar_str(ctx.dd_train.get("samplingMode"), "fixed")
        if sampling_mode != "fixed":
            raise ValueError("paper_classification_dataset currently supports samplingMode='fixed' only")
        if mode in ("rank_figure5_main", "rank_figure5_uniform", "rank_figure5_mse"):
            expected_train_size = 8 if mode == "rank_figure5_uniform" else 6
            expected_input_dim = expected_train_size
            if (
                _scalar_int(ctx.dd_train.get("inputDim"), expected_input_dim)
                != expected_input_dim
                or _scalar_int(ctx.dd_train.get("outputDim"), 4) != 4
                or ctx.train_size != expected_train_size
                or ctx.test_size != 0
                or ctx.input_dim != expected_input_dim
                or ctx.output_dim != 4
            ):
                raise ValueError(
                    f"{mode} is the fixed Figure-5 protocol and requires "
                    f"inputDim={expected_input_dim}, outputDim=4, "
                    f"trainSize={expected_train_size}, testSize=0"
                )
            x_np, y_np = rank_figure5_arrays(mode)
            x_test_np = y_test_np = None
        elif mode in ("rank_orthogonal_scaleup_ce", "rank_orthogonal_scaleup_mse"):
            if ctx.input_dim != ctx.train_size or ctx.test_size != 0:
                raise ValueError(
                    f"{mode} is an all-orthogonal protocol and requires "
                    f"inputDim=trainSize and testSize=0; got "
                    f"inputDim={ctx.input_dim}, trainSize={ctx.train_size}, "
                    f"testSize={ctx.test_size}"
                )
            x_np, y_np = rank_orthogonal_scaleup_arrays(
                sample_count=ctx.train_size,
                class_count=ctx.output_dim,
                frequency_ratio=_scalar_float(ctx.dd_train.get("frequencyRatio"), 2.0),
                mse=mode == "rank_orthogonal_scaleup_mse",
            )
            x_test_np = y_test_np = None
        elif mode == "slingshot_hypercube":
            test_mode = _scalar_str(ctx.dd_test.get("experimentMode"), mode)
            if ctx.test_size > 0 and test_mode != mode:
                raise ValueError("slingshot train/test splits must both use slingshot_hypercube")
            x_np, y_np, x_test_np, y_test_np = slingshot_hypercube_splits(
                ctx.rng,
                train_size=ctx.train_size,
                test_size=ctx.test_size,
                input_dim=ctx.input_dim,
                class_count=ctx.output_dim,
                class_separation=_scalar_float(ctx.dd_train.get("classSeparation"), 1.0),
            )
        elif mode in ("rank_collapse_skewed", "rank_uniform", "rank_mse_control"):
            x_np, labels_np = rank_classification_arrays(
                ctx.rng,
                sample_count=ctx.train_size,
                input_dim=ctx.input_dim,
                class_count=ctx.output_dim,
                frequency_ratio=_scalar_float(ctx.dd_train.get("frequencyRatio"), 2.0),
                uniform=mode == "rank_uniform",
            )
            y_np = (
                np.eye(ctx.output_dim, dtype=np.float32)[labels_np]
                if mode == "rank_mse_control"
                else labels_np
            )
            x_test_np = y_test_np = None
            if ctx.test_size > 0:
                test_mode = _scalar_str(ctx.dd_test.get("experimentMode"), mode)
                if test_mode not in (
                    "rank_collapse_skewed",
                    "rank_uniform",
                    "rank_mse_control",
                ):
                    raise ValueError("rank train/test splits must both use a rank experimentMode")
                if (test_mode == "rank_mse_control") != (mode == "rank_mse_control"):
                    raise ValueError("rank train/test splits must use the same CE versus MSE target kind")
                x_test_np, test_labels_np = rank_classification_arrays(
                    ctx.rng,
                    sample_count=ctx.test_size,
                    input_dim=ctx.input_dim,
                    class_count=ctx.output_dim,
                    frequency_ratio=_scalar_float(ctx.dd_test.get("frequencyRatio"), 2.0),
                    uniform=test_mode == "rank_uniform",
                )
                y_test_np = (
                    np.eye(ctx.output_dim, dtype=np.float32)[test_labels_np]
                    if test_mode == "rank_mse_control"
                    else test_labels_np
                )
        else:
            raise ValueError(
                f"unknown experimentMode {mode!r}; expected one of {', '.join(EXPERIMENT_MODES)}"
            )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return DatasetArrays(
        x_np=x_np,
        y_np=y_np,
        x_test_np=x_test_np,
        y_test_np=y_test_np,
        input_dim=ctx.input_dim,
        output_dim=ctx.output_dim,
    )
