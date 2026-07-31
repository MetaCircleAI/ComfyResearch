#!/usr/bin/env python3
"""Regenerate the kept paper-repro templates (Jastrzębski Fig1 + Keskar Fig2+3 + Thilak Slingshot Fig1)."""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from comfy_research.engine.optimizers.cyclic_schedules import cbs_epochs_to_training_steps

_OUT = _REPO / "data" / "graph_library" / "templates"

# Keskar et al. 1609.04836
KESKAR_CIFAR_TRAIN = 50000
KESKAR_SB_BATCH = 256
KESKAR_LB_BATCH = KESKAR_CIFAR_TRAIN // 10
KESKAR_TRAIN_EPOCHS = 100
KESKAR_FIG3_ALPHA_MIN = -1
KESKAR_FIG3_ALPHA_MAX = 2
KESKAR_FIG3_ALPHA_STEPS = 25

# Jastrzębski et al. 1711.04623 — Fig 1
JAST_CIFAR_TRAIN = 45000
JAST_FIG1_EPOCHS = 300
JAST_SGD_MOMENTUM = 0.0
JAST_SGD_WEIGHT_DECAY = 0.0
JAST_FIG1_CBS_LR = 0.005
JAST_FIG1_CBS_BATCH_MIN = 128
JAST_FIG1_CBS_BATCH_MAX = 640
JAST_FIG1_CLR_BATCH = 128
JAST_FIG1_CLR_LR_MIN = 0.001
JAST_FIG1_CLR_LR_MAX = 0.005
JAST_FIG1_CYCLE_EPOCHS = 10
JAST_FIG1_REF_BATCH = 128


def _edge(src: str, tgt: str, sh: str, th: str) -> dict[str, Any]:
    return {
        "id": f"e-{src}-{tgt}-{sh}-{th}",
        "source": src,
        "target": tgt,
        "sourceHandle": sh,
        "targetHandle": th,
    }


def _keskar_epochs_to_steps(train_size: int, batch: int, epochs: int = KESKAR_TRAIN_EPOCHS) -> int:
    return epochs * math.ceil(train_size / batch)


def _keskar_cifar_ds_node(nid: str) -> dict[str, Any]:
    return {
        "id": nid,
        "type": "cifar10_dataset",
        "position": {"x": 40, "y": 80},
        "data": {
            "trainSize": KESKAR_CIFAR_TRAIN,
            "testSize": 10000,
            "seed": 0,
            "flattenOutput": False,
            "samplingMode": "fixed",
            "instanceTitle": "CIFAR-10 (C1)",
        },
    }


def _keskar_cifar_model_node(nid: str) -> dict[str, Any]:
    return {
        "id": nid,
        "type": "keskar_c1_c2_cnn_model",
        "position": {"x": 40, "y": 300},
        "data": {"architecture": "c1", "seed": 0, "instanceTitle": "Keskar C1"},
    }


def _keskar_shared_opt_lr_ce(ids: dict[str, str]) -> list[dict[str, Any]]:
    return [
        {
            "id": ids["adam"],
            "type": "adam_optimizer",
            "position": {"x": 300, "y": 200},
            "data": {
                "learningRate": 0.001,
                "beta1": 0.9,
                "beta2": 0.999,
                "weightDecay": 0.0,
                "instanceTitle": "Adam",
            },
        },
        {
            "id": ids["lr"],
            "type": "lr_schedule",
            "position": {"x": 300, "y": 60},
            "data": {"lrSchedule": "constant", "lrWarmupSteps": 0, "instanceTitle": "LR constant"},
        },
        {
            "id": ids["ce"],
            "type": "cross_entropy_loss",
            "position": {"x": 300, "y": 360},
            "data": {"lossScale": 1, "instanceTitle": "CE"},
        },
    ]


def _keskar_fig23_ids(prefix: str) -> dict[str, str]:
    keys = (
        "note",
        "ds",
        "model",
        "adam",
        "lr",
        "ce",
        "tr_sb",
        "tr_lb",
        "ckpt_sb",
        "ckpt_lb",
        "acc_sb",
        "acc_lb",
        "oviz_sb",
        "oviz_lb",
        "sampler",
        "cstable_fig3",
        "csviz_fig3",
        "cstable_fig2",
        "csviz_fig2",
    )
    return {k: f"{prefix}{k}" for k in keys}


def _keskar_fig23_edges(ids: dict[str, str]) -> list[dict[str, Any]]:
    return [
        _edge(ids["lr"], ids["adam"], "lr_schedule", "lr_schedule"),
        _edge(ids["ds"], ids["tr_sb"], "dataset", "dataset"),
        _edge(ids["model"], ids["tr_sb"], "model", "model"),
        _edge(ids["adam"], ids["tr_sb"], "optimizer", "optimizer"),
        _edge(ids["ce"], ids["tr_sb"], "loss", "loss"),
        _edge(ids["acc_sb"], ids["tr_sb"], "observables", "observables"),
        _edge(ids["tr_sb"], ids["oviz_sb"], "observable_results", "tensor"),
        _edge(ids["tr_sb"], ids["ckpt_sb"], "checkpoint", "model_checkpoint"),
        _edge(ids["oviz_sb"], ids["cstable_fig2"], "out_tensor", "stream"),
        _edge(ids["ds"], ids["tr_lb"], "dataset", "dataset"),
        _edge(ids["model"], ids["tr_lb"], "model", "model"),
        _edge(ids["adam"], ids["tr_lb"], "optimizer", "optimizer"),
        _edge(ids["ce"], ids["tr_lb"], "loss", "loss"),
        _edge(ids["acc_lb"], ids["tr_lb"], "observables", "observables"),
        _edge(ids["tr_lb"], ids["oviz_lb"], "observable_results", "tensor"),
        _edge(ids["tr_lb"], ids["ckpt_lb"], "checkpoint", "model_checkpoint"),
        _edge(ids["oviz_lb"], ids["cstable_fig2"], "out_tensor", "stream"),
        _edge(ids["cstable_fig2"], ids["csviz_fig2"], "series", "curves"),
        _edge(ids["ds"], ids["sampler"], "dataset", "dataset"),
        _edge(ids["model"], ids["sampler"], "model", "model"),
        _edge(ids["ce"], ids["sampler"], "loss", "loss"),
        _edge(ids["ckpt_sb"], ids["sampler"], "model", "checkpoint_sb"),
        _edge(ids["ckpt_lb"], ids["sampler"], "model", "checkpoint_lb"),
        _edge(ids["sampler"], ids["cstable_fig3"], "stream", "stream"),
        _edge(ids["cstable_fig3"], ids["csviz_fig3"], "series", "curves"),
    ]


def _keskar_fig23_nodes(
    ids: dict[str, str],
    *,
    note_text: str,
    sb_batch: int,
    lb_batch: int,
    sb_steps: int,
    lb_steps: int,
) -> list[dict[str, Any]]:
    return [
        {
            "id": ids["note"],
            "type": "comment",
            "position": {"x": 40, "y": -280},
            "data": {"text": note_text, "instanceTitle": "Keskar Fig 2+3"},
        },
        _keskar_cifar_ds_node(ids["ds"]),
        _keskar_cifar_model_node(ids["model"]),
        *_keskar_shared_opt_lr_ce(ids),
        {
            "id": ids["acc_sb"],
            "type": "observable_accuracy",
            "position": {"x": 300, "y": 520},
            "data": {"instanceTitle": "SB accuracy"},
        },
        {
            "id": ids["acc_lb"],
            "type": "observable_accuracy",
            "position": {"x": 300, "y": 680},
            "data": {"instanceTitle": "LB accuracy"},
        },
        {
            "id": ids["tr_sb"],
            "type": "trainer",
            "position": {"x": 600, "y": 80},
            "data": {
                "trainingLengthMode": "epochs",
                "trainingEpochs": KESKAR_TRAIN_EPOCHS,
                "trainingSteps": sb_steps,
                "logFrequency": math.ceil(KESKAR_CIFAR_TRAIN / sb_batch),
                "batchSize": sb_batch,
                "disableExtraObservables": True,
                "computeDevice": "auto",
                "remoteGpu": False,
                "gradClipMaxNorm": 0,
                "instanceTitle": f"Trainer SB (B={sb_batch})",
            },
        },
        {
            "id": ids["tr_lb"],
            "type": "trainer",
            "position": {"x": 600, "y": 620},
            "data": {
                "trainingLengthMode": "epochs",
                "trainingEpochs": KESKAR_TRAIN_EPOCHS,
                "trainingSteps": lb_steps,
                "logFrequency": math.ceil(KESKAR_CIFAR_TRAIN / lb_batch),
                "batchSize": lb_batch,
                "disableExtraObservables": True,
                "computeDevice": "auto",
                "remoteGpu": False,
                "gradClipMaxNorm": 0,
                "instanceTitle": f"Trainer LB (B={lb_batch})",
            },
        },
        {
            "id": ids["oviz_sb"],
            "type": "observable_viz",
            "position": {"x": 900, "y": 80},
            "data": {
                "pairedObservableId": ids["acc_sb"],
                "pairedTrainerId": ids["tr_sb"],
                "vizVariant": "accuracy",
                "instanceTitle": "SB train/test accuracy",
            },
        },
        {
            "id": ids["oviz_lb"],
            "type": "observable_viz",
            "position": {"x": 900, "y": 620},
            "data": {
                "pairedObservableId": ids["acc_lb"],
                "pairedTrainerId": ids["tr_lb"],
                "vizVariant": "accuracy",
                "instanceTitle": "LB train/test accuracy",
            },
        },
        {
            "id": ids["ckpt_sb"],
            "type": "model_checkpoint",
            "position": {"x": 900, "y": 400},
            "data": {"checkpointSource": "memory", "instanceTitle": f"Checkpoint SB (B={sb_batch})"},
        },
        {
            "id": ids["ckpt_lb"],
            "type": "model_checkpoint",
            "position": {"x": 900, "y": 940},
            "data": {"checkpointSource": "memory", "instanceTitle": f"Checkpoint LB (B={lb_batch})"},
        },
        {
            "id": ids["cstable_fig2"],
            "type": "curve_series_table",
            "position": {"x": 1180, "y": 180},
            "data": {
                "rows": [],
                "captureMetrics": ["train_acc", "test_acc"],
                "instanceTitle": "Fig 2 accuracy curves",
            },
        },
        {
            "id": ids["csviz_fig2"],
            "type": "curve_series_viz",
            "position": {"x": 1460, "y": 180},
            "data": {
                "instanceTitle": "Fig 2 accuracy plot",
                "dualAxis": False,
                "plotXMode": "epoch",
                "plotXKey": "epoch",
            },
        },
        {
            "id": ids["sampler"],
            "type": "parametric_path_sampler",
            "position": {"x": 1180, "y": 680},
            "data": {
                "alphaMin": KESKAR_FIG3_ALPHA_MIN,
                "alphaMax": KESKAR_FIG3_ALPHA_MAX,
                "alphaSteps": KESKAR_FIG3_ALPHA_STEPS,
                "computeDevice": "auto",
                "remoteGpu": False,
                "instanceTitle": "Fig 3 parametric path",
            },
        },
        {
            "id": ids["cstable_fig3"],
            "type": "curve_series_table",
            "position": {"x": 1460, "y": 680},
            "data": {"rows": [], "captureMetrics": [], "instanceTitle": "Fig 3 path curves"},
        },
        {
            "id": ids["csviz_fig3"],
            "type": "curve_series_viz",
            "position": {"x": 1740, "y": 680},
            "data": {
                "instanceTitle": "Fig 3 path plot",
                "dualAxis": True,
                "plotXMode": "param",
                "plotXKey": "param",
            },
        },
    ]


def _saved(name: str, slug: str, nodes: list, edges: list) -> dict[str, Any]:
    return {
        "id": slug,
        "name": name,
        "tier": "medium",
        "document": {
            "version": 1,
            "nodes": nodes,
            "edges": edges,
            "viewport": {"x": 0, "y": 0, "zoom": 0.45},
        },
        "savedAt": time.time() * 1000.0,
    }


def keskar_fig23_sb_lb() -> dict[str, Any]:
    """Fig 2 SB/LB accuracy curves + Fig 3 parametric path on CIFAR-10 C1."""
    ids = _keskar_fig23_ids("k23-")
    sb_steps = _keskar_epochs_to_steps(KESKAR_CIFAR_TRAIN, KESKAR_SB_BATCH)
    lb_steps = _keskar_epochs_to_steps(KESKAR_CIFAR_TRAIN, KESKAR_LB_BATCH)
    note = (
        "Keskar 1609.04836 Fig 2 + Fig 3; C1 / CIFAR-10 only. "
        "Both independent trainers start from model seed 0.\n"
        f"SB: B={KESKAR_SB_BATCH}, {KESKAR_TRAIN_EPOCHS} epochs ({sb_steps} steps). "
        f"LB: B={KESKAR_LB_BATCH}=10% train, {KESKAR_TRAIN_EPOCHS} epochs ({lb_steps} steps).\n"
        "Click Train on both trainers. Each writes its own checkpoint and appends train/test accuracy "
        "to the Fig 2 epoch plot.\n"
        f"Then click Run on Fig 3: {KESKAR_FIG3_ALPHA_STEPS} points over α∈[-1,2]. "
        "Trainable parameters are interpolated; evaluation uses per-batch BatchNorm statistics, "
        "matching the paper protocol."
    )
    nodes = _keskar_fig23_nodes(
        ids,
        note_text=note,
        sb_batch=KESKAR_SB_BATCH,
        lb_batch=KESKAR_LB_BATCH,
        sb_steps=sb_steps,
        lb_steps=lb_steps,
    )
    edges = _keskar_fig23_edges(ids)
    return _saved(
        "repro: Keskar Fig 2+3 SB/LB",
        "repro-keskar-fig23-sb-lb",
        nodes,
        edges,
    )


def keskar_fig3_parametric_path() -> dict[str, Any]:
    """Legacy alias used by validate_only tests."""
    return keskar_fig23_sb_lb()


def _jastr_epochs_to_steps(train_size: int, batch: int, epochs: int = JAST_FIG1_EPOCHS) -> int:
    return epochs * math.ceil(train_size / batch)


def _jastr_shared_ds_model_ce(ids: dict[str, str]) -> list[dict[str, Any]]:
    return [
        {
            "id": ids["ds"],
            "type": "cifar10_dataset",
            "position": {"x": 40, "y": 40},
            "data": {
                "trainSize": JAST_CIFAR_TRAIN,
                "testSize": 10000,
                "initSeed": 777,
                "seed": 777,
                "trainingRecipe": "jastrzbski_fig1",
                "instanceTitle": "CIFAR-10 (45k train, split seed 777)",
            },
        },
        {
            "id": ids["model"],
            "type": "vgg11_cifar_model",
            "position": {"x": 40, "y": 200},
            "data": {"seed": [0, 1, 2, 3, 4], "instanceTitle": "VGG-11 + BN (5 seeds)"},
        },
        {
            "id": ids["ce"],
            "type": "cross_entropy_loss",
            "position": {"x": 280, "y": 120},
            "data": {"lossScale": 1, "instanceTitle": "CE"},
        },
    ]


def _jastr_fig1_cyclic_chain_edges(
    ids: dict[str, str],
    *,
    ds: str,
    model: str,
    sgd: str,
    tr: str,
    tviz: str,
    acc: str,
    oviz: str,
    cbs_sched: str | None = None,
    clr_sched: str | None = None,
    cstable: str | None = None,
    csviz: str | None = None,
) -> list[dict[str, Any]]:
    edges = [
        _edge(ds, tr, "dataset", "dataset"),
        _edge(model, tr, "model", "model"),
        _edge(sgd, tr, "optimizer", "optimizer"),
        _edge(ids["ce"], tr, "loss", "loss"),
        _edge(acc, tr, "observables", "observables"),
        _edge(tr, tviz, "loss_results", "tensor_list"),
        _edge(tr, oviz, "observable_results", "tensor"),
    ]
    if cbs_sched:
        edges.append(_edge(cbs_sched, tr, "batch_schedule", "batch_schedule"))
    if clr_sched:
        edges.append(_edge(clr_sched, sgd, "lr_schedule", "lr_schedule"))
    if cstable:
        edges.append(_edge(oviz, cstable, "out_tensor", "stream"))
    if cstable and csviz:
        edges.append(_edge(cstable, csviz, "series", "curves"))
    return edges


def jastrzbski_fig1_vgg11() -> dict[str, Any]:
    """Fig 1 left: cyclic batch size vs cyclic LR on VGG-11 + BN."""
    p = "j1-"
    ids = {
        k: f"{p}{k}"
        for k in (
            "note",
            "ds",
            "model",
            "ce",
            "sgd_cbs",
            "cbs_sched",
            "tr_cbs",
            "tviz_cbs",
            "acc_cbs",
            "oviz_cbs",
            "sgd_clr",
            "clr_sched",
            "tr_clr",
            "tviz_clr",
            "acc_clr",
            "oviz_clr",
            "cstable",
            "csviz",
        )
    }
    clr_steps = _jastr_epochs_to_steps(JAST_CIFAR_TRAIN, JAST_FIG1_REF_BATCH, JAST_FIG1_EPOCHS)
    cbs_steps = cbs_epochs_to_training_steps(
        JAST_FIG1_EPOCHS,
        train_size=JAST_CIFAR_TRAIN,
        batch_min=JAST_FIG1_CBS_BATCH_MIN,
        batch_max=JAST_FIG1_CBS_BATCH_MAX,
        cycle_length_epochs=JAST_FIG1_CYCLE_EPOCHS,
        mode="square_epoch",
    )
    log_freq = math.ceil(JAST_CIFAR_TRAIN / JAST_FIG1_REF_BATCH)
    note = (
        "Jastrzębski 1711.04623 Fig 1 LEFT (VGG-11 + BN, CIFAR-10). "
        "CBS (blue) vs CLR (red) should overlap.\n"
        "Data: RandomState(777) 45k split; pixel-mean/global-std normalization; crop+flip; epoch shuffle.\n"
        f"CBS: η={JAST_FIG1_CBS_LR} fixed, B∈[{JAST_FIG1_CBS_BATCH_MIN},{JAST_FIG1_CBS_BATCH_MAX}] cyclic.\n"
        f"CLR: B={JAST_FIG1_CLR_BATCH} fixed, η∈[{JAST_FIG1_CLR_LR_MIN},{JAST_FIG1_CLR_LR_MAX}] cyclic.\n"
        "Square wave: 5 epochs high, then 5 epochs low. "
        f"SGD momentum {JAST_SGD_MOMENTUM}, wd {JAST_SGD_WEIGHT_DECAY}.\n"
        "Run Train series on both stacks (seeds 0..4); the plot averages by trainer run.\n"
        f"Fig 1 acc plot ({JAST_FIG1_EPOCHS} data epochs; "
        f"CBS steps={cbs_steps}, CLR steps={clr_steps})."
    )
    cycle_data = {
        "cycleLengthEpochs": JAST_FIG1_CYCLE_EPOCHS,
        "refBatchSize": JAST_FIG1_REF_BATCH,
        "cycleLengthSteps": 0,
        "scheduleMode": "square_epoch",
    }
    nodes = [
        {
            "id": ids["note"],
            "type": "comment",
            "position": {"x": 40, "y": -280},
            "data": {"text": note, "instanceTitle": "Jastrzębski Fig 1 left"},
        },
        *_jastr_shared_ds_model_ce(ids),
        {
            "id": ids["cbs_sched"],
            "type": "cyclic_batch_schedule",
            "position": {"x": 280, "y": 280},
            "data": {
                "batchMin": JAST_FIG1_CBS_BATCH_MIN,
                "batchMax": JAST_FIG1_CBS_BATCH_MAX,
                **cycle_data,
                "instanceTitle": "Cyclic BS (128↔640)",
            },
        },
        {
            "id": ids["sgd_cbs"],
            "type": "sgd_optimizer",
            "position": {"x": 280, "y": 420},
            "data": {
                "learningRate": JAST_FIG1_CBS_LR,
                "momentum": JAST_SGD_MOMENTUM,
                "weightDecay": JAST_SGD_WEIGHT_DECAY,
                "instanceTitle": f"SGD CBS (η={JAST_FIG1_CBS_LR})",
            },
        },
        {
            "id": ids["tr_cbs"],
            "type": "trainer",
            "position": {"x": 560, "y": 280},
            "data": {
                "trainingLengthMode": "epochs",
                "trainingEpochs": JAST_FIG1_EPOCHS,
                "trainingSteps": cbs_steps,
                "logFrequency": log_freq,
                "batchSize": JAST_FIG1_CBS_BATCH_MIN,
                "disableExtraObservables": True,
                "computeDevice": "cuda",
                "remoteGpu": True,
                "instanceTitle": "Trainer CBS (cyclic batch)",
            },
        },
        {
            "id": ids["tviz_cbs"],
            "type": "training_visualization",
            "position": {"x": 860, "y": 280},
            "data": {"instanceTitle": "CBS loss viz"},
        },
        {
            "id": ids["acc_cbs"],
            "type": "observable_accuracy",
            "position": {"x": 560, "y": 440},
            "data": {"instanceTitle": "CBS accuracy"},
        },
        {
            "id": ids["oviz_cbs"],
            "type": "observable_viz",
            "position": {"x": 860, "y": 440},
            "data": {
                "pairedObservableId": ids["acc_cbs"],
                "pairedTrainerId": ids["tr_cbs"],
                "vizVariant": "accuracy",
                "instanceTitle": "CBS acc viz",
            },
        },
        {
            "id": ids["clr_sched"],
            "type": "cyclic_lr_schedule",
            "position": {"x": 280, "y": 620},
            "data": {
                "lrMin": JAST_FIG1_CLR_LR_MIN,
                "lrMax": JAST_FIG1_CLR_LR_MAX,
                **cycle_data,
                "instanceTitle": "Cyclic LR (0.001↔0.005)",
            },
        },
        {
            "id": ids["sgd_clr"],
            "type": "sgd_optimizer",
            "position": {"x": 280, "y": 760},
            "data": {
                "learningRate": JAST_FIG1_CLR_LR_MAX,
                "momentum": JAST_SGD_MOMENTUM,
                "weightDecay": JAST_SGD_WEIGHT_DECAY,
                "instanceTitle": f"SGD CLR (B={JAST_FIG1_CLR_BATCH})",
            },
        },
        {
            "id": ids["tr_clr"],
            "type": "trainer",
            "position": {"x": 560, "y": 620},
            "data": {
                "trainingLengthMode": "epochs",
                "trainingEpochs": JAST_FIG1_EPOCHS,
                "trainingSteps": clr_steps,
                "logFrequency": log_freq,
                "batchSize": JAST_FIG1_CLR_BATCH,
                "disableExtraObservables": True,
                "computeDevice": "cuda",
                "remoteGpu": True,
                "instanceTitle": "Trainer CLR (cyclic LR)",
            },
        },
        {
            "id": ids["tviz_clr"],
            "type": "training_visualization",
            "position": {"x": 860, "y": 620},
            "data": {"instanceTitle": "CLR loss viz"},
        },
        {
            "id": ids["acc_clr"],
            "type": "observable_accuracy",
            "position": {"x": 560, "y": 780},
            "data": {"instanceTitle": "CLR accuracy"},
        },
        {
            "id": ids["oviz_clr"],
            "type": "observable_viz",
            "position": {"x": 860, "y": 780},
            "data": {
                "pairedObservableId": ids["acc_clr"],
                "pairedTrainerId": ids["tr_clr"],
                "vizVariant": "accuracy",
                "instanceTitle": "CLR acc viz",
            },
        },
        {
            "id": ids["cstable"],
            "type": "curve_series_table",
            "position": {"x": 1120, "y": 440},
            "data": {
                "rows": [],
                "captureMetrics": ["train_acc", "test_acc"],
                "instanceTitle": "Fig 1 acc curves",
            },
        },
        {
            "id": ids["csviz"],
            "type": "curve_series_viz",
            "position": {"x": 1380, "y": 440},
            "data": {
                "instanceTitle": "Fig 1 acc plot (5-seed mean)",
                "dualAxis": False,
                "meanByRun": True,
                "plotXMode": "epoch",
                "plotXKey": "epoch",
            },
        },
    ]
    edges = [
        *_jastr_fig1_cyclic_chain_edges(
            ids,
            ds=ids["ds"],
            model=ids["model"],
            sgd=ids["sgd_cbs"],
            tr=ids["tr_cbs"],
            tviz=ids["tviz_cbs"],
            acc=ids["acc_cbs"],
            oviz=ids["oviz_cbs"],
            cbs_sched=ids["cbs_sched"],
            cstable=ids["cstable"],
            csviz=ids["csviz"],
        ),
        *_jastr_fig1_cyclic_chain_edges(
            ids,
            ds=ids["ds"],
            model=ids["model"],
            sgd=ids["sgd_clr"],
            tr=ids["tr_clr"],
            tviz=ids["tviz_clr"],
            acc=ids["acc_clr"],
            oviz=ids["oviz_clr"],
            clr_sched=ids["clr_sched"],
            cstable=ids["cstable"],
        ),
    ]
    return _saved(
        "repro: Jastrzębski Fig 1 cyclic CBS vs CLR",
        "repro-jastrzbski-fig1-vgg11",
        nodes,
        edges,
    )


# Thilak et al. TMLR 2024 (Slingshot Effect) — Fig 1 == Appendix A.3 MLP
SS_TRAIN_SAMPLES = 200
SS_TEST_SAMPLES = 200
SS_MLP_DEPTH = 6
SS_MLP_WIDTH = 200
SS_ADAM_LR = 0.001
SS_ADAM_BETA1 = 0.9
SS_ADAM_BETA2 = 0.95
SS_ADAM_EPS = 1e-8
# full-batch → 1 step = 1 epoch。前 ~7000 步是弹弓早期(norm 增长弥散在循环全程),
# 之后进入论文 Fig 1 的成熟形态:平顶阶梯(增长集中在尖峰前,循环间平台)。
SS_TRAIN_STEPS = 12000


def thilak_fig1_slingshot() -> dict[str, Any]:
    """Fig 1 (== Appendix A.3 Figure 10): 6-layer MLP, 200 CIFAR-10 samples,
    full-batch Adam without weight decay → cyclic Slingshots in the TPT."""
    p = "ss-"
    ids = {
        k: f"{p}{k}"
        for k in ("note", "ds", "model", "adam", "ce", "obs_norm", "obs_acc",
                  "tr", "tviz", "oviz_norm", "oviz_acc", "cstable", "csviz")
    }
    note = (
        "Thilak et al. TMLR 2024 (Slingshot Effect) Fig 1 == Appendix A.3 MLP.\n"
        f"6-layer ReLU MLP (width {SS_MLP_WIDTH}) on {SS_TRAIN_SAMPLES} CIFAR-10 samples, "
        f"FULL-batch Adam: lr={SS_ADAM_LR}, β1={SS_ADAM_BETA1}, β2={SS_ADAM_BETA2}, "
        f"ε={SS_ADAM_EPS}, weight decay=0 (essential). 1 step = 1 epoch.\n"
        "Expected: train acc hits 100% early (~step 400); afterwards Slingshot cycles "
        "every ~600 steps — train loss spikes ≥8 orders of magnitude off a ~1e-8 floor, "
        "each riding a LAST-LAYER WEIGHT NORM growth phase (dashed boxes in Fig 1).\n"
        "The paper's flat-plateau staircase (growth concentrated at spikes, flat "
        "between cycles) is the LATE-stage form — it emerges after ~7000 steps; the "
        "earlier cycles show continuous norm growth with rate modulation.\n"
        f"Fig-1 plot: curve series viz, dual axis — log train loss (left) + linear "
        f"last-layer norm (right), x = step. {SS_TRAIN_STEPS} steps ≈ 1–2 min on CPU/MPS.\n"
        "Quick look: set trainer steps to 1000 to confirm acc→100% + first slingshot (~830)."
    )
    nodes = [
        {
            "id": ids["note"],
            "type": "comment",
            "position": {"x": 40, "y": -300},
            "data": {"text": note, "instanceTitle": "Slingshot Fig 1"},
        },
        {
            "id": ids["ds"],
            "type": "cifar10_dataset",
            "position": {"x": 40, "y": 120},
            "data": {
                "trainSize": SS_TRAIN_SAMPLES,
                "testSize": SS_TEST_SAMPLES,
                "initSeed": 0,
                "seed": 0,
                "flattenOutput": True,
                "samplingMode": "fixed",
                "instanceTitle": f"CIFAR-10 ({SS_TRAIN_SAMPLES} samples)",
            },
        },
        {
            "id": ids["model"],
            "type": "mlp_model",
            "position": {"x": 40, "y": 320},
            "data": {
                "inputDim": 3072,
                "outputDim": 10,
                "depth": SS_MLP_DEPTH,
                "width": SS_MLP_WIDTH,
                "activation": "relu",
                "seed": 0,
                "instanceTitle": f"MLP d{SS_MLP_DEPTH}×w{SS_MLP_WIDTH}",
            },
        },
        {
            "id": ids["adam"],
            "type": "adam_optimizer",
            "position": {"x": 40, "y": 520},
            "data": {
                "learningRate": SS_ADAM_LR,
                "beta1": SS_ADAM_BETA1,
                "beta2": SS_ADAM_BETA2,
                "epsilon": SS_ADAM_EPS,
                "weightDecay": 0,
                "instanceTitle": f"Adam (β2={SS_ADAM_BETA2}, wd=0)",
            },
        },
        {
            "id": ids["ce"],
            "type": "cross_entropy_loss",
            "position": {"x": 40, "y": 680},
            "data": {"instanceTitle": "CE loss"},
        },
        {
            "id": ids["obs_norm"],
            "type": "observable_last_layer_weight_norm",
            "position": {"x": 280, "y": 560},
            "data": {"instanceTitle": "Last layer ‖W‖₂"},
        },
        {
            "id": ids["obs_acc"],
            "type": "observable_accuracy",
            "position": {"x": 280, "y": 700},
            "data": {"instanceTitle": "Train accuracy"},
        },
        {
            "id": ids["tr"],
            "type": "trainer",
            "position": {"x": 560, "y": 280},
            "data": {
                "trainingSteps": SS_TRAIN_STEPS,
                "logFrequency": 2,
                "batchSize": SS_TRAIN_SAMPLES,
                # 不禁 extra observables:last-layer norm 是本模板主角(纯读权重零开销)。
                "disableExtraObservables": False,
                "computeDevice": "auto",
                "remoteGpu": False,
                "instanceTitle": "Trainer (full-batch Adam)",
            },
        },
        {
            "id": ids["tviz"],
            "type": "training_visualization",
            "position": {"x": 860, "y": 120},
            "data": {"instanceTitle": "Train loss viz"},
        },
        {
            "id": ids["oviz_norm"],
            "type": "observable_viz",
            "position": {"x": 860, "y": 420},
            "data": {
                "pairedObservableId": ids["obs_norm"],
                "pairedTrainerId": ids["tr"],
                "vizVariant": "user",
                "instanceTitle": "Last layer norm viz",
            },
        },
        {
            "id": ids["oviz_acc"],
            "type": "observable_viz",
            "position": {"x": 860, "y": 620},
            "data": {
                "pairedObservableId": ids["obs_acc"],
                "pairedTrainerId": ids["tr"],
                "vizVariant": "accuracy",
                "instanceTitle": "Train acc viz",
            },
        },
        {
            "id": ids["cstable"],
            "type": "curve_series_table",
            "position": {"x": 1120, "y": 280},
            "data": {
                "rows": [],
                "captureMetrics": ["train_loss", "observable"],
                "instanceTitle": "Fig 1 loss+norm curves",
            },
        },
        {
            "id": ids["csviz"],
            "type": "curve_series_viz",
            "position": {"x": 1380, "y": 280},
            "data": {
                "instanceTitle": "Fig 1 slingshot plot",
                "dualAxis": True,
                "logScaleY": True,
                "plotXMode": "step",
            },
        },
    ]
    edges = [
        _edge(ids["ds"], ids["tr"], "dataset", "dataset"),
        _edge(ids["model"], ids["tr"], "model", "model"),
        _edge(ids["adam"], ids["tr"], "optimizer", "optimizer"),
        _edge(ids["ce"], ids["tr"], "loss", "loss"),
        _edge(ids["obs_norm"], ids["tr"], "observables", "observables"),
        _edge(ids["obs_acc"], ids["tr"], "observables", "observables"),
        _edge(ids["tr"], ids["tviz"], "loss_results", "tensor_list"),
        _edge(ids["tr"], ids["oviz_norm"], "observable_results", "tensor"),
        _edge(ids["tr"], ids["oviz_acc"], "observable_results", "tensor"),
        _edge(ids["tviz"], ids["cstable"], "out_tensor_list", "stream"),
        _edge(ids["oviz_norm"], ids["cstable"], "out_tensor", "stream"),
        _edge(ids["cstable"], ids["csviz"], "series", "curves"),
    ]
    return _saved(
        "repro: Thilak Fig 1 Slingshot Effect (MLP, 200 CIFAR-10)",
        "repro-thilak-fig1-slingshot",
        nodes,
        edges,
    )


def _entry_text(entry: dict) -> str:
    return json.dumps(entry, indent=2) + "\n"


def _mask_saved_at(text: str) -> str:
    import re

    return re.sub(r'"savedAt": [0-9.]+', '"savedAt": 0', text)


def main() -> int:
    """--check:内存生成与 committed 比较(savedAt 掩码),
    漂移则非零退出且不写工作树;无参 = 写文件。"""
    import sys

    check = "--check" in sys.argv[1:]
    _OUT.mkdir(parents=True, exist_ok=True)
    drift = []
    for entry in (keskar_fig23_sb_lb(), jastrzbski_fig1_vgg11(), thilak_fig1_slingshot()):
        path = _OUT / f"{entry['id']}.json"
        text = _entry_text(entry)
        if check:
            committed = path.read_text(encoding="utf-8") if path.is_file() else ""
            if _mask_saved_at(committed) != _mask_saved_at(text):
                drift.append(str(path))
            continue
        path.write_text(text, encoding="utf-8")
        print(path)
    if check:
        if drift:
            print("TEMPLATE DRIFT: " + ", ".join(drift))
            return 1
        print("repro templates OK: committed files match the seed tool.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
