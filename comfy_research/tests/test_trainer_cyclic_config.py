"""parse_hyperparams_stage cyclic-schedule config: bounds, socket kinds, inert defaults.

MAX_TRAINING_STEPS is 200_000 for paper-reproduction epoch counts;
lr_schedule socket also accepts cyclic_lr_schedule; new trainer batch_schedule
socket accepts cyclic_batch_schedule only. Non-cyclic graphs must produce
inert cyclic context fields (the differential harness relies on this).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
from fastapi import HTTPException  # noqa: E402

from comfy_research.engine.runs.trainer_run import prepare_trainer_run  # noqa: E402
from comfy_research.engine.runs.trainer_run import iter_trainer_events  # noqa: E402
from comfy_research.engine.analysis.snapshot_schedules import idnns_epoch_snapshots  # noqa: E402
from comfy_research.engine.reproductions.random_labels import epoch_permutation  # noqa: E402
from comfy_research.engine.trainer.dataset_helpers import _take_epoch_shuffled_minibatch  # noqa: E402
from comfy_research.engine.trainer.prepare_config import MAX_TRAINING_STEPS  # noqa: E402
from comfy_research.schemas.graph import Edge, Node, NodeKind  # noqa: E402

_LINEAR_DS = {
    "inputDim": 3, "outputDim": 2, "trainSize": 8, "testSize": 4,
    "noiseLevel": 0, "seed": 0, "samplingMode": "fixed",
}
_MLP = {"inputDim": 3, "outputDim": 2, "depth": 2, "width": 6, "activation": "relu", "seed": 0}


def _graph(
    trainer_extra: dict | None = None,
    extra_nodes: list[Node] | None = None,
    extra_edges: list[Edge] | None = None,
) -> tuple[list[Node], list[Edge]]:
    trainer_data = {"computeDevice": "cpu", "batchSize": -1, "trainingSteps": 5, "logFrequency": 1}
    trainer_data.update(trainer_extra or {})
    nodes = [
        Node(id="dataset", type=NodeKind.linear_dataset, data=dict(_LINEAR_DS)),
        Node(id="model", type=NodeKind.mlp_model, data=dict(_MLP)),
        Node(id="optimizer", type=NodeKind.adam_optimizer, data={"learningRate": 0.01}),
        Node(id="loss", type=NodeKind.mse_loss, data={}),
        Node(id="trainer", type=NodeKind.trainer, data=trainer_data),
        *(extra_nodes or []),
    ]
    edges = [
        Edge(id="e-ds", source="dataset", target="trainer", sourceHandle="dataset", targetHandle="dataset"),
        Edge(id="e-m", source="model", target="trainer", sourceHandle="model", targetHandle="model"),
        Edge(id="e-o", source="optimizer", target="trainer", sourceHandle="optimizer", targetHandle="optimizer"),
        Edge(id="e-l", source="loss", target="trainer", sourceHandle="loss", targetHandle="loss"),
        *(extra_edges or []),
    ]
    return nodes, edges


def test_max_training_steps_is_200k() -> None:
    assert MAX_TRAINING_STEPS == 200_000


def test_training_steps_at_limit_accepted() -> None:
    nodes, edges = _graph({"trainingSteps": 200_000})
    ctx = prepare_trainer_run(nodes, edges, "trainer")
    assert ctx.training_steps == 200_000


def test_training_steps_over_limit_rejected() -> None:
    nodes, edges = _graph({"trainingSteps": 200_001})
    with pytest.raises(HTTPException) as ei:
        prepare_trainer_run(nodes, edges, "trainer")
    assert ei.value.status_code == 400
    assert ei.value.detail == "training steps must be 1..200000"


def test_non_cyclic_graph_has_inert_cyclic_fields() -> None:
    nodes, edges = _graph()
    ctx = prepare_trainer_run(nodes, edges, "trainer")
    assert ctx.cyclic_lr_min == 0.0
    assert ctx.cyclic_lr_max == 0.0
    assert ctx.cyclic_lr_cycle_steps == 0
    assert ctx.cyclic_batch_min == 0
    assert ctx.cyclic_batch_max == 0
    assert ctx.cyclic_batch_cycle_steps == 0
    assert ctx.cyclic_schedule_mode == "discrete_epoch"
    assert ctx.cyclic_cycle_epochs == 10
    assert ctx.training_data_epochs == 0
    assert ctx.log_schedule == "fixed_interval"
    assert ctx.log_samples == 1800
    assert ctx.log_aggregation == "last_batch"
    assert ctx.log_timing == "post_update"
    assert ctx.test_evaluation == "log_ticks"
    assert ctx.minibatch_sampling == "independent_step"


def test_affine_minibatches_and_idnns_logging_compile_and_run() -> None:
    nodes, edges = _graph(
        {
            "batchSize": 2,
            "trainingLengthMode": "epochs",
            "trainingEpochs": 10,
            "trainingSteps": 1,
            "logSchedule": "idnns_logspace",
            "logSamples": 10,
            "minibatchSampling": "affine_epoch",
        }
    )
    ctx = prepare_trainer_run(nodes, edges, "trainer")
    assert ctx.training_steps == 40
    assert ctx.log_schedule == "idnns_logspace"
    assert ctx.log_samples == 10
    assert ctx.minibatch_sampling == "affine_epoch"

    complete = next(
        event
        for event in iter_trainer_events(nodes, edges, "trainer")
        if event["type"] == "complete"
    )
    expected_epochs = idnns_epoch_snapshots(10, samples=10).tolist()
    expected_steps = [0, *[epoch * 4 for epoch in expected_epochs if epoch > 0]]
    assert complete["step_ticks"] == expected_steps
    assert complete["epoch_ticks"] == [float(step / 4) for step in expected_steps]


def test_epoch_shuffle_uses_formal_reproduction_permutation() -> None:
    x = torch.arange(24).reshape(8, 3)
    y = torch.arange(8)
    batches = []
    for step_in_epoch in range(4):
        xb, yb = _take_epoch_shuffled_minibatch(
            x,
            y,
            2,
            epoch=3,
            step_in_epoch=step_in_epoch,
            run_seed=1707,
        )
        assert xb.shape == (2, 3)
        batches.extend(yb.tolist())
    assert batches == epoch_permutation(8, seed=1707, epoch=3).tolist()


def test_interval_sample_mean_matches_per_step_losses() -> None:
    per_step_nodes, per_step_edges = _graph(
        {
            "batchSize": 2,
            "trainingSteps": 4,
            "logFrequency": 1,
            "minibatchSampling": "epoch_shuffle",
            "minibatchSeed": 1707,
        }
    )
    interval_nodes, interval_edges = _graph(
        {
            "batchSize": 2,
            "trainingSteps": 4,
            "logFrequency": 4,
            "logAggregation": "interval_sample_mean",
            "minibatchSampling": "epoch_shuffle",
            "minibatchSeed": 1707,
        }
    )
    per_step = next(
        event
        for event in iter_trainer_events(per_step_nodes, per_step_edges, "trainer")
        if event["type"] == "complete"
    )
    interval = next(
        event
        for event in iter_trainer_events(interval_nodes, interval_edges, "trainer")
        if event["type"] == "complete"
    )
    assert interval["step_ticks"][-1] == 4
    assert interval["loss_history"][-1] == pytest.approx(
        sum(per_step["loss_history"][-4:]) / 4
    )


def test_final_only_test_evaluation_skips_intermediate_full_set_passes() -> None:
    nodes, edges = _graph(
        {
            "trainingSteps": 3,
            "logFrequency": 1,
            "testEvaluation": "final_only",
        }
    )
    complete = next(
        event
        for event in iter_trainer_events(nodes, edges, "trainer")
        if event["type"] == "complete"
    )
    assert complete["step_ticks"] == [0, 1, 2, 3]
    assert len(complete["test_loss_history"]) == 1


def test_pre_update_logging_uses_paper_iteration_ticks() -> None:
    nodes, edges = _graph(
        {
            "trainingSteps": 5,
            "logFrequency": 2,
            "logTiming": "pre_update",
        }
    )
    complete = next(
        event
        for event in iter_trainer_events(nodes, edges, "trainer")
        if event["type"] == "complete"
    )
    assert complete["step_ticks"] == [0, 2, 4]


@pytest.mark.parametrize(
    ("trainer_extra", "message"),
    [
        ({"logSchedule": "unknown"}, "logSchedule"),
        ({"logAggregation": "unknown"}, "logAggregation"),
        ({"testEvaluation": "unknown"}, "testEvaluation"),
        ({"logTiming": "unknown"}, "logTiming"),
        (
            {
                "logTiming": "pre_update",
                "logAggregation": "interval_sample_mean",
            },
            "pre_update logging requires",
        ),
        ({"minibatchSampling": "unknown"}, "minibatchSampling"),
        ({"logSchedule": "idnns_logspace"}, "requires Trainer training length mode = epochs"),
        ({"minibatchSampling": "affine_epoch"}, "requires Trainer training length mode = epochs"),
    ],
)
def test_reproduction_trainer_modes_validate(
    trainer_extra: dict,
    message: str,
) -> None:
    nodes, edges = _graph(trainer_extra)
    with pytest.raises(HTTPException) as exc:
        prepare_trainer_run(nodes, edges, "trainer")
    assert message in str(exc.value.detail)


def test_lr_schedule_socket_rejects_other_kinds() -> None:
    bad = Node(id="bad", type=NodeKind.cyclic_batch_schedule, data={})
    e = Edge(id="e-bad", source="bad", target="optimizer", sourceHandle="schedule", targetHandle="lr_schedule")
    nodes, edges = _graph(extra_nodes=[bad], extra_edges=[e])
    with pytest.raises(HTTPException) as ei:
        prepare_trainer_run(nodes, edges, "trainer")
    assert ei.value.status_code == 400
    assert "expects lr_schedule or cyclic_lr_schedule" in str(ei.value.detail)


def test_batch_schedule_socket_rejects_other_kinds() -> None:
    bad = Node(id="bad", type=NodeKind.lr_schedule, data={})
    e = Edge(id="e-bad", source="bad", target="trainer", sourceHandle="schedule", targetHandle="batch_schedule")
    nodes, edges = _graph(extra_nodes=[bad], extra_edges=[e])
    with pytest.raises(HTTPException) as ei:
        prepare_trainer_run(nodes, edges, "trainer")
    assert ei.value.status_code == 400
    assert "batch_schedule socket expects cyclic_batch_schedule" in str(ei.value.detail)


def test_cyclic_batch_max_cannot_exceed_train_size() -> None:
    cbs = Node(
        id="cbs",
        type=NodeKind.cyclic_batch_schedule,
        data={"batchMin": 2, "batchMax": 640, "cycleLengthEpochs": 10, "refBatchSize": 2},
    )
    e = Edge(id="e-cbs", source="cbs", target="trainer", sourceHandle="schedule", targetHandle="batch_schedule")
    nodes, edges = _graph(extra_nodes=[cbs], extra_edges=[e])
    with pytest.raises(HTTPException) as ei:
        prepare_trainer_run(nodes, edges, "trainer")
    assert ei.value.status_code == 400
    assert "batchMax (640) cannot exceed train size (8)" in str(ei.value.detail)


def test_cbs_seeds_batch_size_from_batch_min() -> None:
    cbs = Node(
        id="cbs",
        type=NodeKind.cyclic_batch_schedule,
        data={"batchMin": 2, "batchMax": 4, "cycleLengthEpochs": 2, "refBatchSize": 2},
    )
    e = Edge(id="e-cbs", source="cbs", target="trainer", sourceHandle="schedule", targetHandle="batch_schedule")
    nodes, edges = _graph(extra_nodes=[cbs], extra_edges=[e])
    ctx = prepare_trainer_run(nodes, edges, "trainer")
    assert ctx.train_batch_size == 2
    assert ctx.cyclic_batch_min == 2
    assert ctx.cyclic_batch_max == 4
    assert ctx.cyclic_batch_cycle_steps > 0
    assert ctx.cyclic_lr_cycle_steps == 0


def test_epoch_derived_steps_over_limit_rejected() -> None:
    """epochs 模式推导出的 training_steps 同样过 200k 上限门。"""
    import pytest
    from fastapi import HTTPException

    from comfy_research.engine.runs.trainer_run import prepare_trainer_run
    from comfy_research.tests.test_cyclic_trainer_compile import _clr_graph

    nodes, edges = _clr_graph()
    for n in nodes:
        if n.id == "trainer":
            n.data["trainingEpochs"] = 600  # 600*391 = 234600 > 200k
    with pytest.raises(HTTPException) as e:
        prepare_trainer_run(nodes, edges, "trainer", validate_only=True)
    assert "training steps must be 1..200000" in str(e.value.detail)


def test_epoch_mode_overrides_small_stored_training_steps() -> None:
    """The frontend no longer overwrites trainingSteps in epochs mode,
    so the stored value (e.g. legacy default 100) reaches the backend. It must pass the
    raw bounds gate, then the epochs override recomputes training_steps server-side
    (prepare_config is the single source) and re-validates the derived value."""
    nodes, edges = _graph(
        {"trainingSteps": 100, "batchSize": 2, "trainingLengthMode": "epochs", "trainingEpochs": 3}
    )
    ctx = prepare_trainer_run(nodes, edges, "trainer")
    # trainSize=8, batchSize=2 → 4 steps/epoch → 3 epochs → 12 steps, not the stored 100.
    assert ctx.training_steps == 12
    assert ctx.training_data_epochs == 3


def test_cyclic_epoch_run_reports_optimizer_steps_and_exact_epochs_separately() -> None:
    clr = Node(
        id="clr",
        type=NodeKind.cyclic_lr_schedule,
        data={
            "lrMin": 0.001,
            "lrMax": 0.005,
            "cycleLengthEpochs": 2,
            "refBatchSize": 2,
            "scheduleMode": "square_epoch",
        },
    )
    edge = Edge(
        id="e-clr",
        source="clr",
        target="optimizer",
        sourceHandle="lr_schedule",
        targetHandle="lr_schedule",
    )
    nodes, edges = _graph(
        {
            "batchSize": 2,
            "trainingLengthMode": "epochs",
            "trainingEpochs": 2,
            "trainingSteps": 1,
            "logFrequency": 99,
        },
        extra_nodes=[clr],
        extra_edges=[edge],
    )
    complete = next(event for event in iter_trainer_events(nodes, edges, "trainer") if event["type"] == "complete")
    assert complete["step_ticks"] == [0, 4, 8]
    assert complete["epoch_ticks"] == [0.0, 1.0, 2.0]


def test_cyclic_lr_preserves_param_group_ratios() -> None:
    """cyclic 绝对 LR 按 base group 比例应用(μP 组不被抹平);
    单组/等 base 时退化为原语义。直接测 loop 内的应用逻辑等价式。"""
    import torch

    from comfy_research.engine.optimizers.cyclic_schedules import cyclic_lr_for_step

    params1 = [torch.nn.Parameter(torch.zeros(2))]
    params2 = [torch.nn.Parameter(torch.zeros(2))]
    opt = torch.optim.SGD([
        {"params": params1, "lr": 0.02},
        {"params": params2, "lr": 0.01},
    ])
    base = [0.02, 0.01]
    abs_lr = cyclic_lr_for_step(
        0, lr_min=0.001, lr_max=0.005, mode="discrete_epoch",
        cycle_length_epochs=10, cycle_length_steps=0, steps_per_epoch=391,
    )
    ref = base[0]
    for gi, group in enumerate(opt.param_groups):
        group["lr"] = abs_lr * (base[gi] / ref)
    lrs = [g["lr"] for g in opt.param_groups]
    assert lrs[0] == abs_lr
    assert abs(lrs[1] / lrs[0] - 0.5) < 1e-12
