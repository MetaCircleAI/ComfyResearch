"""CI checks for the kept paper-repro templates: schema, compile, expected params."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from comfy_research.engine.runs.trainer_run import prepare_trainer_run
from comfy_research.generated.node_params import validate_node_params
from comfy_research.schemas.graph import Edge, GraphDocument, Node, NodeKind
from comfy_research.schemas.saved_graph_library import SavedGraphEntry

_REPO = Path(__file__).resolve().parents[2]
_TEMPLATES = _REPO / "data" / "graph_library" / "templates"

# Only these ship as on-disk repro templates.
REPRO_TEMPLATE_SLUGS = (
    "repro-keskar-fig23-sb-lb",
    "repro-jastrzbski-fig1-vgg11",
    "repro-thilak-fig1-slingshot",
)

REPRO_TEMPLATE_NAMES = {
    "repro-keskar-fig23-sb-lb": "repro: Keskar Fig 2+3 SB/LB",
    "repro-jastrzbski-fig1-vgg11": "repro: Jastrzębski Fig 1 cyclic CBS vs CLR",
    "repro-thilak-fig1-slingshot": "repro: Thilak Fig 1 Slingshot Effect (MLP, 200 CIFAR-10)",
}

# validate_only=True must not touch the vision dataset build (CIFAR/MNIST download).
_VISION_BUILD = "comfy_research.engine.trainer.dataset_materialize.build_vision_numpy_arrays"


def _load_template(slug: str) -> SavedGraphEntry:
    path = _TEMPLATES / f"{slug}.json"
    assert path.is_file(), f"missing template {path}"
    return SavedGraphEntry.model_validate(json.loads(path.read_text(encoding="utf-8")))


def _nodes_edges(entry: SavedGraphEntry) -> tuple[list[Node], list[Edge]]:
    doc = GraphDocument.model_validate(entry.document)
    return list(doc.nodes), list(doc.edges)


@pytest.mark.parametrize("slug", REPRO_TEMPLATE_SLUGS)
def test_repro_template_saved_graph_entry(slug: str) -> None:
    entry = _load_template(slug)
    assert entry.id == slug
    assert entry.name == REPRO_TEMPLATE_NAMES[slug]
    assert entry.tier == "medium"
    assert entry.savedAt > 0
    GraphDocument.model_validate(entry.document)


@pytest.mark.parametrize("slug", REPRO_TEMPLATE_SLUGS)
def test_repro_template_node_kinds_and_params(slug: str) -> None:
    entry = _load_template(slug)
    doc = GraphDocument.model_validate(entry.document)
    for node in doc.nodes:
        kind = node.type.value if isinstance(node.type, NodeKind) else str(node.type)
        NodeKind(kind)
        data = node.data if isinstance(node.data, dict) else {}
        validate_node_params(kind, data)


def test_jastrzbski_fig1_template_compiles_with_paper_lengths() -> None:
    entry = _load_template("repro-jastrzbski-fig1-vgg11")
    nodes, edges = _nodes_edges(entry)
    types = {n.type for n in nodes}
    assert "vgg11_cifar_model" in types
    assert "cyclic_batch_schedule" in types
    assert "cyclic_lr_schedule" in types
    assert "cifar10_dataset" in types

    dataset = next(n for n in nodes if n.type == "cifar10_dataset")
    model = next(n for n in nodes if n.type == "vgg11_cifar_model")
    assert (dataset.data or {})["trainSize"] == 45000
    assert (dataset.data or {})["testSize"] == 10000
    assert (dataset.data or {})["initSeed"] == 777
    assert (dataset.data or {})["trainingRecipe"] == "jastrzbski_fig1"
    assert (model.data or {})["seed"] == [0, 1, 2, 3, 4]

    schedules = [n for n in nodes if n.type in {"cyclic_batch_schedule", "cyclic_lr_schedule"}]
    assert all((n.data or {})["scheduleMode"] == "square_epoch" for n in schedules)
    optimizers = [n for n in nodes if n.type == "sgd_optimizer"]
    assert len(optimizers) == 2
    assert all((n.data or {})["momentum"] == 0 for n in optimizers)
    assert all((n.data or {})["weightDecay"] == 0 for n in optimizers)

    curve_viz = next(n for n in nodes if n.type == "curve_series_viz")
    assert (curve_viz.data or {})["plotXMode"] == "epoch"
    assert (curve_viz.data or {})["dualAxis"] is False
    assert (curve_viz.data or {})["meanByRun"] is True
    trainers = [n for n in nodes if n.type == "trainer"]
    assert all((n.data or {})["computeDevice"] == "cuda" for n in trainers)
    assert all((n.data or {})["remoteGpu"] is True for n in trainers)
    for trainer in trainers:
        trainer.data["computeDevice"] = "cpu"
        trainer.data["remoteGpu"] = False

    tr_cbs = next(
        n.id
        for n in nodes
        if n.type == "trainer" and "CBS" in str((n.data or {}).get("instanceTitle", ""))
    )
    tr_clr = next(
        n.id
        for n in nodes
        if n.type == "trainer" and "CLR" in str((n.data or {}).get("instanceTitle", ""))
    )
    with patch(_VISION_BUILD) as mocked:
        ctx_cbs = prepare_trainer_run(nodes, edges, tr_cbs, validate_only=True)
        ctx_clr = prepare_trainer_run(nodes, edges, tr_clr, validate_only=True)
        mocked.assert_not_called()

    # Paper Fig 1: 300 data-epochs; CBS steps from varying batch; CLR fixed B=128.
    assert ctx_cbs.cyclic_batch_cycle_steps > 0
    assert ctx_cbs.cyclic_lr_cycle_steps == 0
    assert ctx_cbs.training_data_epochs == 300
    assert ctx_cbs.training_steps == 63450
    assert ctx_cbs.cyclic_batch_min == 128
    assert ctx_cbs.cyclic_batch_max == 640

    assert ctx_clr.cyclic_lr_cycle_steps > 0
    assert ctx_clr.cyclic_batch_cycle_steps == 0
    assert ctx_clr.training_steps == 105600
    assert ctx_clr.cyclic_lr_min == pytest.approx(0.001)
    assert ctx_clr.cyclic_lr_max == pytest.approx(0.005)


def test_thilak_fig1_template_compiles_with_paper_params() -> None:
    entry = _load_template("repro-thilak-fig1-slingshot")
    nodes, edges = _nodes_edges(entry)
    types = {n.type for n in nodes}
    assert "mlp_model" in types
    assert "cifar10_dataset" in types
    assert "observable_last_layer_weight_norm" in types
    assert "adam_optimizer" in types
    assert "curve_series_table" in types

    adam = next(n for n in nodes if n.type == "adam_optimizer")
    ad = adam.data or {}
    # 附录 A.3：lr 1e-3、β1 .9、β2 .95、ε 1e-8、wd=0（弹弓必要条件）。
    assert ad.get("learningRate") == pytest.approx(0.001)
    assert ad.get("beta1") == pytest.approx(0.9)
    assert ad.get("beta2") == pytest.approx(0.95)
    assert ad.get("epsilon") == pytest.approx(1e-8)
    assert ad.get("weightDecay") == 0

    table = next(n for n in nodes if n.type == "curve_series_table")
    assert (table.data or {}).get("captureMetrics") == ["train_loss", "observable"]
    csviz = next(n for n in nodes if n.type == "curve_series_viz")
    assert (csviz.data or {}).get("dualAxis") is True
    assert (csviz.data or {}).get("logScaleY") is True
    assert (csviz.data or {}).get("plotXMode") == "step"

    tr = next(n for n in nodes if n.type == "trainer")
    td = tr.data or {}
    ds = next(n for n in nodes if n.type == "cifar10_dataset")
    # full-batch:batchSize == trainSize(1 step = 1 epoch)。
    assert td.get("batchSize") == (ds.data or {}).get("trainSize") == 200
    # flatten 路径在 prepare 即物化(取扁平维度)——用形状正确的假数组挡下载。
    import numpy as np

    fake = (
        np.zeros((200, 3, 32, 32), dtype=np.float32),
        np.zeros((200,), dtype=np.int64),
        np.zeros((200, 3, 32, 32), dtype=np.float32),
        np.zeros((200,), dtype=np.int64),
    )
    with patch(_VISION_BUILD, return_value=fake) as mocked:
        ctx = prepare_trainer_run(nodes, edges, tr.id, validate_only=True)
        mocked.assert_called()
    assert ctx.training_steps == 12000
    # flattenOutput=true → MLP 走稠密 CE 任务(非 vision_classification)。
    assert ctx.trainer_task == "cross_entropy_dense"

    # 钉死 Fig-1 捕获接线，seed 工具改动不得静默破坏双轴图。
    norm_obs = next(n for n in nodes if n.type == "observable_last_layer_weight_norm")
    tviz = next(n for n in nodes if n.type == "training_visualization")
    norm_viz = next(
        n for n in nodes
        if n.type == "observable_viz" and (n.data or {}).get("pairedObservableId") == norm_obs.id
    )
    assert (norm_viz.data or {}).get("pairedTrainerId") == tr.id
    handle_pairs = {
        (e.source, e.target, e.sourceHandle, e.targetHandle) for e in edges
    }
    assert (tviz.id, table.id, "out_tensor_list", "stream") in handle_pairs
    assert (norm_viz.id, table.id, "out_tensor", "stream") in handle_pairs
    # accuracy viz 是佐证线,刻意不进曲线表。
    acc_viz_ids = {
        n.id for n in nodes
        if n.type == "observable_viz" and (n.data or {}).get("vizVariant") == "accuracy"
    }
    assert not any(src in acc_viz_ids and tgt == table.id for (src, tgt, _, _) in handle_pairs)


def test_keskar_fig23_template_compiles_sb_lb_stack() -> None:
    entry = _load_template("repro-keskar-fig23-sb-lb")
    nodes, edges = _nodes_edges(entry)
    types = {n.type for n in nodes}
    assert "keskar_c1_c2_cnn_model" in types
    assert "cifar10_dataset" in types
    assert "mnist_dataset" not in types
    assert "parametric_path_sampler" in types
    assert "curve_series_table" in types
    assert "model_checkpoint" in types

    trainers = [n for n in nodes if n.type == "trainer"]
    assert len(trainers) == 2
    trainers_by_batch = {int((n.data or {})["batchSize"]): n for n in trainers}
    assert set(trainers_by_batch) == {256, 5000}

    with patch(_VISION_BUILD) as mocked:
        contexts = {
            batch: prepare_trainer_run(nodes, edges, trainer.id, validate_only=True)
            for batch, trainer in trainers_by_batch.items()
        }
        mocked.assert_not_called()

    expected_steps = {256: 19600, 5000: 1000}
    expected_log_frequency = {256: 196, 5000: 10}
    for batch, trainer in trainers_by_batch.items():
        data = trainer.data or {}
        assert data["trainingLengthMode"] == "epochs"
        assert data["trainingEpochs"] == 100
        assert data["trainingSteps"] == expected_steps[batch]
        assert data["logFrequency"] == expected_log_frequency[batch]
        assert contexts[batch].training_steps == expected_steps[batch]
        assert contexts[batch].training_data_epochs == 100
        assert contexts[batch].trainer_task == "vision_classification"

    sampler = next(n for n in nodes if n.type == "parametric_path_sampler")
    assert "interpolationMode" not in (sampler.data or {})
    assert (sampler.data or {})["alphaSteps"] == 25
    comment = next(n for n in nodes if n.type == "comment")
    assert "per-batch BatchNorm statistics" in str((comment.data or {}).get("text"))

    curve_viz = {
        str((n.data or {}).get("instanceTitle")): n
        for n in nodes
        if n.type == "curve_series_viz"
    }
    assert (curve_viz["Fig 2 accuracy plot"].data or {})["plotXMode"] == "epoch"
    assert (curve_viz["Fig 3 path plot"].data or {})["plotXMode"] == "param"

    def has_edge(source: str, target: str, source_handle: str, target_handle: str) -> bool:
        return any(
            e.source == source
            and e.target == target
            and e.sourceHandle == source_handle
            and e.targetHandle == target_handle
            for e in edges
        )

    checkpoints = {
        str((n.data or {}).get("instanceTitle")): n
        for n in nodes
        if n.type == "model_checkpoint"
    }
    assert has_edge(
        trainers_by_batch[256].id,
        checkpoints["Checkpoint SB (B=256)"].id,
        "checkpoint",
        "model_checkpoint",
    )
    assert has_edge(
        trainers_by_batch[5000].id,
        checkpoints["Checkpoint LB (B=5000)"].id,
        "checkpoint",
        "model_checkpoint",
    )

    fig2_table = next(
        n for n in nodes
        if n.type == "curve_series_table" and (n.data or {}).get("instanceTitle") == "Fig 2 accuracy curves"
    )
    accuracy_viz = [n for n in nodes if n.type == "observable_viz"]
    assert len(accuracy_viz) == 2
    assert all(has_edge(n.id, fig2_table.id, "out_tensor", "stream") for n in accuracy_viz)

    fig3_table = next(
        n for n in nodes
        if n.type == "curve_series_table" and (n.data or {}).get("instanceTitle") == "Fig 3 path curves"
    )
    assert (fig3_table.data or {})["captureMetrics"] == []

    dataset = next(n for n in nodes if n.type == "cifar10_dataset")
    model = next(n for n in nodes if n.type == "keskar_c1_c2_cnn_model")
    for target in [*trainers, sampler]:
        assert has_edge(dataset.id, target.id, "dataset", "dataset")
        assert has_edge(model.id, target.id, "model", "model")
