"""observable_accuracy — NodeDef-channel definition + recorder.

Generic component. The user_scalar spawn omits unit, so the visualization has
no vizYAxisLabel key. ``observable_viz.py`` writes a ``{id}::test`` subseries
when a test split exists.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from comfy_research.nodes.registry import observable_def, recorder_for
from comfy_research.nodes.schema import FrontendSpec, ObservableDef, SpawnSpec, VizSpec

if TYPE_CHECKING:  # pragma: no cover
    from comfy_research.engine.trainer.recorder import ObservableRecorder
    from comfy_research.schemas.graph import Node

ACCURACY = observable_def(
    ObservableDef(
        type="observable_accuracy",
        label="Accuracy",
        hint="Classification accuracy (top-1) from trainer predictions versus target labels at each log step.",
        viz=VizSpec(
            variant="accuracy",
            title="Accuracy",
            info_markdown=(
                "**Accuracy** — classification accuracy on the train (and test when available) "
                "batch used for logging."
            ),
            spawns=True,
            user_whitelisted=True,
            spawn=SpawnSpec(kind="user_scalar"),
        ),
        frontend=FrontendSpec(),
    )
)


@recorder_for(ACCURACY)
def record(rec: "ObservableRecorder", on: "Node") -> None:
    """Body moved verbatim from ObservableRecorder._record_accuracy."""
    from comfy_research.engine.trainer.eval_batches import (
        _batched_classification_accuracy,
    )

    model = rec.model
    observable_metric_histories = rec.observable_metric_histories
    trainer_task = rec.trainer_task
    x_test_log = rec._x_test_log
    xr = rec._xr
    y_test_log = rec._y_test_log
    yr = rec._yr
    if trainer_task == "diffusion_noise":
        observable_metric_histories[on.id].append(float("nan"))
        test_key = f"{on.id}::test"
        observable_metric_histories.setdefault(test_key, [])
        observable_metric_histories[test_key].append(float("nan"))
        return

    override = getattr(rec, "_metric_overrides", {}).get("train_accuracy")
    if override is None:
        override = _batched_classification_accuracy(
            model,
            xr,
            yr,
            batch_size=rec.eval_batch_size,
            trainer_task=trainer_task,
        )
    observable_metric_histories[on.id].append(float(override))

    if x_test_log is not None and y_test_log is not None:
        test_key = f"{on.id}::test"
        observable_metric_histories.setdefault(test_key, [])
        observable_metric_histories[test_key].append(
            _batched_classification_accuracy(
                model,
                x_test_log,
                y_test_log,
                batch_size=rec.eval_batch_size,
                trainer_task=trainer_task,
            )
        )
