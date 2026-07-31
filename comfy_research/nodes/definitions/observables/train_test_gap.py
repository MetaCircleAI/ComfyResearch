"""observable_train_test_gap — NodeDef-channel definition + recorder.

Generic component; user-variant scalar path (user_whitelisted derives the
observable_viz user whitelist entry).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from comfy_research.nodes.registry import observable_def, recorder_for
from comfy_research.nodes.schema import FrontendSpec, ObservableDef, SpawnSpec, VizSpec

if TYPE_CHECKING:  # pragma: no cover
    from comfy_research.engine.trainer.recorder import ObservableRecorder
    from comfy_research.schemas.graph import Node

TRAIN_TEST_GAP = observable_def(
    ObservableDef(
        type="observable_train_test_gap",
        label="Train vs test gap",
        hint="Logs test loss minus train loss when a train/test split exists.",
        viz=VizSpec(
            variant="user",
            title="Train vs test gap",
            info_markdown=(
                "**Train vs test gap** — test loss minus train loss at each logged step when a "
                "train/test split exists."
            ),
            spawns=True,
            user_whitelisted=True,
            spawn=SpawnSpec(kind="user_scalar", unit="test − train loss"),
        ),
        frontend=FrontendSpec(),
    )
)


@recorder_for(TRAIN_TEST_GAP)
def record(rec: "ObservableRecorder", on: "Node") -> None:
    """Body moved verbatim from ObservableRecorder._record_train_test_gap."""
    loss_history = rec.loss_history
    observable_metric_histories = rec.observable_metric_histories
    test_loss_history = rec.test_loss_history
    if test_loss_history and len(test_loss_history) == len(loss_history):
        observable_metric_histories[on.id].append(
            float(test_loss_history[-1]) - float(loss_history[-1])
        )
    else:
        observable_metric_histories[on.id].append(float("nan"))
