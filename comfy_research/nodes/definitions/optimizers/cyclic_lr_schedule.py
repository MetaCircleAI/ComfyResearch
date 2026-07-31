"""cyclic_lr_schedule — OptimizerDef-channel thin definition.

lr_schedule 同款卫星型(挂 optimizer 的 lr_schedule handle,零 provider——
三角/离散周期语义在 trainer runtime)。family trainer_lr_schedule:
connectionRules 的 lr-schedule 源资格由该 family 派生(lr_schedule/
mup_lr_schedule/cyclic_lr_schedule 恰三员)。custom 组件 CyclicLrScheduleNode。
"""
from __future__ import annotations

from comfy_research.nodes.registry import optimizer_def
from comfy_research.nodes.schema import EnumField, FloatField, FrontendSpec, IntField, OptimizerDef

DEF = optimizer_def(
    OptimizerDef(
        type="cyclic_lr_schedule",
        label="Cyclic LR schedule",
        hint="Cyclic learning rate (Jastrzębski Fig 1 left CLR). Wire to optimizer **lr schedule** socket.",
        family=("trainer_lr_schedule",),
        fields=(
            FloatField(key="lrMin", label="Lr Min", default=0.001, min=0),
            FloatField(key="lrMax", label="Lr Max", default=0.005, min=0),
            IntField(key="cycleLengthEpochs", label="Cycle Length Epochs", default=10, min=1),
            IntField(key="refBatchSize", label="Ref Batch Size", default=128, min=1),
            IntField(key="cycleLengthSteps", label="Cycle Length Steps", default=0, min=0),
            EnumField(
                key="scheduleMode",
                label="Schedule Mode",
                default="discrete_epoch",
                options=("discrete_epoch", "square_epoch", "triangular_step"),
            ),
        ),
        frontend=FrontendSpec(component_key="CyclicLrScheduleNode"),
    )
)
