"""cyclic_batch_schedule — OptimizerDef-channel thin definition.

trainer 的 batch_schedule socket 专属卫星(零 provider)。family
trainer_batch_schedule 单员——connectionRules 的 trainer batch_schedule
规则按 type 恰等收紧。custom 组件 CyclicBatchScheduleNode。
"""
from __future__ import annotations

from comfy_research.nodes.registry import optimizer_def
from comfy_research.nodes.schema import EnumField, FrontendSpec, IntField, OptimizerDef

DEF = optimizer_def(
    OptimizerDef(
        type="cyclic_batch_schedule",
        label="Cyclic batch schedule",
        hint="Cyclic batch size (Jastrzębski Fig 1 left CBS). Wire to trainer **batch schedule** socket.",
        family=("trainer_batch_schedule",),
        fields=(
            IntField(key="batchMin", label="Batch Min", default=128, min=1),
            IntField(key="batchMax", label="Batch Max", default=640, min=1),
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
        frontend=FrontendSpec(component_key="CyclicBatchScheduleNode"),
    )
)
