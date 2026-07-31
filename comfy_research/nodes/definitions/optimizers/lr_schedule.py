"""lr_schedule — OptimizerDef-channel thin definition.

程序化生成自 manifest(labels title-case,fields=defaults 推断)。卫星型:
不在 trainer optimizer socket 上(挂 optimizer 的 lr_schedule handle),
**零 provider**——warmup/cosine 语义由 trainer runtime 实现；sweep 轴
经 ``collectAxes`` 的 optimizer 卫星链收集并去重。
custom 组件 LrScheduleNode。
"""
from __future__ import annotations

from comfy_research.nodes.registry import optimizer_def
from comfy_research.nodes.schema import EnumField, FloatField, FrontendSpec, IntField, OptimizerDef

DEF = optimizer_def(
    OptimizerDef(
        type="lr_schedule",
        label="LR schedule",
        hint="Warmup steps, constant vs cosine decay, and cosine floor — wire to an optimizer’s **lr schedule** socket.",
        # trainer_lr_schedule family(lr/mup/cyclic 恰三员)——
        # connectionRules 的 lr-schedule 源资格由此派生。
        family=("trainer_lr_schedule",),
        fields=(
            IntField(key="lrWarmupSteps", label='Lr Warmup Steps', default=0, min=0),
            EnumField(
                key="lrSchedule",
                label="Lr Schedule",
                default="constant",
                options=("constant", "cosine", "stable_stable_decay", "exponential_epoch"),
            ),
            FloatField(key="cosineLrMinFraction", label='Cosine Lr Min Fraction', default=0, min=0),
            FloatField(key="exponentialDecayFactor", label="Exponential Decay Factor", default=0.95, min=0, max=1),
            IntField(key="exponentialDecayEpochs", label="Exponential Decay Every Epochs", default=1, min=1),
        ),
        frontend=FrontendSpec(component_key="LrScheduleNode", codegen_key="lr_schedule"),
    )
)
