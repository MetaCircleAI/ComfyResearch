"""mup_lr_schedule — OptimizerDef-channel thin definition.

程序化生成自 manifest。卫星型(lr_schedule 同款)+ **双路径 hazard**:可挂
optimizer 的 lr_schedule handle 或 mup_lr_schedule handle——collectAxes 的
mupLrAxisSources 去重结构 stay(仅轴收集调用换 generated)。零 provider
(muP 分组由 build_optimizer_stage/trainer runtime 处理)。custom 组件
MupLrScheduleNode。
"""
from __future__ import annotations

from comfy_research.nodes.registry import optimizer_def
from comfy_research.nodes.schema import FloatField, FrontendSpec, OptimizerDef

DEF = optimizer_def(
    OptimizerDef(
        type="mup_lr_schedule",
        label="MuP LR schedule",
        hint="μP Adam LR multipliers by parameter group (embed / hidden / output) — wire to the same **lr schedule** socket (Adam + supported models); combine with an lr_schedule node for warmup / cosine.",
        # trainer_lr_schedule family(lr/mup/cyclic 恰三员);mup 专属
        # 路径(mup_lr_schedule handle/去重)仍按 type 恒等,family 只放宽源资格。
        family=("trainer_lr_schedule",),
        fields=(
            FloatField(key="mupEmbedLrMult", label='Mup Embed Lr Mult', default=1, min=0),
            FloatField(key="mupHiddenLrMult", label='Mup Hidden Lr Mult', default=1, min=0),
            FloatField(key="mupOutputLrMult", label='Mup Output Lr Mult', default=1, min=0),
        ),
        frontend=FrontendSpec(component_key="MupLrScheduleNode", codegen_key="mup_lr_schedule"),
    )
)
