"""sgd_optimizer — OptimizerDef-channel definition + builder provider.

程序化生成自 manifest;富 UI 元数据(tooltip/aria/min/max/positiveOnly)与
info 文案逐字转写自 OVERRIDES + optimizerNodeInfoContent.ts(python 持真相)。
SchemaNode 泛型渲染三面:fields/ui/specCode(SPEC_CODE_ADAPTERS)。
``build_optimizer_stage`` maps node metadata to the runtime configuration.
"""
from __future__ import annotations

from comfy_research.nodes.registry import optimizer_builder_for, optimizer_def
from comfy_research.nodes.schema import FloatListField, FrontendSpec, OptimizerDef, UiSpec

INFO_TEXT = "Classic stochastic gradient descent with optional momentum.\n\nUse momentum close to 0 for plain SGD; values in [0.8, 0.99] often smooth noisy updates.\n\nweight decay is PyTorch's L2 penalty coefficient on parameters (0 disables).\n\nFor sweeps, comma-separated values run as a Cartesian product with trainer and model settings.\n\n**lr schedule** input: one socket on the left. Wire a **lr_schedule** node and/or a **mup_lr_schedule** node (runtime applies muP multipliers only with **adam_optimizer**)."

DEF = optimizer_def(
    OptimizerDef(
        type="sgd_optimizer",
        label='SGD',
        hint="Stochastic gradient descent optimizer with momentum.",
        family=("optimizer_node",),
        fields=(
            FloatListField(key="learningRate", label='learning rate', default=0.01, positive_only=True, tooltip='e.g. 0.01 — comma-separated for multiple runs', aria_label='SGD learning rate'),
            FloatListField(key="momentum", label='momentum', default=0, min=0, max=1, aria_label='SGD momentum'),
            FloatListField(key="weightDecay", label='weight decay', default=0, min=0, tooltip='L2 penalty coefficient λ in PyTorch SGD (0 disables)', aria_label='SGD weight decay'),
        ),
        ui=UiSpec(
            accent="optimizer",
            socket_rows="optimizerLrSchedule",
            code_kind="optimizer",
            info_title='SGD optimizer',
            info_text=INFO_TEXT,
        ),
        frontend=FrontendSpec(component_key="SchemaNode", codegen_key="sgd_optimizer", spec_code_key="sgd_optimizer"),
    )
)


@optimizer_builder_for(DEF)
def build(model, config):
    from comfy_research.engine.optimizers.optimizer_builders import _build_sgd_optimizer

    return _build_sgd_optimizer(model, config)
