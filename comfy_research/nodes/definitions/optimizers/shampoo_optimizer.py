"""shampoo_optimizer — OptimizerDef-channel definition + builder provider.

程序化生成自 manifest;富 UI 元数据(tooltip/aria/min/max/positiveOnly)与
info 文案逐字转写自 OVERRIDES + optimizerNodeInfoContent.ts(python 持真相)。
SchemaNode 泛型渲染三面:fields/ui/specCode(SPEC_CODE_ADAPTERS)。
``build_optimizer_stage`` maps node metadata to the runtime configuration.
"""
from __future__ import annotations

from comfy_research.nodes.registry import optimizer_builder_for, optimizer_def
from comfy_research.nodes.schema import FloatListField, IntListField, FrontendSpec, OptimizerDef, UiSpec

INFO_TEXT = 'Shampoo is a matrix-preconditioned optimizer.\n\nIt keeps per-dimension gradient statistics and applies inverse-root preconditioners to the update.\n\nprecond freq controls how often inverse roots are refreshed; max matrix dim skips very large dimensions to avoid oversized preconditioner matrices.\n\nweight decay is decoupled from the Shampoo preconditioned update.\n\nOptional **lr schedule** input: one socket, matching Adam/SGD. muP grouping is not applied with Shampoo in this build.'

DEF = optimizer_def(
    OptimizerDef(
        type="shampoo_optimizer",
        label='Shampoo',
        hint="Shampoo matrix-preconditioned optimizer.",
        family=("optimizer_node",),
        fields=(
            FloatListField(key="learningRate", label='learning rate', default=0.01, positive_only=True, tooltip='e.g. 0.01, comma-separated for multiple runs', aria_label='Shampoo learning rate'),
            FloatListField(key="momentum", label='momentum', default=0, min=0, max=1, aria_label='Shampoo momentum'),
            FloatListField(key="epsilon", label='epsilon', default=1e-08, positive_only=True, aria_label='Shampoo epsilon'),
            FloatListField(key="weightDecay", label='weight decay', default=0, min=0, tooltip='Decoupled weight decay coefficient', aria_label='Shampoo weight decay'),
            IntListField(key="preconditionFrequency", label='precond freq', default=10, min=1, tooltip='Optimizer steps between inverse-root refreshes', aria_label='Shampoo precondition frequency'),
            IntListField(key="maxPreconditionerDim", label='max matrix dim', default=1024, min=1, tooltip='Skip full preconditioner matrices above this dimension', aria_label='Shampoo max preconditioner dimension'),
        ),
        ui=UiSpec(
            accent="optimizer",
            socket_rows="optimizerLrSchedule",
            code_kind="optimizer",
            info_title='Shampoo optimizer',
            info_text=INFO_TEXT,
        ),
        frontend=FrontendSpec(component_key="SchemaNode", codegen_key="shampoo_optimizer", spec_code_key="shampoo_optimizer"),
    )
)


@optimizer_builder_for(DEF)
def build(model, config):
    from comfy_research.engine.optimizers.optimizer_builders import _build_shampoo_optimizer

    return _build_shampoo_optimizer(model, config)
