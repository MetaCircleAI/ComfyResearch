"""adam_optimizer — OptimizerDef-channel definition + builder provider.

程序化生成自 manifest;富 UI 元数据(tooltip/aria/min/max/positiveOnly)与
info 文案逐字转写自 OVERRIDES + optimizerNodeInfoContent.ts(python 持真相)。
SchemaNode 泛型渲染三面:fields/ui/specCode(SPEC_CODE_ADAPTERS)。
``build_optimizer_stage`` maps node metadata to the runtime configuration.
"""
from __future__ import annotations

from comfy_research.nodes.registry import optimizer_builder_for, optimizer_def
from comfy_research.nodes.schema import FloatListField, FrontendSpec, OptimizerDef, UiSpec

INFO_TEXT = "Adaptive moment estimation with first- and second-moment running averages.\n\nbeta1 controls momentum of the gradient mean; beta2 controls momentum of the squared-gradient estimate.\n\nepsilon is a small numerical stabilizer in the denominator.\n\nweight decay is PyTorch's L2 penalty coefficient on parameters (0 disables).\n\nOptional **lr schedule** input: one socket. **lr_schedule** (warmup / cosine) and **mup_lr_schedule** (layer-wise muP multipliers, Adam + supported models) each attach to the same port; you can wire either, both, or neither."

DEF = optimizer_def(
    OptimizerDef(
        type="adam_optimizer",
        label='Adam',
        hint="Adam optimizer for training the connected MLP.",
        family=("optimizer_node",),
        fields=(
            FloatListField(key="learningRate", label='learning rate', default=0.001, positive_only=True, tooltip='e.g. 0.0001 or 1e-4 — comma-separated for multiple runs', aria_label='Learning rate'),
            FloatListField(key="beta1", label='beta1', default=0.9, min=0, max=1, aria_label='Adam beta1'),
            FloatListField(key="beta2", label='beta2', default=0.999, min=0, max=1, aria_label='Adam beta2'),
            FloatListField(key="epsilon", label='epsilon', default=1e-08, positive_only=True, tooltip='e.g. 1e-8', aria_label='Adam epsilon'),
            FloatListField(key="weightDecay", label='weight decay', default=0, min=0, tooltip='L2 penalty coefficient λ in PyTorch Adam (0 disables)', aria_label='Adam weight decay'),
        ),
        ui=UiSpec(
            accent="optimizer",
            socket_rows="optimizerLrSchedule",
            code_kind="optimizer",
            info_title='Adam optimizer',
            info_text=INFO_TEXT,
        ),
        frontend=FrontendSpec(component_key="SchemaNode", codegen_key="adam_optimizer", spec_code_key="adam_optimizer"),
    )
)


@optimizer_builder_for(DEF)
def build(model, config):
    from comfy_research.engine.optimizers.optimizer_builders import _build_adam_optimizer

    return _build_adam_optimizer(model, config)
