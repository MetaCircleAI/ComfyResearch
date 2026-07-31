"""soap_optimizer — OptimizerDef-channel definition + builder provider.

程序化生成自 manifest;富 UI 元数据(tooltip/aria/min/max/positiveOnly)与
info 文案逐字转写自 OVERRIDES + optimizerNodeInfoContent.ts(python 持真相)。
SchemaNode 泛型渲染三面:fields/ui/specCode(SPEC_CODE_ADAPTERS)。
``build_optimizer_stage`` maps node metadata to the runtime configuration.
"""
from __future__ import annotations

from comfy_research.nodes.registry import optimizer_builder_for, optimizer_def
from comfy_research.nodes.schema import FloatListField, IntListField, FrontendSpec, OptimizerDef, UiSpec

INFO_TEXT = 'SOAP runs Adam-style moments in eigenspaces induced by Shampoo statistics.\n\nbeta1 and beta2 control the Adam-style first and second moments in the current preconditioner basis.\n\nprecond freq controls how often Shampoo eigenspaces are refreshed; max matrix dim skips very large dimensions to avoid oversized preconditioner matrices.\n\nweight decay is decoupled from the SOAP update.\n\nOptional **lr schedule** input: one socket, matching Adam/SGD. muP grouping is not applied with SOAP in this build.'

DEF = optimizer_def(
    OptimizerDef(
        type="soap_optimizer",
        label='SOAP',
        hint="SOAP optimizer: Adam moments in Shampoo eigenspaces.",
        family=("optimizer_node",),
        fields=(
            FloatListField(key="learningRate", label='learning rate', default=0.0003, positive_only=True, tooltip='e.g. 0.0003 or 3e-4, comma-separated for multiple runs', aria_label='SOAP learning rate'),
            FloatListField(key="beta1", label='beta1', default=0.9, min=0, max=1, aria_label='SOAP beta1'),
            FloatListField(key="beta2", label='beta2', default=0.95, min=0, max=1, aria_label='SOAP beta2'),
            FloatListField(key="epsilon", label='epsilon', default=1e-08, positive_only=True, aria_label='SOAP epsilon'),
            FloatListField(key="weightDecay", label='weight decay', default=0, min=0, tooltip='Decoupled weight decay coefficient', aria_label='SOAP weight decay'),
            IntListField(key="preconditionFrequency", label='precond freq', default=10, min=1, tooltip='Optimizer steps between Shampoo-basis refreshes', aria_label='SOAP precondition frequency'),
            IntListField(key="maxPreconditionerDim", label='max matrix dim', default=1024, min=1, tooltip='Skip full preconditioner matrices above this dimension', aria_label='SOAP max preconditioner dimension'),
        ),
        ui=UiSpec(
            accent="optimizer",
            socket_rows="optimizerLrSchedule",
            code_kind="optimizer",
            info_title='SOAP optimizer',
            info_text=INFO_TEXT,
        ),
        frontend=FrontendSpec(component_key="SchemaNode", codegen_key="soap_optimizer", spec_code_key="soap_optimizer"),
    )
)


@optimizer_builder_for(DEF)
def build(model, config):
    from comfy_research.engine.optimizers.optimizer_builders import _build_soap_optimizer

    return _build_soap_optimizer(model, config)
