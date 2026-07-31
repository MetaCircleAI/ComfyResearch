"""teacher_dataset — NodeDef-channel definition for the dataset channel.

程序化生成自 manifest。materialization keeps teacher dispatch ahead of dense
families and the mixer early return ahead of hooks. Teacher fields are not sweepable.
"""
from __future__ import annotations

from comfy_research.nodes.registry import dataset_def, dataset_preview_for
from comfy_research.nodes.schema import DatasetDef, EnumField, FrontendSpec, InPort, PortAccept

DEF = dataset_def(
    DatasetDef(
        type="teacher_dataset",
        label="Teacher dataset",
        hint="Regression pairs y = f_teacher(x): wire an MLP teacher + train/test sample tensors from input sampler nodes.",
        family=("vector_regression_dataset", "activation_sample_dataset", "canvas_dataset_source", "canvas_activation_dataset_source", "canvas_trainer_autoconnect_dataset",),
        fields=(
            EnumField(key="samplingMode", label="Sampling Mode", default='fixed', sweepable=False),
        ),
        # 试点:model 口 = cascade 三谓词逐字转写(fullModelModelSocket /
        # fullModelTensorIoSource / combinedModelModelModeOut;source_io_mode 首战);
        # train/test_input = input_sampler 纯对。
        ports=(
            InPort(id="model", accepts=(
                PortAccept(handles=("model",), source_family="canvas_full_model", source_io_mode="model"),
                PortAccept(handles=("tensor_out", "tensor"), source_family="canvas_full_model", source_io_mode="input-output"),
                PortAccept(handles=("tensor", "model"), source_type="combined_model", source_io_mode="model"),
            )),
            InPort(id="train_input", accepts=(
                PortAccept(handles=("sample_tensor",), source_type="input_sampler"),
            )),
            InPort(id="test_input", accepts=(
                PortAccept(handles=("sample_tensor",), source_type="input_sampler"),
            )),
        ),
        frontend=FrontendSpec(component_key="TeacherDatasetNode", codegen_key="teacher_dataset"),
    )
)

def _preview_teacher(req):
    """teacher preview 体(engine 移层版),graph 上下文经 DatasetPreviewRequest。"""
    from comfy_research.engine.datasets.dataset_preview_helpers import _build_teacher_arrays

    return _build_teacher_arrays(req.dataset_node_id, req.data, req.graph_nodes, req.graph_edges)


dataset_preview_for(DEF)(_preview_teacher)
