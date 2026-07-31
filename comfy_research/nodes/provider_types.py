"""Provider 协议类型(stdlib-only,与 schema 同级)。

放在 nodes 层而非 api 层:dataset def/provider 模块要
type против这些协议,若协议住在 api.dataset_tensor 会形成
api → nodes.registry → definitions.datasets → api 的 import cycle。
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DatasetMaterializeContext:
    """NodeDef-channel dataset materializer provider 的入参。

    语义 = "搬 family arm,不重解释 materialize":
    - ds_type 装 ``ds_train.type`` **原值**(NodeKind,非 def 字符串)——旧 arm 直接
      把它传给 engine builder;
    - input_dim/output_dim 就是 ``_resolve_linear_dataset_mlp_dims`` 在 hook 点
      算出的值;provider 不得自行重解析 dims。
    provider 返回 DatasetArrays(通常经 ``_materialize_paired_split`` 原体)。
    """

    ds_type: object  # NodeKind | str,原值透传
    rng: object
    dd_train: dict
    dd_test: dict
    train_size: int
    test_size: int
    input_dim: int
    output_dim: int
    # _materialize_linear_like 用它判定 test 目标读 dd_test
    # 还是 dd_train;hook 必须传 req.ds_test_raw,不能靠默认值。
    ds_test_raw: object = None


@dataclass(frozen=True)
class DatasetPreviewRequest:
    """NodeDef-channel dataset preview provider 的入参。

    original_type 保留 alias normalization 前的真实 type(ai4science 系映射到
    linear_dataset 后 provider 仍能区分);graph_* 供 teacher 等图上下文 preview。
    """

    original_type: str
    effective_type: str
    data: dict
    graph_nodes: list = field(default_factory=list)
    graph_edges: list = field(default_factory=list)
    dataset_node_id: str = ""
