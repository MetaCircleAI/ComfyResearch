"""NodeDef + provider registries. Definitions are pure data (frozen); providers
register here via recorder_for(DEF) — never by mutating the def."""
from __future__ import annotations

import importlib
import pkgutil
from typing import Callable

from comfy_research.nodes.schema import DatasetDef, FrontendNodeDef, InitializationDef, LossDef, ModelDef, ObservableDef, OptimizerDef, RecorderFn, TrainerDef

NODE_DEFS: dict[str, ObservableDef] = {}
DATASET_DEFS: dict[str, DatasetDef] = {}
DATASET_PREVIEWS: dict[str, Callable] = {}
DATASET_MATERIALIZERS: dict[str, Callable] = {}
MODEL_DEFS: dict[str, ModelDef] = {}
MODEL_BUILDER_PROVIDERS: dict[str, Callable] = {}
OPTIMIZER_DEFS: dict[str, OptimizerDef] = {}
OPTIMIZER_BUILDER_PROVIDERS: dict[str, Callable] = {}
LOSS_DEFS: dict[str, LossDef] = {}
LOSS_CRITERION_PROVIDERS: dict[str, Callable] = {}
INITIALIZATION_DEFS: dict[str, InitializationDef] = {}
TRAINER_DEFS: dict[str, TrainerDef] = {}
FRONTEND_DEFS: dict[str, FrontendNodeDef] = {}
PROVIDERS: dict[str, RecorderFn] = {}

KNOWN_NODEDEF_MIGRATION_DEFERRED: dict[str, str] = {}


def observable_def(def_: ObservableDef) -> ObservableDef:
    if def_.type in NODE_DEFS or def_.type in DATASET_DEFS or def_.type in MODEL_DEFS or def_.type in OPTIMIZER_DEFS or def_.type in LOSS_DEFS or def_.type in INITIALIZATION_DEFS or def_.type in TRAINER_DEFS or def_.type in FRONTEND_DEFS:
        raise RuntimeError(f"duplicate NodeDef registration: {def_.type!r}")
    NODE_DEFS[def_.type] = def_
    return def_


def model_def(def_: ModelDef) -> ModelDef:
    if def_.type in MODEL_DEFS or def_.type in NODE_DEFS or def_.type in DATASET_DEFS or def_.type in OPTIMIZER_DEFS or def_.type in LOSS_DEFS or def_.type in INITIALIZATION_DEFS or def_.type in TRAINER_DEFS or def_.type in FRONTEND_DEFS:
        raise RuntimeError(f"duplicate NodeDef registration: {def_.type!r}")
    MODEL_DEFS[def_.type] = def_
    return def_


def model_builder_for(def_: ModelDef) -> Callable[[Callable], Callable]:
    """runtime provider(签名恒等 model_builders.ModelBuilder)。仅 27 个
    MODEL_BUILDERS 成员注册;atomic/辅助节点零 provider。"""

    def register(fn: Callable) -> Callable:
        if def_.type in MODEL_BUILDER_PROVIDERS:
            raise RuntimeError(f"duplicate model builder registration: {def_.type!r}")
        MODEL_BUILDER_PROVIDERS[def_.type] = fn
        return fn

    return register


def all_model_defs() -> tuple[ModelDef, ...]:
    load_definitions()
    return tuple(sorted(MODEL_DEFS.values(), key=lambda d: d.type))


def model_def_types() -> frozenset[str]:
    return frozenset(d.type for d in all_model_defs())


def model_defs_builders() -> dict[str, Callable]:
    load_definitions()
    validate_defs()
    return dict(MODEL_BUILDER_PROVIDERS)


def optimizer_def(def_: OptimizerDef) -> OptimizerDef:
    if (def_.type in OPTIMIZER_DEFS or def_.type in NODE_DEFS
            or def_.type in DATASET_DEFS or def_.type in MODEL_DEFS or def_.type in LOSS_DEFS
            or def_.type in INITIALIZATION_DEFS or def_.type in TRAINER_DEFS or def_.type in FRONTEND_DEFS):
        raise RuntimeError(f"duplicate NodeDef registration: {def_.type!r}")
    OPTIMIZER_DEFS[def_.type] = def_
    return def_


def optimizer_builder_for(def_: OptimizerDef) -> Callable[[Callable], Callable]:
    """runtime provider(签名恒等 optimizer_builders.OptimizerBuilder:
    (model, OptimizerBuildConfig) -> torch.optim.Optimizer)。仅 7 个
    OPTIMIZER_BUILDERS 成员注册;lr_schedule/mup_lr_schedule 零 provider。"""

    def register(fn: Callable) -> Callable:
        if def_.type in OPTIMIZER_BUILDER_PROVIDERS:
            raise RuntimeError(f"duplicate optimizer builder registration: {def_.type!r}")
        OPTIMIZER_BUILDER_PROVIDERS[def_.type] = fn
        return fn

    return register


def all_optimizer_defs() -> tuple[OptimizerDef, ...]:
    load_definitions()
    return tuple(sorted(OPTIMIZER_DEFS.values(), key=lambda d: d.type))


def optimizer_def_types() -> frozenset[str]:
    return frozenset(d.type for d in all_optimizer_defs())


def optimizer_defs_builders() -> dict[str, Callable]:
    load_definitions()
    validate_defs()
    return dict(OPTIMIZER_BUILDER_PROVIDERS)


def loss_def(def_: LossDef) -> LossDef:
    if (def_.type in LOSS_DEFS or def_.type in NODE_DEFS or def_.type in DATASET_DEFS
            or def_.type in MODEL_DEFS or def_.type in OPTIMIZER_DEFS
            or def_.type in INITIALIZATION_DEFS or def_.type in TRAINER_DEFS or def_.type in FRONTEND_DEFS):
        raise RuntimeError(f"duplicate NodeDef registration: {def_.type!r}")
    LOSS_DEFS[def_.type] = def_
    return def_


def loss_criterion_for(def_: LossDef) -> Callable[[Callable], Callable]:
    """runtime provider(签名恒等 loss_builders.LossCriterionBuilder:
    (loss_d, LossCriterionContext) -> nn.Module)。仅 3 primary 注册;
    l1/l2/l2_projection/kan_reg 零 provider。"""

    def register(fn: Callable) -> Callable:
        if def_.type in LOSS_CRITERION_PROVIDERS:
            raise RuntimeError(f"duplicate loss criterion registration: {def_.type!r}")
        LOSS_CRITERION_PROVIDERS[def_.type] = fn
        return fn

    return register


def all_loss_defs() -> tuple[LossDef, ...]:
    load_definitions()
    return tuple(sorted(LOSS_DEFS.values(), key=lambda d: d.type))


def loss_def_types() -> frozenset[str]:
    return frozenset(d.type for d in all_loss_defs())


def loss_defs_criteria() -> dict[str, Callable]:
    load_definitions()
    validate_defs()
    return dict(LOSS_CRITERION_PROVIDERS)


def initialization_def(def_: InitializationDef) -> InitializationDef:
    if (def_.type in INITIALIZATION_DEFS or def_.type in NODE_DEFS or def_.type in DATASET_DEFS
            or def_.type in MODEL_DEFS or def_.type in OPTIMIZER_DEFS or def_.type in LOSS_DEFS
            or def_.type in TRAINER_DEFS or def_.type in FRONTEND_DEFS):
        raise RuntimeError(f"duplicate NodeDef registration: {def_.type!r}")
    INITIALIZATION_DEFS[def_.type] = def_
    return def_


def all_initialization_defs() -> tuple[InitializationDef, ...]:
    load_definitions()
    return tuple(sorted(INITIALIZATION_DEFS.values(), key=lambda d: d.type))


def initialization_def_types() -> frozenset[str]:
    return frozenset(d.type for d in all_initialization_defs())


def trainer_def(def_: TrainerDef) -> TrainerDef:
    if (def_.type in TRAINER_DEFS or def_.type in NODE_DEFS or def_.type in DATASET_DEFS
            or def_.type in MODEL_DEFS or def_.type in OPTIMIZER_DEFS or def_.type in LOSS_DEFS
            or def_.type in INITIALIZATION_DEFS or def_.type in FRONTEND_DEFS):
        raise RuntimeError(f"duplicate NodeDef registration: {def_.type!r}")
    TRAINER_DEFS[def_.type] = def_
    return def_


def all_trainer_defs() -> tuple[TrainerDef, ...]:
    load_definitions()
    return tuple(sorted(TRAINER_DEFS.values(), key=lambda d: d.type))


def trainer_def_types() -> frozenset[str]:
    return frozenset(d.type for d in all_trainer_defs())


def frontend_node_def(def_: FrontendNodeDef) -> FrontendNodeDef:
    """第八通道:纯前端节点(零后端 runtime——无 provider 槽位,
    validate_defs 的 orphan 守卫对本通道无对应项,豁免是结构性的)。"""
    if (def_.type in FRONTEND_DEFS or def_.type in NODE_DEFS or def_.type in DATASET_DEFS
            or def_.type in MODEL_DEFS or def_.type in OPTIMIZER_DEFS or def_.type in LOSS_DEFS
            or def_.type in INITIALIZATION_DEFS or def_.type in TRAINER_DEFS):
        raise RuntimeError(f"duplicate NodeDef registration: {def_.type!r}")
    FRONTEND_DEFS[def_.type] = def_
    return def_


def all_frontend_defs() -> tuple[FrontendNodeDef, ...]:
    load_definitions()
    return tuple(sorted(FRONTEND_DEFS.values(), key=lambda d: d.type))


def frontend_def_types() -> frozenset[str]:
    return frozenset(d.type for d in all_frontend_defs())


# Dataset materializers and previews are independent optional providers.
# Auxiliary nodes such as input_sampler may intentionally register neither.
def dataset_def(def_: DatasetDef) -> DatasetDef:
    if def_.type in DATASET_DEFS or def_.type in NODE_DEFS or def_.type in MODEL_DEFS or def_.type in OPTIMIZER_DEFS or def_.type in LOSS_DEFS or def_.type in INITIALIZATION_DEFS or def_.type in TRAINER_DEFS or def_.type in FRONTEND_DEFS:
        raise RuntimeError(f"duplicate NodeDef registration: {def_.type!r}")
    DATASET_DEFS[def_.type] = def_
    return def_


def dataset_materializer_for(def_: DatasetDef) -> Callable[[Callable], Callable]:
    def register(fn: Callable) -> Callable:
        if def_.type in DATASET_MATERIALIZERS:
            raise RuntimeError(f"duplicate dataset materializer registration: {def_.type!r}")
        DATASET_MATERIALIZERS[def_.type] = fn
        return fn

    return register


def dataset_defs_materializers() -> dict[str, Callable]:
    load_definitions()
    validate_defs()
    return dict(DATASET_MATERIALIZERS)


def dataset_preview_for(def_: DatasetDef) -> Callable[[Callable], Callable]:
    def register(fn: Callable) -> Callable:
        if def_.type in DATASET_PREVIEWS:
            raise RuntimeError(f"duplicate dataset preview registration: {def_.type!r}")
        DATASET_PREVIEWS[def_.type] = fn
        return fn

    return register


def all_dataset_defs() -> tuple[DatasetDef, ...]:
    load_definitions()
    return tuple(sorted(DATASET_DEFS.values(), key=lambda d: d.type))


def dataset_def_types() -> frozenset[str]:
    return frozenset(d.type for d in all_dataset_defs())


def dataset_defs_previews() -> dict[str, Callable]:
    load_definitions()
    validate_defs()
    return dict(DATASET_PREVIEWS)


def recorder_for(def_: ObservableDef) -> Callable[[RecorderFn], RecorderFn]:
    def register(fn: RecorderFn) -> RecorderFn:
        if def_.type in PROVIDERS:
            raise RuntimeError(f"duplicate recorder registration: {def_.type!r}")
        PROVIDERS[def_.type] = fn
        return fn

    return register


_LOADED = False


def load_definitions() -> None:
    global _LOADED
    if _LOADED:
        return
    from comfy_research.nodes import definitions

    for mod in pkgutil.walk_packages(definitions.__path__, definitions.__name__ + "."):
        importlib.import_module(mod.name)
    _LOADED = True


def all_observable_defs() -> tuple[ObservableDef, ...]:
    load_definitions()
    return tuple(sorted(NODE_DEFS.values(), key=lambda d: d.type))


def observable_def_types() -> frozenset[str]:
    return frozenset(d.type for d in all_observable_defs())


def defs_recorders() -> dict[str, RecorderFn]:
    load_definitions()
    validate_defs()
    return dict(PROVIDERS)


def validate_defs() -> None:
    """半接线在根上消灭:observable 定义与 recorder 双向完备;dataset 通道
    role-based(provider 槽位全可选,但孤儿 provider——注册了 provider 却无
    def——FATAL)。"""
    load_definitions()
    missing = set(NODE_DEFS) - set(PROVIDERS)
    orphaned = set(PROVIDERS) - set(NODE_DEFS)
    if missing:
        raise RuntimeError(f"ObservableDef(s) without recorder: {sorted(missing)}")
    if orphaned:
        raise RuntimeError(f"recorder(s) without ObservableDef: {sorted(orphaned)}")
    ds_orphaned = set(DATASET_PREVIEWS) - set(DATASET_DEFS)
    if ds_orphaned:
        raise RuntimeError(f"dataset preview provider(s) without DatasetDef: {sorted(ds_orphaned)}")
    mat_orphaned = set(DATASET_MATERIALIZERS) - set(DATASET_DEFS)
    if mat_orphaned:
        raise RuntimeError(f"dataset materializer provider(s) without DatasetDef: {sorted(mat_orphaned)}")
    mb_orphaned = set(MODEL_BUILDER_PROVIDERS) - set(MODEL_DEFS)
    if mb_orphaned:
        raise RuntimeError(f"model builder provider(s) without ModelDef: {sorted(mb_orphaned)}")
    ob_orphaned = set(OPTIMIZER_BUILDER_PROVIDERS) - set(OPTIMIZER_DEFS)
    if ob_orphaned:
        raise RuntimeError(f"optimizer builder provider(s) without OptimizerDef: {sorted(ob_orphaned)}")
    lc_orphaned = set(LOSS_CRITERION_PROVIDERS) - set(LOSS_DEFS)
    if lc_orphaned:
        raise RuntimeError(f"loss criterion provider(s) without LossDef: {sorted(lc_orphaned)}")
