"""Branch-level provider contracts for the four-way trainer build.

Data flow: PrepareState -> _build_branch -> DatasetMaterializeRequest ->
DATASET_PROVIDERS[branch] -> DatasetArrays -> tensor conversion ->
ModelBuildRequest -> MODEL_PROVIDERS[branch] -> ModelBuildResult ->
PrepareState writes. Providers never see PrepareState — the prepare_finalize
stages own the state judgment, request assembly, and every state write.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol

from comfy_research.engine.trainer.dataset_arrays import DatasetArrays

BuildBranch = Literal["vision", "atomic", "vector", "token"]


def _build_branch(
    trainer_task: str,
    is_sequential_atomic: bool,
    *,
    image_diffusion: bool = False,
) -> BuildBranch:
    """The four-way dispatch, verbatim from build_training_objects_stage."""
    if trainer_task in ("mse_regression", "cross_entropy_dense", "diffusion_noise", "vision_classification"):
        if trainer_task == "vision_classification" or image_diffusion:
            return "vision"
        if is_sequential_atomic:
            return "atomic"
        return "vector"
    return "token"


@dataclass(frozen=True)
class DatasetMaterializeRequest:
    branch: BuildBranch
    ce_mem_a_slots: int
    combined_flat_chain: Any
    dd_test: dict[str, Any]
    dd_train: dict[str, Any]
    ds_test_raw: Any
    ds_train: Any
    edges: Any
    model_node: Any
    nmap: Any
    nodes: Any
    rng: Any
    seed: int
    test_size: int
    train_size: int
    trainer_task: str  # dense families condition on it as DATA (branch owns dispatch)
    binary_single_logit: bool = False


@dataclass(frozen=True)
class ModelBuildRequest:
    branch: BuildBranch
    arrays: DatasetArrays
    ce_mem_a_slots: int
    combined_flat_chain: Any
    ds_train: Any
    edges: Any
    model_node: Any
    nmap: Any
    seed: int
    want_kan_reg: bool


@dataclass(frozen=True)
class ModelBuildResult:
    model: Any
    depth: int
    stack_io: tuple[int, int] | None  # None = branch does not write stack_io


class DatasetProvider(Protocol):
    def __call__(self, req: DatasetMaterializeRequest) -> DatasetArrays: ...


class ModelProvider(Protocol):
    def __call__(self, req: ModelBuildRequest) -> ModelBuildResult: ...
