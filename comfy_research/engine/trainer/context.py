"""TrainerRunContext: the prepare→train handoff object (extracted from trainer_run)."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

import torch
import torch.nn as nn

from comfy_research.engine.losses.loss_builders import TrainerTask
from comfy_research.schemas.graph import Edge, Node


@dataclass
class TrainerRunContext:
    """Mutable training state built before streaming so HTTPException can be raised before SSE headers."""

    nodes: list[Node]
    edges: list[Edge]
    nmap: dict[str, Node]
    trainer_node_id: str
    observable_nodes: list[Node]
    model: nn.Module
    optimizer: torch.optim.Optimizer
    criterion: nn.Module
    loss_scale: float
    trainer_task: TrainerTask
    x_t: torch.Tensor
    y_t: torch.Tensor
    x_test_t: torch.Tensor | None
    y_test_t: torch.Tensor | None
    training_steps: int
    log_frequency: int
    log_schedule: str
    log_samples: int
    log_aggregation: str
    log_timing: str
    test_evaluation: str
    start_step: int
    loss_history: list[float]
    test_loss_history: list[float]
    reg_loss_history: list[float]
    step_ticks: list[int]
    epoch_ticks: list[float]
    observable_metric_histories: dict[str, list[float]]
    observable_embedding_histories: dict[str, list[list[list[float]]]]
    observable_attention_slice_histories: dict[str, list[dict[str, object]]]
    depth: int
    test_size: int
    resuming: bool
    hessian_oversized_mode: Literal["off", "skip", "force"]
    train_batch_size: int
    minibatch_sampling: str
    train_streaming: bool
    dataset_base_seed: int
    train_materialize: Callable[[int], tuple[torch.Tensor, torch.Tensor]]
    test_materialize: Callable[[int], tuple[torch.Tensor | None, torch.Tensor | None]]
    minibatch_perm_seed: int
    device: torch.device
    grad_clip_max_norm: float = 0.0
    lr_warmup_steps: int = 0
    lr_schedule: str = "constant"
    cosine_lr_min_fraction: float = 0.0
    exponential_lr_decay_factor: float = 0.95
    exponential_lr_decay_epochs: int = 1
    cyclic_lr_min: float = 0.0
    cyclic_lr_max: float = 0.0
    cyclic_lr_cycle_steps: int = 0
    cyclic_batch_min: int = 0
    cyclic_batch_max: int = 0
    cyclic_batch_cycle_steps: int = 0
    cyclic_schedule_mode: str = "discrete_epoch"
    cyclic_cycle_epochs: int = 10
    cyclic_steps_per_epoch: int = 1
    training_data_epochs: int = 0
    train_size: int = 0
    token_lm_seq_len: int = 0
    optimizer_base_group_lrs: tuple[float, ...] = ()
    weight_reg_loss_nodes: list[Node] = field(default_factory=list)
    l2_projection_nodes: list[Node] = field(default_factory=list)
    disable_extra_observables: bool = False
    training_recipe: str = "standard"
    training_recipe_pixel_mean: Any = None
    training_recipe_global_std: float = 1.0
    run_seed: int = 0
