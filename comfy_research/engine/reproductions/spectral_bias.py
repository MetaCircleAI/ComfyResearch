"""Paper-faithful utilities for Rahaman et al. Figure 1(a).

The published figure was produced by the authors' released notebook.  That
notebook contains a non-standard "spectral norm" estimator for square hidden
layers (power iteration on ``W`` rather than ``W.T @ W`` plus a precedence
quirk in the final Rayleigh expression).  We preserve that estimator under an
explicit ``author_*`` name for figure fidelity and record a conventional
10-step singular-value estimate alongside it for auditability.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Callable

import numpy as np
import torch
import torch.nn as nn


@dataclass(frozen=True)
class SpectralBiasProtocol:
    """Fully specified Figure 1(a) protocol for one serial phase run."""

    samples: int = 200
    frequencies: tuple[int, ...] = (5, 10, 15, 20, 25, 30, 35, 40, 45, 50)
    amplitudes: tuple[float, ...] = (1.0,) * 10
    repeats: int = 10
    iterations: int = 80_000
    record_every: int = 100
    input_dim: int = 1
    output_dim: int = 1
    width: int = 256
    linear_layers: int = 6
    learning_rate: float = 3e-4
    adam_betas: tuple[float, float] = (0.9, 0.999)
    adam_eps: float = 1e-8
    power_iterations: int = 10
    phase_seed_base: int = 1703
    model_seed_base: int = 2703
    metric_seed_base: int = 3703
    endpoint: bool = False

    def validate(self) -> None:
        if self.samples < 2:
            raise ValueError("samples must be at least 2")
        if not self.frequencies or len(self.frequencies) != len(self.amplitudes):
            raise ValueError("frequencies and amplitudes must be non-empty and equal length")
        if any(frequency < 0 for frequency in self.frequencies):
            raise ValueError("frequencies must be non-negative")
        if any(amplitude <= 0 for amplitude in self.amplitudes):
            raise ValueError("amplitudes must be positive")
        if self.repeats < 1 or self.iterations < 1 or self.record_every < 1:
            raise ValueError("repeats, iterations, and record_every must be positive")
        if self.width < 1 or self.linear_layers < 2:
            raise ValueError("width must be positive and linear_layers must be at least 2")
        if self.power_iterations < 1:
            raise ValueError("power_iterations must be positive")
        if max(self.frequencies) >= self.samples // 2:
            raise ValueError("target frequencies must be below the one-sided Nyquist bin")

    @property
    def recorded_iterations(self) -> np.ndarray:
        # This mirrors the released notebook's ``range(NUM_ITER)`` and logging
        # condition ``iter_num % REC_FRQ == 0``.  A separate final prediction
        # is retained at exactly ``iterations`` for the Figure 2-style audit.
        return np.arange(0, self.iterations, self.record_every, dtype=np.int64)

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["frequencies"] = list(self.frequencies)
        payload["amplitudes"] = list(self.amplitudes)
        payload["adam_betas"] = list(self.adam_betas)
        payload["record_count"] = int(self.recorded_iterations.size)
        payload["last_recorded_iteration"] = int(self.recorded_iterations[-1])
        payload["final_prediction_iteration"] = int(self.iterations)
        return payload


class PaperSpectralBiasMlp(nn.Sequential):
    """Six-Linear-layer ReLU MLP matching the authors' ``make_model``."""

    def __init__(
        self,
        *,
        input_dim: int = 1,
        output_dim: int = 1,
        width: int = 256,
        linear_layers: int = 6,
        seed: int = 0,
    ) -> None:
        if linear_layers < 2:
            raise ValueError("linear_layers must be at least 2")
        torch.manual_seed(int(seed))
        layers: list[nn.Module] = [nn.Linear(input_dim, width), nn.ReLU()]
        for _ in range(linear_layers - 2):
            layers.extend((nn.Linear(width, width), nn.ReLU()))
        layers.append(nn.Linear(width, output_dim))
        super().__init__(*layers)

    @property
    def linear_modules(self) -> list[nn.Linear]:
        return [module for module in self if isinstance(module, nn.Linear)]


def make_phase_target(
    protocol: SpectralBiasProtocol,
    *,
    phase_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return endpoint-excluded grid, target, and phase fractions in [0, 1)."""
    protocol.validate()
    # np.random.rand in the released notebook uses MT19937/RandomState rather
    # than the newer Generator API, so retain that algorithm deliberately.
    rng = np.random.RandomState(int(phase_seed))
    phase_fractions = rng.rand(len(protocol.frequencies)).astype(np.float64)
    x = np.linspace(
        0.0,
        1.0,
        num=protocol.samples,
        endpoint=protocol.endpoint,
        dtype=np.float64,
    )
    target = np.zeros_like(x)
    for frequency, amplitude, phase in zip(
        protocol.frequencies,
        protocol.amplitudes,
        phase_fractions,
    ):
        target += float(amplitude) * np.sin(
            2.0 * math.pi * (float(frequency) * x + float(phase))
        )
    return x.astype(np.float32), target.astype(np.float32), phase_fractions


def normalized_dft_amplitudes(
    values: np.ndarray,
    *,
    frequencies: tuple[int, ...],
    amplitudes: tuple[float, ...],
) -> np.ndarray:
    """One-sided DFT amplitudes normalized exactly as the author notebook."""
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    spectrum = np.fft.fft(flat) / float(flat.size)
    selected = np.abs(spectrum[np.asarray(frequencies, dtype=np.int64)])
    return (2.0 * selected / np.asarray(amplitudes, dtype=np.float64)).astype(np.float64)


def author_power_iteration_estimate(
    weight: torch.Tensor,
    *,
    iterations: int = 10,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Reproduce the released notebook's square-layer estimator byte-for-byte in math.

    This is *not* a standard matrix spectral norm.  The final expression
    intentionally preserves Python's left-associative ``/`` and ``@``
    evaluation from ``(b.T @ A @ b) / b.T @ b``.
    """
    matrix = weight.detach()
    if matrix.ndim != 2 or int(matrix.shape[0]) != int(matrix.shape[1]):
        raise ValueError("author estimator is defined only for square matrices")
    vector = torch.randn(
        (int(matrix.shape[1]), 1),
        dtype=matrix.dtype,
        device=matrix.device,
        generator=generator,
    )
    for _ in range(int(iterations)):
        next_vector = matrix @ vector
        norm = torch.linalg.vector_norm(next_vector)
        vector = next_vector / norm.clamp_min(torch.finfo(matrix.dtype).tiny)
    # Keep the author's operator-precedence behavior.  Do not add denominator
    # parentheses here: doing so changes the plotted scale by roughly width.
    return (((vector.T @ matrix @ vector) / vector.T) @ vector).squeeze().abs()


def singular_norm_power_estimate(
    weight: torch.Tensor,
    *,
    iterations: int = 10,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Conventional top singular-value estimate via power iteration on W.T W."""
    matrix = weight.detach()
    if matrix.ndim != 2:
        raise ValueError("weight must be a matrix")
    vector = torch.randn(
        (int(matrix.shape[1]), 1),
        dtype=matrix.dtype,
        device=matrix.device,
        generator=generator,
    )
    vector = vector / torch.linalg.vector_norm(vector).clamp_min(torch.finfo(matrix.dtype).tiny)
    for _ in range(int(iterations)):
        next_vector = matrix.T @ (matrix @ vector)
        vector = next_vector / torch.linalg.vector_norm(next_vector).clamp_min(
            torch.finfo(matrix.dtype).tiny
        )
    return torch.linalg.vector_norm(matrix @ vector)


def layer_norm_estimates(
    model: PaperSpectralBiasMlp,
    *,
    iterations: int,
    author_generator: torch.Generator,
    singular_generator: torch.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Return author-figure and conventional estimates for all Linear layers."""
    author_values: list[float] = []
    singular_values: list[float] = []
    with torch.no_grad():
        for layer in model.linear_modules:
            weight = layer.weight
            if layer.in_features == layer.out_features:
                author_value = author_power_iteration_estimate(
                    weight,
                    iterations=iterations,
                    generator=author_generator,
                )
            else:
                # For the first/last vector-shaped matrices, Frobenius and
                # operator norm coincide; this is the authors' exact branch.
                author_value = torch.linalg.vector_norm(weight.detach())
            singular_value = singular_norm_power_estimate(
                weight,
                iterations=iterations,
                generator=singular_generator,
            )
            author_values.append(float(author_value.item()))
            singular_values.append(float(singular_value.item()))
    return np.asarray(author_values, dtype=np.float64), np.asarray(
        singular_values, dtype=np.float64
    )


@dataclass
class SpectralBiasRepeatResult:
    iterations: np.ndarray
    train_loss: np.ndarray
    normalized_amplitudes: np.ndarray
    author_layer_norms: np.ndarray
    singular_layer_norms: np.ndarray
    prediction_iterations: np.ndarray
    predictions: np.ndarray
    phase_fractions: np.ndarray
    final_prediction: np.ndarray


ProgressCallback = Callable[[int, int, bool], None]
CheckpointCallback = Callable[
    [
        int,
        PaperSpectralBiasMlp,
        torch.optim.Optimizer,
        dict[str, list[np.ndarray] | list[float] | list[int]],
        torch.Generator,
        torch.Generator,
    ],
    None,
]


def train_spectral_bias_repeat(
    protocol: SpectralBiasProtocol,
    *,
    repeat_index: int,
    device: str | torch.device = "cpu",
    progress: ProgressCallback | None = None,
    checkpoint_every: int = 0,
    checkpoint: CheckpointCallback | None = None,
    resume_state: dict[str, object] | None = None,
) -> tuple[SpectralBiasRepeatResult, PaperSpectralBiasMlp]:
    """Train one phase/model realization; callers serialize repeats explicitly."""
    protocol.validate()
    if repeat_index < 0 or repeat_index >= protocol.repeats:
        raise ValueError("repeat_index out of range")
    target_device = torch.device(device)
    phase_seed = protocol.phase_seed_base + int(repeat_index)
    model_seed = protocol.model_seed_base + int(repeat_index)
    metric_seed = protocol.metric_seed_base + int(repeat_index)
    x_np, y_np, phases = make_phase_target(protocol, phase_seed=phase_seed)
    x = torch.from_numpy(x_np).reshape(-1, 1).to(target_device)
    y = torch.from_numpy(y_np).reshape(-1, 1).to(target_device)
    model = PaperSpectralBiasMlp(
        input_dim=protocol.input_dim,
        output_dim=protocol.output_dim,
        width=protocol.width,
        linear_layers=protocol.linear_layers,
        seed=model_seed,
    ).to(target_device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=protocol.learning_rate,
        betas=protocol.adam_betas,
        eps=protocol.adam_eps,
    )
    loss_fn = nn.MSELoss()
    author_generator = torch.Generator(device=target_device)
    author_generator.manual_seed(metric_seed)
    singular_generator = torch.Generator(device=target_device)
    singular_generator.manual_seed(metric_seed + 1_000_003)

    recorded: dict[str, list[np.ndarray] | list[float] | list[int]] = {
        "iterations": [],
        "train_loss": [],
        "normalized_amplitudes": [],
        "author_layer_norms": [],
        "singular_layer_norms": [],
        "prediction_iterations": [],
        "predictions": [],
    }
    start_iteration = 0
    if resume_state is not None:
        saved_repeat = int(resume_state.get("repeat_index", repeat_index))
        if saved_repeat != repeat_index:
            raise ValueError("resume checkpoint repeat_index does not match")
        model_state = resume_state.get("model_state")
        optimizer_state = resume_state.get("optimizer_state")
        saved_recorded = resume_state.get("recorded")
        if not isinstance(model_state, dict) or not isinstance(optimizer_state, dict):
            raise ValueError("resume checkpoint is missing model/optimizer state")
        if not isinstance(saved_recorded, dict):
            raise ValueError("resume checkpoint is missing recorded histories")
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
        start_iteration = int(resume_state.get("completed_iterations", 0))
        if start_iteration < 0 or start_iteration > protocol.iterations:
            raise ValueError("resume checkpoint completed_iterations is invalid")
        for key in recorded:
            row = saved_recorded.get(key, [])
            if not isinstance(row, list):
                raise ValueError(f"resume checkpoint history {key!r} is invalid")
            recorded[key] = row
        author_state = resume_state.get("author_generator_state")
        singular_state = resume_state.get("singular_generator_state")
        if isinstance(author_state, torch.Tensor):
            author_generator.set_state(author_state)
        if isinstance(singular_state, torch.Tensor):
            singular_generator.set_state(singular_state)
    prediction_steps = {100, 1_000, 10_000}
    model.train()
    for iteration in range(start_iteration, protocol.iterations):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x)
        loss = loss_fn(prediction, y)
        loss.backward()

        is_record = iteration % protocol.record_every == 0
        if is_record:
            prediction_np = prediction.detach().cpu().numpy().reshape(-1)
            author_norms, singular_norms = layer_norm_estimates(
                model,
                iterations=protocol.power_iterations,
                author_generator=author_generator,
                singular_generator=singular_generator,
            )
            cast_iterations = recorded["iterations"]
            cast_losses = recorded["train_loss"]
            cast_amplitudes = recorded["normalized_amplitudes"]
            cast_author = recorded["author_layer_norms"]
            cast_singular = recorded["singular_layer_norms"]
            assert isinstance(cast_iterations, list)
            assert isinstance(cast_losses, list)
            assert isinstance(cast_amplitudes, list)
            assert isinstance(cast_author, list)
            assert isinstance(cast_singular, list)
            cast_iterations.append(iteration)
            cast_losses.append(float(loss.item()))
            cast_amplitudes.append(
                normalized_dft_amplitudes(
                    prediction_np,
                    frequencies=protocol.frequencies,
                    amplitudes=protocol.amplitudes,
                )
            )
            cast_author.append(author_norms)
            cast_singular.append(singular_norms)
            if iteration in prediction_steps:
                cast_prediction_steps = recorded["prediction_iterations"]
                cast_predictions = recorded["predictions"]
                assert isinstance(cast_prediction_steps, list)
                assert isinstance(cast_predictions, list)
                cast_prediction_steps.append(iteration)
                cast_predictions.append(prediction_np.copy())

        # Record all metrics from the same pre-update model snapshot; update only
        # after the snapshot is complete so frequency and layer curves align.
        optimizer.step()
        completed = iteration + 1
        if (
            checkpoint_every > 0
            and checkpoint is not None
            and completed % checkpoint_every == 0
        ):
            checkpoint(
                completed,
                model,
                optimizer,
                recorded,
                author_generator,
                singular_generator,
            )
        if progress is not None:
            progress(completed, protocol.iterations, is_record)

    model.eval()
    with torch.no_grad():
        final_prediction = model(x).detach().cpu().numpy().reshape(-1)
    prediction_iterations = recorded["prediction_iterations"]
    predictions = recorded["predictions"]
    assert isinstance(prediction_iterations, list)
    assert isinstance(predictions, list)
    prediction_iterations.append(protocol.iterations)
    predictions.append(final_prediction.copy())

    result = SpectralBiasRepeatResult(
        iterations=np.asarray(recorded["iterations"], dtype=np.int64),
        train_loss=np.asarray(recorded["train_loss"], dtype=np.float64),
        normalized_amplitudes=np.asarray(recorded["normalized_amplitudes"], dtype=np.float64),
        author_layer_norms=np.asarray(recorded["author_layer_norms"], dtype=np.float64),
        singular_layer_norms=np.asarray(recorded["singular_layer_norms"], dtype=np.float64),
        prediction_iterations=np.asarray(prediction_iterations, dtype=np.int64),
        predictions=np.asarray(predictions, dtype=np.float32),
        phase_fractions=phases,
        final_prediction=final_prediction.astype(np.float32),
    )
    return result, model
