"""Per-Linear-layer spectral-norm observable."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from comfy_research.nodes.registry import observable_def, recorder_for
from comfy_research.nodes.schema import EnumField, FrontendSpec, IntField, ObservableDef, SpawnSpec, VizSpec

if TYPE_CHECKING:  # pragma: no cover
    from comfy_research.engine.trainer.recorder import ObservableRecorder
    from comfy_research.schemas.graph import Node


LAYER_SPECTRAL_NORM = observable_def(
    ObservableDef(
        type="observable_layer_spectral_norm",
        label="Layer spectral norm",
        hint="Logs one norm curve per Linear layer using a deterministic, RNG-neutral measurement.",
        viz=VizSpec(
            variant="layer_spectral_norm",
            title="Layer spectral norm",
            info_markdown=(
                "**Layer spectral norm** records every `Linear.weight` matrix without changing training RNG. "
                "`singular_value` is conventional power iteration on `W.T @ W`; `author_figure1` preserves "
                "the released Rahaman Figure 1 notebook estimator. Vector-shaped first/last layers use "
                "their exact vector norm; optional seeded Gaussian starts replay the notebook without "
                "touching training RNG."
            ),
            spawns=True,
            spawn=SpawnSpec(kind="hessian_topk", fixed_top_k=6),
        ),
        fields=(
            EnumField(key="estimator", label="Estimator", default="singular_value", options=("singular_value", "author_figure1")),
            IntField(key="powerIterations", label="Power iterations", default=10, min=1),
            EnumField(key="startVector", label="Start vector", default="deterministic", options=("deterministic", "seeded_gaussian")),
            IntField(key="seed", label="Seed", default=0, min=0),
        ),
        frontend=FrontendSpec(component_key="LayerSpectralNormObservableNode"),
    )
)


@recorder_for(LAYER_SPECTRAL_NORM)
def record(rec: "ObservableRecorder", on: "Node") -> None:
    """Record one aligned series per Linear layer; failures become NaN, never training errors."""
    import math
    import torch
    import torch.nn as nn

    from comfy_research.engine.trainer.scalar import _scalar_int
    from comfy_research.engine.trainer.spectral_norm import author_figure1_power_estimate, singular_norm_power_estimate

    histories = rec.observable_metric_histories
    data: dict[str, Any] = on.data or {}
    estimator = str(data.get("estimator") or "singular_value").strip()
    iterations = max(1, _scalar_int(data.get("powerIterations"), 10))
    start_vector = str(data.get("startVector") or "deterministic").strip()
    generator = None
    if estimator == "author_figure1" and start_vector == "seeded_gaussian":
        generator = rec.observable_rng_generators.get(on.id)
        if generator is None:
            linear_layers = [
                layer for layer in rec.model.modules() if isinstance(layer, nn.Linear)
            ]
            if linear_layers:
                first_weight = linear_layers[0].weight
                generator = torch.Generator(device=first_weight.device)
                generator.manual_seed(_scalar_int(data.get("seed"), 0))
                prior_ticks = len(histories.get(on.id, []))
                square_widths = [
                    int(layer.weight.shape[1])
                    for layer in linear_layers
                    if int(layer.weight.shape[0]) == int(layer.weight.shape[1])
                ]
                for _ in range(prior_ticks):
                    for width in square_widths:
                        torch.randn(
                            (width, 1),
                            dtype=first_weight.dtype,
                            device=first_weight.device,
                            generator=generator,
                        )
                rec.observable_rng_generators[on.id] = generator
    values: list[float] = []
    try:
        with torch.no_grad():
            for layer in (m for m in rec.model.modules() if isinstance(m, nn.Linear)):
                try:
                    if estimator == "author_figure1":
                        value = author_figure1_power_estimate(
                            layer.weight,
                            iterations=iterations,
                            generator=generator,
                        )
                    else:
                        value = singular_norm_power_estimate(layer.weight, iterations=iterations)
                    values.append(float(value.item()))
                except Exception:
                    values.append(float("nan"))
    except Exception:
        values = []
    primary = values[0] if values else float("nan")
    histories[on.id].append(primary)
    length = len(histories[on.id])
    for index, value in enumerate(values, 1):
        row = histories.setdefault(f"{on.id}::layer::{index}", [])
        while len(row) < length - 1:
            row.append(float("nan"))
        row.append(value if math.isfinite(value) else float("nan"))
