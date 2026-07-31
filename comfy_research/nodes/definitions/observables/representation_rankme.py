"""RankMe of a selected hidden representation's centered covariance."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from comfy_research.nodes.registry import observable_def, recorder_for
from comfy_research.nodes.schema import BoolField, EnumField, FrontendSpec, ObservableDef, SpawnSpec, VizSpec

if TYPE_CHECKING:  # pragma: no cover
    from comfy_research.engine.trainer.recorder import ObservableRecorder
    from comfy_research.schemas.graph import Node


REPRESENTATION_RANKME = observable_def(
    ObservableDef(
        type="observable_representation_rankme",
        label="Representation RankMe",
        hint="Entropy effective rank of a selected representation's centered feature covariance.",
        viz=VizSpec(
            variant="user",
            title="Representation RankMe",
            info_markdown=(
                "**Representation RankMe** — flatten the selected activation to samples × features, "
                "center each feature, form `Σ = FᵀF / M`, and report the exponential entropy of "
                "its normalized eigenvalues. The two largest covariance eigenvalues are also logged."
            ),
            spawns=True,
            user_whitelisted=True,
            spawn=SpawnSpec(kind="user_scalar", unit="rank"),
        ),
        fields=(
            EnumField(
                key="representationId",
                label="Representation ID",
                default="0::output",
            ),
            BoolField(
                key="captureTrajectories",
                label="Capture Features + Head",
                default=False,
            ),
            BoolField(
                key="tokenPositionsAsSamples",
                label="Token Positions As Samples",
                default=False,
            ),
        ),
        frontend=FrontendSpec(codegen_key="observable_representation_rankme"),
    )
)


def _rankme_and_eigenvalues(
    representation: Any,
    *,
    token_positions_as_samples: bool = False,
) -> tuple[float, list[float]]:
    """Return RankMe and descending covariance eigenvalues for an activation."""
    import math

    import torch

    if not isinstance(representation, torch.Tensor) or representation.ndim < 1:
        return float("nan"), []
    sample_count = int(representation.shape[0])
    if sample_count <= 0 or representation.numel() == 0:
        return float("nan"), []

    try:
        values = representation.detach().cpu().to(dtype=torch.float64)
        if token_positions_as_samples and values.ndim >= 3:
            features = values.reshape(-1, values.shape[-1])
        else:
            features = values.reshape(sample_count, -1)
        features = features - features.mean(dim=0, keepdim=True)
        covariance = features.T @ features / float(sample_count)
        eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0).flip(0)
        if not bool(torch.isfinite(eigenvalues).all()):
            return float("nan"), []
        total = eigenvalues.sum()
        if not math.isfinite(float(total.item())) or float(total.item()) <= 0.0:
            return float("nan"), [float(v.item()) for v in eigenvalues]
        probabilities = eigenvalues / total
        positive = probabilities > 0
        entropy = -(probabilities[positive] * probabilities[positive].log()).sum()
        rankme = float(torch.exp(entropy).item())
        return rankme, [float(v.item()) for v in eigenvalues]
    except (RuntimeError, TypeError, ValueError):
        return float("nan"), []


@recorder_for(REPRESENTATION_RANKME)
def record(rec: "ObservableRecorder", on: "Node") -> None:
    """Log centered-covariance RankMe and the two leading eigenvalues."""
    import torch.nn as nn

    from comfy_research.engine.trainer.scalar import _scalar_bool, _scalar_str

    histories = rec.observable_metric_histories
    data: dict[str, Any] = on.data or {}
    representation_id = _scalar_str(data.get("representationId"), "0::output").strip()
    representation = rec._representation_tensors_for_log().get(representation_id)
    rankme, eigenvalues = _rankme_and_eigenvalues(
        representation,
        token_positions_as_samples=_scalar_bool(data.get("tokenPositionsAsSamples"), False),
    )

    if _scalar_bool(data.get("captureTrajectories"), False):
        features = (
            []
            if representation is None
            else representation.detach().cpu().reshape(int(representation.shape[0]), -1).tolist()
        )
        last_linear: nn.Linear | None = None
        for module in rec.model.modules():
            if isinstance(module, nn.Linear):
                last_linear = module
        # Store W in the paper convention (feature_dim x class_count), whereas
        # nn.Linear.weight is class_count x feature_dim.
        weights = [] if last_linear is None else last_linear.weight.detach().cpu().T.tolist()
        rec.observable_embedding_histories.setdefault(f"{on.id}::features", []).append(features)
        rec.observable_embedding_histories.setdefault(f"{on.id}::weights", []).append(weights)

    histories[on.id].append(rankme)
    history_length = len(histories[on.id])
    for index in range(2):
        value = eigenvalues[index] if index < len(eigenvalues) else float("nan")
        key = f"{on.id}::eig::{index}"
        row = histories.setdefault(key, [])
        while len(row) < history_length - 1:
            row.append(float("nan"))
        row.append(float(value))
