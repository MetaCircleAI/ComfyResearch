"""Declared supplemental Information Bottleneck control matrix."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class InformationBottleneckControl:
    control_id: str
    comparison_group: str
    activation: str
    bins: int
    binning: str
    optimizer: str
    batch_size: int
    learning_rate: float
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


SUPPLEMENTAL_CONTROLS = (
    InformationBottleneckControl(
        control_id="activation_tanh_fixedwidth_adam_minibatch",
        comparison_group="activation",
        activation="tanh",
        bins=30,
        binning="saxe_fixed_width_0_07",
        optimizer="adam",
        batch_size=256,
        learning_rate=4e-4,
        rationale="Released Saxe notebook-style tanh/fixed-width/Adam reference.",
    ),
    InformationBottleneckControl(
        control_id="activation_relu_fixedwidth_adam_minibatch",
        comparison_group="activation",
        activation="relu",
        bins=30,
        binning="saxe_fixed_width_0_07",
        optimizer="adam",
        batch_size=256,
        learning_rate=4e-4,
        rationale="Changes only the nonlinearity to test saturation dependence.",
    ),
    InformationBottleneckControl(
        control_id="binning_tanh_adaptive30_adam_minibatch",
        comparison_group="binning_resolution",
        activation="tanh",
        bins=30,
        binning="adaptive_minmax",
        optimizer="adam",
        batch_size=256,
        learning_rate=4e-4,
        rationale="Thirty adaptive bins over the observed activation range.",
    ),
    InformationBottleneckControl(
        control_id="binning_tanh_adaptive100_adam_minibatch",
        comparison_group="binning_resolution",
        activation="tanh",
        bins=100,
        binning="adaptive_minmax",
        optimizer="adam",
        batch_size=256,
        learning_rate=4e-4,
        rationale="Changes only bin resolution from 30 to 100.",
    ),
    InformationBottleneckControl(
        control_id="noise_tanh_adaptive30_sgd_minibatch",
        comparison_group="optimizer_noise",
        activation="tanh",
        bins=30,
        binning="adaptive_minmax",
        optimizer="sgd",
        batch_size=256,
        learning_rate=1e-2,
        rationale="Plain SGD with mini-batch noise; learning rate is a recorded control choice.",
    ),
    InformationBottleneckControl(
        control_id="noise_tanh_adaptive30_sgd_fullbatch",
        comparison_group="optimizer_noise",
        activation="tanh",
        bins=30,
        binning="adaptive_minmax",
        optimizer="sgd",
        batch_size=3277,
        learning_rate=1e-2,
        rationale="Same SGD setting with the complete 80% training subset per update.",
    ),
)


def supplemental_control_map() -> dict[str, InformationBottleneckControl]:
    controls = {control.control_id: control for control in SUPPLEMENTAL_CONTROLS}
    if len(controls) != len(SUPPLEMENTAL_CONTROLS):
        raise RuntimeError("supplemental Information Bottleneck control ids must be unique")
    return controls
