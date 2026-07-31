"""Unit tests for cyclic LR / batch schedules."""

from __future__ import annotations

from comfy_research.engine.optimizers.cyclic_schedules import (
    cbs_epochs_to_training_steps,
    cyclic_batch_for_data_epoch,
    cyclic_batch_for_step,
    cyclic_lr_for_data_epoch,
    cyclic_lr_for_step,
    data_epoch_state_for_cyclic_batch,
    epoch_index_for_step,
    epochs_to_cycle_steps,
    steps_per_epoch,
    triangular_cycle_fraction,
)


def test_triangular_cycle_fraction_endpoints() -> None:
    cl = 10
    assert triangular_cycle_fraction(0, cl) == 0.0
    assert abs(triangular_cycle_fraction(5, cl) - 1.0) < 1e-9
    assert abs(triangular_cycle_fraction(10, cl)) < 1e-9


def test_square_epoch_matches_successful_jastrzbski_reproduction() -> None:
    batches = [
        cyclic_batch_for_data_epoch(
            epoch,
            batch_min=128,
            batch_max=640,
            cycle_length_epochs=10,
            mode="square_epoch",
        )
        for epoch in range(10)
    ]
    lrs = [
        cyclic_lr_for_data_epoch(
            epoch,
            lr_min=0.001,
            lr_max=0.005,
            cycle_length_epochs=10,
            mode="square_epoch",
        )
        for epoch in range(10)
    ]
    assert batches == [128] * 5 + [640] * 5
    assert lrs == [0.005] * 5 + [0.001] * 5

    assert cbs_epochs_to_training_steps(
        300,
        train_size=45000,
        batch_min=128,
        batch_max=640,
        cycle_length_epochs=10,
        mode="square_epoch",
    ) == 63450


def test_discrete_epoch_lr_jastr_fig1() -> None:
    spe = steps_per_epoch(50000, 128)
    assert spe == 391
    # CLR inverted vs CBS: ep0 → η_max so η/S matches CBS at B_min
    assert cyclic_lr_for_step(
        0,
        lr_min=0.001,
        lr_max=0.005,
        mode="discrete_epoch",
        cycle_length_epochs=10,
        cycle_length_steps=3910,
        steps_per_epoch=spe,
    ) == 0.005
    # epoch 5 → cycle trough (η_min), matches CBS B_max
    step_epoch5 = 5 * spe
    mid = cyclic_lr_for_step(
        step_epoch5,
        lr_min=0.001,
        lr_max=0.005,
        mode="discrete_epoch",
        cycle_length_epochs=10,
        cycle_length_steps=3910,
        steps_per_epoch=spe,
    )
    assert abs(mid - 0.001) < 1e-9
    # same epoch, any step → same LR
    same = cyclic_lr_for_step(
        step_epoch5 + 200,
        lr_min=0.001,
        lr_max=0.005,
        mode="discrete_epoch",
        cycle_length_epochs=10,
        cycle_length_steps=3910,
        steps_per_epoch=spe,
    )
    assert abs(same - mid) < 1e-9


def test_discrete_epoch_batch_constant_within_epoch() -> None:
    spe = 391
    epoch = 3
    b0 = cyclic_batch_for_step(
        epoch * spe,
        batch_min=128,
        batch_max=640,
        mode="discrete_epoch",
        cycle_length_epochs=10,
        cycle_length_steps=3910,
        steps_per_epoch=spe,
    )
    b1 = cyclic_batch_for_step(
        epoch * spe + 200,
        batch_min=128,
        batch_max=640,
        mode="discrete_epoch",
        cycle_length_epochs=10,
        cycle_length_steps=3910,
        steps_per_epoch=spe,
    )
    assert b0 == b1
    assert epoch_index_for_step(epoch * spe + 200, steps_per_epoch=spe) == epoch


def test_triangular_step_mode_differs_within_epoch() -> None:
    spe = 391
    cl = epochs_to_cycle_steps(10, 50000, 128)
    b_start = cyclic_batch_for_step(
        0,
        batch_min=128,
        batch_max=640,
        mode="triangular_step",
        cycle_length_epochs=10,
        cycle_length_steps=cl,
        steps_per_epoch=spe,
    )
    b_mid = cyclic_batch_for_step(
        200,
        batch_min=128,
        batch_max=640,
        mode="triangular_step",
        cycle_length_epochs=10,
        cycle_length_steps=cl,
        steps_per_epoch=spe,
    )
    assert b_start == 128
    assert b_mid != b_start


def test_cbs_data_epoch_variable_length() -> None:
    train_size = 50000
    b0 = cyclic_batch_for_data_epoch(0, batch_min=128, batch_max=640, cycle_length_epochs=10)
    assert b0 == 128
    assert steps_per_epoch(train_size, b0) == 391
    ep, steps_in, batch = data_epoch_state_for_cyclic_batch(
        391,
        train_size=train_size,
        batch_min=128,
        batch_max=640,
        cycle_length_epochs=10,
    )
    assert ep == 1
    assert steps_in == 0
    assert batch == cyclic_batch_for_data_epoch(1, batch_min=128, batch_max=640, cycle_length_epochs=10)


def test_cbs_data_epoch_batch_constant_within_epoch() -> None:
    train_size = 50000
    for step in range(200):
        _, _, batch = data_epoch_state_for_cyclic_batch(
            step,
            train_size=train_size,
            batch_min=128,
            batch_max=640,
            cycle_length_epochs=10,
        )
        assert batch == cyclic_batch_for_data_epoch(0, batch_min=128, batch_max=640, cycle_length_epochs=10)


def test_clr_data_epoch_lr() -> None:
    assert cyclic_lr_for_data_epoch(0, lr_min=0.001, lr_max=0.005, cycle_length_epochs=10) == 0.005
    trough = cyclic_lr_for_data_epoch(5, lr_min=0.001, lr_max=0.005, cycle_length_epochs=10)
    assert abs(trough - 0.001) < 1e-9


def test_cbs_clr_eta_over_s_in_phase() -> None:
    """Fig 1 left: CBS and CLR share η/S phase (high together, low together)."""
    for ep in (0, 5, 10):
        b = cyclic_batch_for_data_epoch(ep, batch_min=128, batch_max=640, cycle_length_epochs=10)
        lr = cyclic_lr_for_data_epoch(ep, lr_min=0.001, lr_max=0.005, cycle_length_epochs=10)
        assert abs((0.005 / b) - (lr / 128)) < 1e-9
    # Mid-cycle: both move toward low η/S together (not exact match — B is discrete).
    b1 = cyclic_batch_for_data_epoch(1, batch_min=128, batch_max=640, cycle_length_epochs=10)
    lr1 = cyclic_lr_for_data_epoch(1, lr_min=0.001, lr_max=0.005, cycle_length_epochs=10)
    assert b1 > 128 and lr1 < 0.005
    assert (0.005 / b1) < (0.005 / 128) and (lr1 / 128) < (0.005 / 128)


def test_cbs_epochs_to_training_steps_fig1() -> None:
    steps = cbs_epochs_to_training_steps(
        300,
        train_size=50000,
        batch_min=128,
        batch_max=640,
        cycle_length_epochs=10,
    )
    assert steps == 48720
    assert steps < 300 * 391
