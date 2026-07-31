from __future__ import annotations

import unittest

import numpy as np
import torch
import torch.nn.functional as F

from comfy_research.engine.datasets.information_bottleneck_dataset import (
    build_information_bottleneck_arrays,
)
from comfy_research.engine.datasets.sampling_orders import affine_epoch_positions
from comfy_research.engine.optimizers.idnns_init import apply_idnns_init
from comfy_research.engine.reproductions.information_bottleneck import (
    BatchedSeedMlp,
    InformationBottleneckProtocol,
    idnns_epoch_snapshots,
    saxe_epoch_snapshots,
    train_information_bottleneck_ensemble,
)

class InformationBottleneckReproductionTests(unittest.TestCase):
    def test_affine_epoch_positions_are_permutations_and_replay_exactly(self) -> None:
        first_rng = np.random.default_rng(1_001_711)
        replay_rng = np.random.default_rng(1_001_711)
        first = [
            affine_epoch_positions(first_rng, count=17)[0]
            for _ in range(4)
        ]
        replay = [
            affine_epoch_positions(replay_rng, count=17)[0]
            for _ in range(4)
        ]

        for actual, expected in zip(first, replay):
            self.assertTrue(np.array_equal(actual, expected))
            self.assertEqual(sorted(actual.tolist()), list(range(17)))

    def test_paper_snapshot_schedule_matches_released_idnns_run_name(self) -> None:
        epochs = idnns_epoch_snapshots(10_000, samples=1800)

        self.assertEqual(len(epochs), 964)
        self.assertEqual(int(epochs[0]), 0)
        self.assertEqual(int(epochs[-1]), 9998)
        self.assertTrue(np.all(np.diff(epochs) > 0))

    def test_saxe_snapshot_schedule_matches_notebook_callback(self) -> None:
        epochs = saxe_epoch_snapshots(10_000)

        self.assertIn(19, epochs)
        self.assertIn(20, epochs)
        self.assertNotIn(21, epochs)
        self.assertIn(1980, epochs)
        self.assertNotIn(1990, epochs)
        self.assertIn(9900, epochs)

    def test_batched_model_keeps_repeats_independent(self) -> None:
        model = BatchedSeedMlp(
            repeats=3,
            input_dim=12,
            hidden_dims=(10, 8),
            output_mode="binary_sigmoid",
            activation="tanh",
            initializer="idnns_fan_in",
            seed=5,
        )
        x = torch.zeros(7, 12)
        logits, hidden = model(x)

        self.assertEqual(tuple(logits.shape), (3, 7, 1))
        self.assertEqual([tuple(value.shape) for value in hidden], [(3, 7, 10), (3, 7, 8)])
        self.assertFalse(torch.equal(model.weights[0][0], model.weights[0][1]))

    @staticmethod
    def _graph_model(seed: int) -> tuple[torch.nn.Sequential, list[torch.nn.Linear]]:
        widths = (12, 10, 8, 6, 4, 2, 1)
        graph_layers: list[torch.nn.Linear] = []
        modules: list[torch.nn.Module] = []
        for index, (in_dim, out_dim) in enumerate(zip(widths[:-1], widths[1:])):
            layer = torch.nn.Linear(in_dim, out_dim)
            graph_layers.append(layer)
            modules.append(layer)
            if index < len(widths) - 2:
                modules.append(torch.nn.Tanh())
        model = torch.nn.Sequential(*modules)
        apply_idnns_init(model, seed=seed)
        return model, graph_layers

    def test_graph_idnns_initialization_matches_reference_repeat_exactly(self) -> None:
        seed = 1708
        graph_model, graph_layers = self._graph_model(seed)
        self.assertIsInstance(graph_model, torch.nn.Sequential)
        reference = BatchedSeedMlp(
            repeats=1,
            input_dim=12,
            hidden_dims=(10, 8, 6, 4, 2),
            output_mode="binary_sigmoid",
            activation="tanh",
            initializer="idnns_fan_in",
            seed=seed,
        )

        for graph_layer, reference_weight, reference_bias in zip(
            graph_layers,
            reference.weights,
            reference.biases,
        ):
            self.assertTrue(torch.equal(graph_layer.weight, reference_weight[0].T))
            self.assertTrue(torch.equal(graph_layer.bias, reference_bias[0, 0]))

    def test_single_logit_full_batch_updates_match_reference_trajectory(self) -> None:
        seed = 1708
        x_np, y_np, _x_test, _y_test = build_information_bottleneck_arrays(
            np.random.default_rng(seed),
            train_size=205,
            test_size=0,
        )
        x = torch.from_numpy(x_np)
        y = torch.from_numpy(y_np)
        graph_model, graph_layers = self._graph_model(seed)
        graph_optimizer = torch.optim.Adam(graph_model.parameters(), lr=4e-4)
        reference = BatchedSeedMlp(
            repeats=1,
            input_dim=12,
            hidden_dims=(10, 8, 6, 4, 2),
            output_mode="binary_sigmoid",
            activation="tanh",
            initializer="idnns_fan_in",
            seed=seed,
        )
        reference_optimizer = torch.optim.Adam(reference.parameters(), lr=4e-4)

        for _step in range(3):
            graph_optimizer.zero_grad(set_to_none=True)
            graph_loss = F.binary_cross_entropy_with_logits(
                graph_model(x).reshape(-1),
                y.float(),
            )
            graph_loss.backward()
            graph_optimizer.step()

            reference_optimizer.zero_grad(set_to_none=True)
            reference_logits, _hidden = reference(x)
            reference_loss = F.binary_cross_entropy_with_logits(
                reference_logits[0, :, 0],
                y.float(),
            )
            reference_loss.backward()
            reference_optimizer.step()

            self.assertAlmostEqual(
                float(graph_loss.detach()),
                float(reference_loss.detach()),
                places=6,
            )
            for graph_layer, reference_weight, reference_bias in zip(
                graph_layers,
                reference.weights,
                reference.biases,
            ):
                self.assertTrue(
                    torch.allclose(
                        graph_layer.weight,
                        reference_weight[0].T,
                        atol=1e-7,
                        rtol=1e-6,
                    )
                )
                self.assertTrue(
                    torch.allclose(
                        graph_layer.bias,
                        reference_bias[0, 0],
                        atol=1e-7,
                        rtol=1e-6,
                    )
                )

    def test_affine_minibatch_updates_match_single_reference_repeat(self) -> None:
        seed = 1748
        train_size = 1843
        batch_size = 256
        epochs = 2
        x_np, y_np, _x_test, _y_test = build_information_bottleneck_arrays(
            np.random.default_rng(seed),
            train_size=train_size,
            test_size=0,
        )
        x = torch.from_numpy(x_np)
        y = torch.from_numpy(y_np)
        graph_model, graph_layers = self._graph_model(seed)
        graph_optimizer = torch.optim.Adam(graph_model.parameters(), lr=4e-4)
        shuffle_rng = np.random.default_rng(seed + 1_000_003)

        for _epoch in range(epochs):
            order = affine_epoch_positions(shuffle_rng, count=train_size)[0]
            for start in range(0, train_size, batch_size):
                idx = torch.from_numpy(order[start : start + batch_size])
                graph_optimizer.zero_grad(set_to_none=True)
                loss = F.binary_cross_entropy_with_logits(
                    graph_model(x.index_select(0, idx)).reshape(-1),
                    y.index_select(0, idx).float(),
                )
                loss.backward()
                graph_optimizer.step()

        protocol = InformationBottleneckProtocol(
            train_percent=45,
            repeats=1,
            epochs=epochs,
            batch_size=batch_size,
            snapshot_samples=1,
            seed=seed,
        )
        _result, reference, subsets = train_information_bottleneck_ensemble(
            protocol,
            device="cpu",
        )
        self.assertEqual(subsets.shape, (1, train_size))
        for graph_layer, reference_weight, reference_bias in zip(
            graph_layers,
            reference.weights,
            reference.biases,
        ):
            self.assertTrue(
                torch.allclose(
                    graph_layer.weight,
                    reference_weight[0].T,
                    atol=1e-7,
                    rtol=1e-6,
                )
            )
            self.assertTrue(
                torch.allclose(
                    graph_layer.bias,
                    reference_bias[0, 0],
                    atol=1e-7,
                    rtol=1e-6,
                )
            )

    def test_cpu_smoke_records_per_seed_information(self) -> None:
        protocol = InformationBottleneckProtocol(
            train_percent=5,
            repeats=2,
            epochs=2,
            batch_size=205,
            snapshot_samples=2,
            seed=19,
        )
        result, _model, subsets = train_information_bottleneck_ensemble(protocol, device="cpu")

        self.assertEqual(result.train_loss.shape, (2, 2))
        self.assertEqual(result.information_x.shape, (2, 2, 6))
        self.assertEqual(result.information_y.shape, (2, 2, 6))
        self.assertEqual(subsets.shape, (2, 205))
        self.assertTrue(np.isfinite(result.information_x).all())
        self.assertTrue(np.isfinite(result.information_y).all())

    def test_saxe_profile_can_use_published_callback_schedule(self) -> None:
        protocol = InformationBottleneckProtocol(
            train_percent=80,
            repeats=1,
            epochs=3,
            batch_size=256,
            hidden_dims=(10, 7, 5, 4, 3),
            output_mode="two_softmax",
            activation="relu",
            initializer="saxe_fan_out",
            binning="saxe_fixed_width_0_07",
            snapshot_schedule="saxe_callback",
            seed=23,
        )
        result, _model, subsets = train_information_bottleneck_ensemble(protocol, device="cpu")

        self.assertEqual(result.epochs.tolist(), [0, 1, 2])
        self.assertEqual(result.information_x.shape, (3, 1, 6))
        self.assertEqual(subsets.shape, (1, 3277))


if __name__ == "__main__":
    unittest.main()
