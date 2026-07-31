from __future__ import annotations

import unittest

from comfy_research.engine.runs.train_coordinate_descent import CurveSimilarityConfig, _prepare_curve, _score_curve


class CoordinateDescentCurveMathTests(unittest.TestCase):
    def test_prepare_curve_sorts_and_dedupes(self) -> None:
        xs, ys = _prepare_curve([2, 0, 1, 1], [0.2, 0.9, 0.5, 0.4])
        self.assertEqual(xs, [0.0, 1.0, 2.0])
        self.assertEqual(ys, [0.9, 0.4, 0.2])

    def test_score_prefers_matching_curve(self) -> None:
        cfg = CurveSimilarityConfig(end_weight=1.0, smoothness_weight=0.0)
        target_steps = [0.0, 5.0, 10.0]
        target_vals = [1.0, 0.6, 0.2]
        near_score, _, _ = _score_curve(
            source_steps=[0, 5, 10],
            source_values=[1.0, 0.58, 0.19],
            target_steps=target_steps,
            target_values=target_vals,
            cfg=cfg,
        )
        far_score, _, _ = _score_curve(
            source_steps=[0, 5, 10],
            source_values=[1.0, 0.9, 0.85],
            target_steps=target_steps,
            target_values=target_vals,
            cfg=cfg,
        )
        self.assertLess(near_score, far_score)


if __name__ == "__main__":
    unittest.main()
