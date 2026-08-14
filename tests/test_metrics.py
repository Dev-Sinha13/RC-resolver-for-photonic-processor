import unittest

import numpy as np

from rc_photonics.metrics import (
    mean_squared_error,
    normalized_mean_squared_error,
)


class MetricTests(unittest.TestCase):
    def test_mse_matches_hand_calculation(self) -> None:
        target = np.array([1.0, 2.0, 3.0])
        prediction = np.array([1.0, 1.0, 2.0])

        self.assertAlmostEqual(
            mean_squared_error(target, prediction),
            2.0 / 3.0,
        )

    def test_perfect_prediction_has_zero_error(self) -> None:
        target = np.array([-2.0, 0.0, 3.0, 5.0])

        self.assertEqual(mean_squared_error(target, target), 0.0)
        self.assertEqual(normalized_mean_squared_error(target, target), 0.0)

    def test_mean_predictor_has_nmse_one(self) -> None:
        target = np.array([1.0, 2.0, 4.0, 8.0])
        prediction = np.full_like(target, np.mean(target))

        self.assertAlmostEqual(
            normalized_mean_squared_error(target, prediction),
            1.0,
        )

    def test_nmse_matches_hand_calculation(self) -> None:
        target = np.array([1.0, 2.0, 3.0])
        prediction = np.array([1.0, 1.0, 2.0])

        self.assertAlmostEqual(
            normalized_mean_squared_error(target, prediction),
            1.0,
        )

    def test_metrics_accept_array_like_inputs(self) -> None:
        self.assertAlmostEqual(
            mean_squared_error([1.0, 2.0], [1.0, 1.0]),
            0.5,
        )

    def test_metrics_reject_invalid_pairs(self) -> None:
        invalid_pairs = [
            ([], []),
            ([1.0, 2.0], [1.0]),
            ([[1.0, 2.0]], [[1.0, 2.0]]),
            ([1.0, np.nan], [1.0, 2.0]),
            ([1.0, 2.0], [1.0, np.inf]),
        ]

        for target, prediction in invalid_pairs:
            with self.subTest(target=target, prediction=prediction):
                with self.assertRaises(ValueError):
                    mean_squared_error(target, prediction)
                with self.assertRaises(ValueError):
                    normalized_mean_squared_error(target, prediction)

    def test_nmse_rejects_constant_target(self) -> None:
        with self.assertRaisesRegex(ValueError, "variance"):
            normalized_mean_squared_error(
                [4.0, 4.0, 4.0],
                [4.0, 4.0, 4.0],
            )


if __name__ == "__main__":
    unittest.main()
