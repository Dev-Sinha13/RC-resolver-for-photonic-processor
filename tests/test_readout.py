import unittest

import numpy as np

from rc_photonics.readout import RidgeReadout, fit_ridge_readout


class RidgeReadoutTests(unittest.TestCase):
    def test_exact_affine_mapping_is_recovered(self) -> None:
        states = np.array(
            [
                [-2.0, 1.0],
                [-1.0, -1.0],
                [0.0, 2.0],
                [1.0, 0.0],
                [2.0, -2.0],
            ]
        )
        targets = 1.5 * states[:, 0] - 0.75 * states[:, 1] + 0.25

        model = fit_ridge_readout(states, targets, regularization=0.0)

        np.testing.assert_allclose(model.weights, [1.5, -0.75], atol=1e-12)
        self.assertAlmostEqual(model.intercept, 0.25, places=12)
        np.testing.assert_allclose(model.predict(states), targets, atol=1e-12)

    def test_regularization_shrinks_weights(self) -> None:
        rng = np.random.default_rng(42)
        states = rng.normal(size=(200, 5))
        targets = states @ np.array([2.0, -1.0, 0.5, 3.0, -2.0])

        unregularized = fit_ridge_readout(
            states,
            targets,
            regularization=0.0,
        )
        regularized = fit_ridge_readout(
            states,
            targets,
            regularization=1_000.0,
        )

        self.assertLess(
            np.linalg.norm(regularized.weights),
            np.linalg.norm(unregularized.weights),
        )

    def test_prediction_shape_and_type(self) -> None:
        model = RidgeReadout(
            weights=np.array([2.0, -1.0]),
            intercept=0.5,
            regularization=0.1,
        )
        prediction = model.predict([[1.0, 2.0], [3.0, 4.0]])

        self.assertEqual(prediction.shape, (2,))
        self.assertEqual(prediction.dtype, np.float64)
        np.testing.assert_allclose(prediction, [0.5, 2.5])

    def test_invalid_fit_inputs_are_rejected(self) -> None:
        invalid_arguments = (
            ([1.0, 2.0], [1.0, 2.0], 0.1),
            ([[1.0], [2.0]], [1.0], 0.1),
            ([[1.0], [np.nan]], [1.0, 2.0], 0.1),
            ([[1.0], [2.0]], [1.0, np.inf], 0.1),
            ([[1.0], [2.0]], [1.0, 2.0], -1.0),
        )

        for states, targets, penalty in invalid_arguments:
            with self.subTest(states=states, targets=targets, penalty=penalty):
                with self.assertRaises(ValueError):
                    fit_ridge_readout(
                        states,
                        targets,
                        regularization=penalty,
                    )

    def test_predict_rejects_wrong_state_count(self) -> None:
        model = RidgeReadout(
            weights=np.array([1.0, 2.0]),
            intercept=0.0,
            regularization=0.0,
        )

        with self.assertRaisesRegex(ValueError, "state count"):
            model.predict([[1.0, 2.0, 3.0]])


if __name__ == "__main__":
    unittest.main()
