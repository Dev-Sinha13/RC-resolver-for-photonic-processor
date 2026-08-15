import unittest

import numpy as np

from rc_photonics.autoregression import (
    AutoregressiveRidge,
    fit_current_sample_ridge,
    fit_autoregressive_ridge,
)


class AutoregressiveRidgeTests(unittest.TestCase):
    def test_current_sample_regression_recovers_affine_mapping(self) -> None:
        observations = np.linspace(-2.0, 2.0, 100)
        targets = 3.0 * observations - 0.25
        model = fit_current_sample_ridge(
            observations,
            targets,
            regularization=0.0,
        )

        self.assertAlmostEqual(model.coefficient, 3.0)
        self.assertAlmostEqual(model.intercept, -0.25)
        np.testing.assert_allclose(model.predict(observations), targets)

    def test_fit_recovers_known_one_lag_mapping(self) -> None:
        rng = np.random.default_rng(12)
        observations = rng.normal(size=200)
        targets = np.empty_like(observations)
        targets[0] = 0.0
        targets[1:] = 2.0 * observations[:-1] + 0.5

        model = fit_autoregressive_ridge(
            observations,
            targets,
            n_lags=1,
            regularization=0.0,
        )

        np.testing.assert_allclose(model.coefficients, [2.0], atol=1e-12)
        self.assertAlmostEqual(model.intercept, 0.5, places=12)
        np.testing.assert_allclose(model.predict(observations), targets[1:])

    def test_recursive_restoration_forecasts_through_gap(self) -> None:
        model = AutoregressiveRidge(
            coefficients=np.array([1.0]),
            intercept=1.0,
            n_lags=1,
            regularization=0.0,
        )

        restored = model.restore_missing(
            [1.0, 2.0, 0.0, 0.0, 5.0],
            [True, True, False, False, True],
        )

        np.testing.assert_array_equal(restored, [1.0, 2.0, 3.0, 4.0, 5.0])

    def test_predictions_do_not_depend_on_future_inputs(self) -> None:
        training = np.sin(np.linspace(0.0, 8.0, 200))
        model = fit_autoregressive_ridge(
            training,
            n_lags=2,
            regularization=1e-3,
        )
        original = training[:20].copy()
        changed = original.copy()
        changed[10:] += 1_000.0

        original_predictions = model.predict(original)
        changed_predictions = model.predict(changed)

        np.testing.assert_allclose(
            original_predictions[:9],
            changed_predictions[:9],
        )

    def test_invalid_training_configuration_is_rejected(self) -> None:
        invalid_arguments = (
            {"n_lags": 0, "regularization": 0.1},
            {"n_lags": 10, "regularization": 0.1},
            {"n_lags": 1, "regularization": -1.0},
        )

        for arguments in invalid_arguments:
            with self.subTest(arguments=arguments):
                with self.assertRaises(ValueError):
                    fit_autoregressive_ridge([1.0, 2.0], **arguments)


if __name__ == "__main__":
    unittest.main()
