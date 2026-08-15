import unittest

from rc_photonics.experiments import (
    run_gaussian_baseline_experiment,
    run_missing_gap_experiment,
)


class GaussianBaselineExperimentTests(unittest.TestCase):
    def test_experiment_is_deterministic_and_returns_finite_scores(self) -> None:
        arguments = {
            "n_samples": 1_000,
            "noise_levels": (0.05, 0.1),
            "moving_average_windows": (1, 3, 5),
            "current_regularizations": (0.0, 0.1),
            "autoregressive_lags": (2, 5),
            "autoregressive_regularizations": (1e-3,),
            "seed": 7,
        }

        first = run_gaussian_baseline_experiment(**arguments)
        second = run_gaussian_baseline_experiment(**arguments)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 2)
        for result in first:
            self.assertIn(result.selected_window_size, (1, 3, 5))
            self.assertIn(result.selected_ar_lags, (2, 5))
            self.assertGreaterEqual(result.identity_nmse, 0.0)
            self.assertGreaterEqual(result.moving_average_nmse, 0.0)
            self.assertGreaterEqual(result.current_sample_nmse, 0.0)
            self.assertGreaterEqual(result.autoregressive_nmse, 0.0)

    def test_experiment_rejects_empty_candidate_collections(self) -> None:
        with self.assertRaisesRegex(ValueError, "noise_levels"):
            run_gaussian_baseline_experiment(noise_levels=())
        with self.assertRaisesRegex(ValueError, "moving_average_windows"):
            run_gaussian_baseline_experiment(moving_average_windows=())

    def test_missing_gap_experiment_is_deterministic(self) -> None:
        arguments = {
            "n_samples": 1_000,
            "gap_lengths": (5, 10),
            "n_repeats": 2,
            "moving_average_windows": (1, 3),
            "autoregressive_lags": (2, 5),
            "autoregressive_regularizations": (1e-3,),
        }

        first = run_missing_gap_experiment(**arguments)
        second = run_missing_gap_experiment(**arguments)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 2)
        for result in first:
            self.assertIn(result.selected_window_size, (1, 3))
            self.assertIn(result.selected_ar_lags, (2, 5))
            self.assertGreaterEqual(result.carried_forward_nmse, 0.0)
            self.assertGreaterEqual(result.masked_moving_average_nmse, 0.0)
            self.assertGreaterEqual(result.autoregressive_nmse, 0.0)

    def test_missing_gap_experiment_rejects_invalid_repeat_count(self) -> None:
        for n_repeats in (0, -1, 1.5, True):
            with self.subTest(n_repeats=n_repeats):
                with self.assertRaisesRegex(ValueError, "n_repeats"):
                    run_missing_gap_experiment(n_repeats=n_repeats)


if __name__ == "__main__":
    unittest.main()
