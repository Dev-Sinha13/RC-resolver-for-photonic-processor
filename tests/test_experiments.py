import unittest

from rc_photonics.experiments import run_gaussian_baseline_experiment


class GaussianBaselineExperimentTests(unittest.TestCase):
    def test_experiment_is_deterministic_and_returns_finite_scores(self) -> None:
        arguments = {
            "n_samples": 1_000,
            "noise_levels": (0.05, 0.1),
            "moving_average_windows": (1, 3, 5),
            "seed": 7,
        }

        first = run_gaussian_baseline_experiment(**arguments)
        second = run_gaussian_baseline_experiment(**arguments)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 2)
        for result in first:
            self.assertIn(result.selected_window_size, (1, 3, 5))
            self.assertGreaterEqual(result.identity_nmse, 0.0)
            self.assertGreaterEqual(result.moving_average_nmse, 0.0)

    def test_experiment_rejects_empty_candidate_collections(self) -> None:
        with self.assertRaisesRegex(ValueError, "noise_levels"):
            run_gaussian_baseline_experiment(noise_levels=())
        with self.assertRaisesRegex(ValueError, "moving_average_windows"):
            run_gaussian_baseline_experiment(moving_average_windows=())


if __name__ == "__main__":
    unittest.main()
