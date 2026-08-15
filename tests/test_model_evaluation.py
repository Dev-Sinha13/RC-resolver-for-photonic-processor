import unittest

import numpy as np

from rc_photonics.esn import ESNConfig, EchoStateNetwork
from rc_photonics.model_evaluation import (
    ReservoirCandidate,
    fit_reservoir_restorer,
    default_photonic_candidates,
    make_reservoir_inputs,
    run_photonic_robustness_experiment,
    run_reservoir_gap_experiment,
    run_reservoir_gaussian_experiment,
    run_reservoir_impulse_experiment,
)
from rc_photonics.hardware import HardwareImpairments
from rc_photonics.photonic_delay import PhotonicDelayConfig


class ReservoirEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        config = ESNConfig(n_nodes=10, connectivity=0.5, seed=3)
        self.candidates = (
            ReservoirCandidate(
                "small_esn",
                lambda: EchoStateNetwork(config),
            ),
        )

    def test_inputs_include_value_and_mask_channels(self) -> None:
        inputs = make_reservoir_inputs(
            [1.0, 0.0, 3.0],
            [True, False, True],
        )

        np.testing.assert_array_equal(
            inputs,
            [[1.0, 1.0], [0.0, 0.0], [3.0, 1.0]],
        )

    def test_fit_and_predict_preserve_sequence_shape(self) -> None:
        signal = np.sin(np.linspace(0.0, 8.0, 200))
        mask = np.ones(signal.shape, dtype=np.bool_)
        restorer = fit_reservoir_restorer(
            self.candidates[0].factory(),
            signal,
            mask,
            signal,
            regularization=1e-3,
            washout=20,
        )

        prediction = restorer.predict(signal, mask)
        self.assertEqual(prediction.shape, signal.shape)
        self.assertTrue(np.all(np.isfinite(prediction)))

    def test_small_gaussian_experiment_is_deterministic(self) -> None:
        arguments = {
            "n_samples": 1_000,
            "noise_levels": (0.1,),
            "readout_regularizations": (1e-3,),
            "washout": 20,
        }

        first = run_reservoir_gaussian_experiment(
            self.candidates,
            **arguments,
        )
        second = run_reservoir_gaussian_experiment(
            self.candidates,
            **arguments,
        )

        self.assertEqual(first, second)
        self.assertGreaterEqual(first[0].identity_nmse, 0.0)
        self.assertGreaterEqual(first[0].nmse, 0.0)

    def test_small_gap_experiment_is_deterministic(self) -> None:
        arguments = {
            "n_samples": 1_000,
            "gap_lengths": (5,),
            "readout_regularizations": (1e-3,),
            "washout": 20,
            "n_repeats": 2,
        }

        first = run_reservoir_gap_experiment(self.candidates, **arguments)
        second = run_reservoir_gap_experiment(self.candidates, **arguments)

        self.assertEqual(first, second)
        self.assertGreaterEqual(first[0].nmse, 0.0)

    def test_small_impulse_experiment_is_deterministic(self) -> None:
        arguments = {
            "n_samples": 1_000,
            "impulse_probabilities": (0.05,),
            "readout_regularizations": (1e-3,),
            "washout": 20,
        }

        first = run_reservoir_impulse_experiment(self.candidates, **arguments)
        second = run_reservoir_impulse_experiment(self.candidates, **arguments)

        self.assertEqual(first, second)
        self.assertGreaterEqual(first[0].identity_nmse, 0.0)

    def test_photonic_candidates_use_common_interface(self) -> None:
        candidate = default_photonic_candidates()[0]
        inputs = np.column_stack((np.linspace(0.0, 1.0, 10), np.ones(10)))

        states = candidate.factory().collect_states(inputs)

        self.assertEqual(states.shape[0], 10)
        self.assertTrue(np.all(np.isfinite(states)))

    def test_robustness_experiment_is_reproducible(self) -> None:
        arguments = {
            "base_config": PhotonicDelayConfig(n_virtual_nodes=10, seed=4),
            "impairment_cases": (
                ("ideal", HardwareImpairments()),
                (
                    "noise",
                    HardwareImpairments(internal_noise_std=0.01, seed=8),
                ),
            ),
            "n_samples": 1_000,
            "washout": 20,
        }

        first = run_photonic_robustness_experiment(**arguments)
        second = run_photonic_robustness_experiment(**arguments)

        self.assertEqual(first, second)
        self.assertEqual([result.label for result in first], ["ideal", "noise"])


if __name__ == "__main__":
    unittest.main()
