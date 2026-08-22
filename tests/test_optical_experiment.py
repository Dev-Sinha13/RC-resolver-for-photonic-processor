import unittest

import numpy as np

from rc_photonics.esn import ESNConfig
from rc_photonics.optical_channel import OpticalLinkConfig
from rc_photonics.optical_experiment import (
    causal_tap_matrix,
    run_optical_equalization_experiment,
)
from rc_photonics.photonic_delay import PhotonicDelayConfig


class OpticalEqualizationTests(unittest.TestCase):
    def test_tap_matrix_is_causal(self) -> None:
        values = np.arange(10, dtype=np.float64)
        changed = values.copy()
        changed[6:] += 100.0

        original_states = causal_tap_matrix(values, n_taps=4)
        changed_states = causal_tap_matrix(changed, n_taps=4)

        np.testing.assert_array_equal(original_states[:6], changed_states[:6])
        np.testing.assert_array_equal(original_states[4], [4.0, 3.0, 2.0, 1.0])

    def test_small_experiment_is_deterministic_and_finite(self) -> None:
        arguments = {
            "config": OpticalLinkConfig(
                ssfm_steps=3,
                guard_symbols=12,
                seed=17,
            ),
            "n_train_bits": 300,
            "n_validation_bits": 180,
            "n_test_bits": 300,
            "ffe_taps": 7,
            "washout": 20,
            "esn_config": ESNConfig(n_nodes=15, seed=5),
            "photonic_config": PhotonicDelayConfig(n_virtual_nodes=15, seed=5),
            "seed": 99,
        }

        first = run_optical_equalization_experiment(**arguments)
        second = run_optical_equalization_experiment(**arguments)

        for name in ("raw", "feed_forward", "esn", "photonic"):
            first_score = getattr(first, name)
            second_score = getattr(second, name)
            self.assertEqual(first_score, second_score)
            self.assertTrue(0.0 <= first_score.bit_error_rate <= 1.0)
            self.assertTrue(np.isfinite(first_score.nmse))
        np.testing.assert_array_equal(first.test_bits, second.test_bits)
        for name in first.test_predictions:
            np.testing.assert_array_equal(
                first.test_predictions[name],
                second.test_predictions[name],
            )

    def test_invalid_partition_size_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            run_optical_equalization_experiment(
                n_train_bits=20,
                n_validation_bits=20,
                n_test_bits=20,
                washout=20,
            )

    def test_receiver_latency_does_not_misalign_raw_symbols(self) -> None:
        ideal = OpticalLinkConfig(
            fibre_length_km=0.0,
            attenuation_db_per_km=0.0,
            dispersion_ps_nm_km=0.0,
            nonlinear_coefficient_per_w_km=0.0,
            transmitter_bandwidth_ghz=None,
            receiver_bandwidth_ghz=None,
            detector_snr_db=None,
            timing_jitter_std_ui=0.0,
            ssfm_steps=2,
            guard_symbols=8,
        )
        result = run_optical_equalization_experiment(
            ideal,
            n_train_bits=128,
            n_validation_bits=96,
            n_test_bits=128,
            ffe_taps=5,
            decision_delay_symbols=1,
            washout=10,
            esn_config=ESNConfig(n_nodes=8, input_dim=1, seed=3),
            photonic_config=PhotonicDelayConfig(
                n_virtual_nodes=8,
                input_dim=1,
                seed=3,
            ),
        )

        self.assertEqual(result.raw.bit_error_rate, 0.0)


if __name__ == "__main__":
    unittest.main()
