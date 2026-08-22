import unittest
from dataclasses import replace

import numpy as np

from rc_photonics.optical_channel import (
    OpticalLinkConfig,
    bit_error_rate,
    generate_ook_bits,
    hard_decisions,
    select_binary_threshold,
    simulate_ook_link,
)


class OpticalChannelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.bits = np.tile(np.array([0, 1, 1, 0], dtype=np.int8), 64)
        self.ideal = OpticalLinkConfig(
            samples_per_symbol=8,
            fibre_length_km=0.0,
            attenuation_db_per_km=0.0,
            dispersion_ps_nm_km=0.0,
            nonlinear_coefficient_per_w_km=0.0,
            transmitter_bandwidth_ghz=None,
            receiver_bandwidth_ghz=None,
            detector_snr_db=None,
            timing_jitter_std_ui=0.0,
            ssfm_steps=4,
            guard_symbols=16,
            seed=9,
        )

    def test_ideal_link_has_zero_ber(self) -> None:
        result = simulate_ook_link(self.bits, self.ideal)
        threshold = select_binary_threshold(self.bits, result.sampled_values)
        decisions = hard_decisions(result.sampled_values, threshold=threshold)

        self.assertEqual(bit_error_rate(self.bits, decisions), 0.0)
        self.assertEqual(result.detector_waveform.size, self.bits.size * 8)
        np.testing.assert_array_equal(result.bits, self.bits)

    def test_seeded_nonideal_link_is_deterministic(self) -> None:
        config = OpticalLinkConfig(seed=123, ssfm_steps=4, guard_symbols=16)

        first = simulate_ook_link(self.bits, config)
        second = simulate_ook_link(self.bits, config)

        np.testing.assert_array_equal(first.detector_waveform, second.detector_waveform)
        np.testing.assert_array_equal(first.sampled_values, second.sampled_values)

    def test_attenuation_reduces_received_power_by_expected_ratio(self) -> None:
        config = replace(
            self.ideal,
            fibre_length_km=20.0,
            attenuation_db_per_km=0.2,
        )
        result = simulate_ook_link(self.bits, config)
        observed_ratio = float(
            np.sum(result.received_power) / np.sum(result.transmitted_power)
        )
        expected_ratio = 10.0 ** (-(0.2 * 20.0) / 10.0)

        self.assertAlmostEqual(observed_ratio, expected_ratio, places=10)

    def test_dispersion_changes_pulse_shape_without_changing_total_energy(self) -> None:
        config = replace(
            self.ideal,
            fibre_length_km=25.0,
            dispersion_ps_nm_km=16.7,
        )
        result = simulate_ook_link(self.bits, config)

        self.assertGreater(
            float(np.max(np.abs(result.received_power - result.transmitted_power))),
            1e-6,
        )
        relative_energy_change = abs(
            float(np.sum(result.received_power))
            - float(np.sum(result.transmitted_power))
        ) / float(np.sum(result.transmitted_power))
        self.assertLess(
            relative_energy_change,
            1e-3,
        )

    def test_pure_kerr_effect_preserves_intensity(self) -> None:
        config = replace(
            self.ideal,
            fibre_length_km=10.0,
            nonlinear_coefficient_per_w_km=2.0,
        )
        result = simulate_ook_link(self.bits, config)

        np.testing.assert_allclose(
            result.received_power,
            result.transmitted_power,
            rtol=1e-12,
            atol=1e-12,
        )

    def test_split_step_refinement_converges(self) -> None:
        bits = generate_ook_bits(128, seed=71)
        coarse_config = OpticalLinkConfig(
            ssfm_steps=16,
            detector_snr_db=None,
            timing_jitter_std_ui=0.0,
            guard_symbols=32,
            seed=4,
        )
        coarse = simulate_ook_link(bits, coarse_config).received_power
        medium = simulate_ook_link(
            bits,
            replace(coarse_config, ssfm_steps=32),
        ).received_power
        fine = simulate_ook_link(
            bits,
            replace(coarse_config, ssfm_steps=64),
        ).received_power

        coarse_change = np.linalg.norm(coarse - medium) / np.linalg.norm(medium)
        fine_change = np.linalg.norm(medium - fine) / np.linalg.norm(fine)

        self.assertLess(fine_change, coarse_change)
        self.assertLess(fine_change, 2e-5)

    def test_threshold_and_ber_match_hand_calculation(self) -> None:
        target = np.array([0, 0, 1, 1], dtype=np.int8)
        scores = np.array([0.1, 0.4, 0.6, 0.9])
        threshold = select_binary_threshold(target, scores)

        self.assertEqual(bit_error_rate(target, hard_decisions(scores, threshold=threshold)), 0.0)

    def test_invalid_configuration_and_bits_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            OpticalLinkConfig(samples_per_symbol=1).validate()
        with self.assertRaises(ValueError):
            OpticalLinkConfig(receiver_bandwidth_ghz=100.0).validate()
        with self.assertRaises(ValueError):
            generate_ook_bits(0, seed=1)
        with self.assertRaises(ValueError):
            simulate_ook_link([0, 2, 1], self.ideal)


if __name__ == "__main__":
    unittest.main()
