import unittest

import numpy as np

from rc_photonics.hardware import HardwareImpairments
from rc_photonics.photonic_delay import (
    PhotonicDelayConfig,
    PhotonicDelayReservoir,
)


class PhotonicDelayReservoirTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = PhotonicDelayConfig(
            n_virtual_nodes=20,
            feedback_gain=0.7,
            leak_rate=0.4,
            seed=11,
        )
        self.inputs = np.column_stack(
            (np.linspace(-1.0, 1.0, 30), np.ones(30))
        )

    def test_seeded_reservoir_is_deterministic(self) -> None:
        first = PhotonicDelayReservoir(self.config).collect_states(self.inputs)
        second = PhotonicDelayReservoir(self.config).collect_states(self.inputs)

        np.testing.assert_array_equal(first, second)

    def test_state_shape_bounds_and_type(self) -> None:
        states = PhotonicDelayReservoir(self.config).collect_states(self.inputs)

        self.assertEqual(states.shape, (30, 20))
        self.assertEqual(states.dtype, np.float64)
        self.assertTrue(np.all(states >= 0.0))
        self.assertTrue(np.all(states <= 1.0))

    def test_reset_reproduces_state_sequence(self) -> None:
        reservoir = PhotonicDelayReservoir(self.config)
        first = reservoir.collect_states(self.inputs)
        reservoir.collect_states(self.inputs, reset=False)
        second = reservoir.collect_states(self.inputs, reset=True)

        np.testing.assert_array_equal(first, second)

    def test_future_inputs_do_not_change_past_states(self) -> None:
        changed = self.inputs.copy()
        changed[15:, 0] += 1_000.0

        original = PhotonicDelayReservoir(self.config).collect_states(self.inputs)
        modified = PhotonicDelayReservoir(self.config).collect_states(changed)

        np.testing.assert_array_equal(original[:15], modified[:15])

    def test_zero_impairments_match_ideal_reservoir(self) -> None:
        ideal = PhotonicDelayReservoir(self.config).collect_states(self.inputs)
        explicit_zero = PhotonicDelayReservoir(
            self.config,
            impairments=HardwareImpairments(),
        ).collect_states(self.inputs)

        np.testing.assert_array_equal(ideal, explicit_zero)

    def test_noisy_reservoir_is_seeded_and_distinct(self) -> None:
        impairments = HardwareImpairments(internal_noise_std=0.02, seed=5)
        first = PhotonicDelayReservoir(
            self.config,
            impairments=impairments,
        ).collect_states(self.inputs)
        second = PhotonicDelayReservoir(
            self.config,
            impairments=impairments,
        ).collect_states(self.inputs)
        ideal = PhotonicDelayReservoir(self.config).collect_states(self.inputs)

        np.testing.assert_array_equal(first, second)
        self.assertFalse(np.array_equal(first, ideal))


if __name__ == "__main__":
    unittest.main()
