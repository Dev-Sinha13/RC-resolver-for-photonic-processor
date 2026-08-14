import unittest

import numpy as np

from rc_photonics.corruption import (
    add_gaussian_noise,
    add_impulse_noise,
    mask_interval,
)


class CorruptionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.signal = np.linspace(-1.0, 1.0, 20)

    def test_gaussian_noise_is_reproducible(self) -> None:
        first = add_gaussian_noise(
            self.signal,
            standard_deviation=0.2,
            seed=42,
        )
        second = add_gaussian_noise(
            self.signal,
            standard_deviation=0.2,
            seed=42,
        )

        np.testing.assert_array_equal(first.values, second.values)
        self.assertTrue(np.all(first.observation_mask))
        self.assertTrue(np.all(first.corruption_mask))

    def test_zero_gaussian_noise_does_not_mark_corruption(self) -> None:
        result = add_gaussian_noise(
            self.signal,
            standard_deviation=0.0,
            seed=42,
        )

        np.testing.assert_array_equal(result.values, self.signal)
        self.assertFalse(np.any(result.corruption_mask))

    def test_impulse_noise_reports_changed_samples(self) -> None:
        result = add_impulse_noise(
            self.signal,
            probability=1.0,
            magnitude=2.0,
            seed=7,
        )

        self.assertTrue(np.all(result.observation_mask))
        self.assertTrue(np.all(result.corruption_mask))
        np.testing.assert_allclose(np.abs(result.values - self.signal), 2.0)

    def test_mask_interval_marks_samples_unobserved(self) -> None:
        result = mask_interval(self.signal, start=5, length=4)

        np.testing.assert_array_equal(result.values[5:9], 0.0)
        self.assertTrue(np.all(result.observation_mask[:5]))
        self.assertFalse(np.any(result.observation_mask[5:9]))
        self.assertTrue(np.all(result.observation_mask[9:]))
        np.testing.assert_array_equal(
            result.corruption_mask,
            ~result.observation_mask,
        )

    def test_corruption_does_not_mutate_input(self) -> None:
        original = self.signal.copy()

        add_impulse_noise(
            self.signal,
            probability=1.0,
            magnitude=2.0,
            seed=7,
        )
        mask_interval(self.signal, start=2, length=3)

        np.testing.assert_array_equal(self.signal, original)


if __name__ == "__main__":
    unittest.main()
