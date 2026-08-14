import unittest

import numpy as np

from rc_photonics.baselines import causal_moving_average, identity_restoration


class BaselineTests(unittest.TestCase):
    def test_identity_returns_independent_float64_copy(self) -> None:
        signal = np.array([1, 2, 3], dtype=np.int64)
        restored = identity_restoration(signal)

        self.assertEqual(restored.dtype, np.float64)
        np.testing.assert_array_equal(restored, signal)

        restored[0] = 100.0
        self.assertEqual(signal[0], 1)

    def test_causal_moving_average_matches_hand_calculation(self) -> None:
        restored = causal_moving_average(
            [1.0, 2.0, 3.0, 4.0],
            window_size=3,
        )

        np.testing.assert_allclose(restored, [1.0, 1.5, 2.0, 3.0])
        self.assertEqual(restored.dtype, np.float64)

    def test_window_one_matches_identity(self) -> None:
        signal = np.array([-1.0, 0.0, 2.0])

        np.testing.assert_array_equal(
            causal_moving_average(signal, window_size=1),
            signal,
        )

    def test_future_input_does_not_change_past_outputs(self) -> None:
        original = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        changed = original.copy()
        changed[3:] = 1_000.0

        original_output = causal_moving_average(original, window_size=3)
        changed_output = causal_moving_average(changed, window_size=3)

        np.testing.assert_array_equal(original_output[:3], changed_output[:3])

    def test_baselines_do_not_mutate_input(self) -> None:
        signal = np.array([1.0, 2.0, 3.0])
        original = signal.copy()

        identity_restoration(signal)
        causal_moving_average(signal, window_size=2)

        np.testing.assert_array_equal(signal, original)

    def test_invalid_window_sizes_are_rejected(self) -> None:
        for window_size in (0, -1, 1.5, True):
            with self.subTest(window_size=window_size):
                with self.assertRaises(ValueError):
                    causal_moving_average([1.0, 2.0], window_size=window_size)

    def test_invalid_signals_are_rejected(self) -> None:
        invalid_signals = ([], [[1.0, 2.0]], [1.0, np.nan])

        for signal in invalid_signals:
            with self.subTest(signal=signal):
                with self.assertRaises(ValueError):
                    identity_restoration(signal)
                with self.assertRaises(ValueError):
                    causal_moving_average(signal, window_size=2)


if __name__ == "__main__":
    unittest.main()
