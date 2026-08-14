import unittest

import numpy as np

from rc_photonics.signals import MackeyGlassParameters, generate_mackey_glass


class MackeyGlassTests(unittest.TestCase):
    def test_generator_returns_finite_nonconstant_signal(self) -> None:
        signal = generate_mackey_glass(500, washout=1_000)

        self.assertEqual(signal.shape, (500,))
        self.assertEqual(signal.dtype, np.float64)
        self.assertTrue(np.all(np.isfinite(signal)))
        self.assertGreater(float(np.std(signal)), 0.01)

    def test_generator_is_deterministic(self) -> None:
        first = generate_mackey_glass(100)
        second = generate_mackey_glass(100)

        np.testing.assert_array_equal(first, second)

    def test_delay_must_align_with_integration_step(self) -> None:
        parameters = MackeyGlassParameters(tau=17.05, dt=0.1)

        with self.assertRaisesRegex(ValueError, "integer multiple"):
            generate_mackey_glass(10, parameters=parameters)


if __name__ == "__main__":
    unittest.main()
