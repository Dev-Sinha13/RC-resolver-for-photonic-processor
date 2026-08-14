import unittest

import numpy as np

from rc_photonics.datasets import chronological_split


class ChronologicalSplitTests(unittest.TestCase):
    def test_split_preserves_order(self) -> None:
        signal = np.arange(10, dtype=np.float64)

        split = chronological_split(
            signal,
            train_fraction=0.6,
            validation_fraction=0.2,
        )

        np.testing.assert_array_equal(split.train, np.arange(6))
        np.testing.assert_array_equal(split.validation, np.arange(6, 8))
        np.testing.assert_array_equal(split.test, np.arange(8, 10))

    def test_split_returns_independent_arrays(self) -> None:
        signal = np.arange(10, dtype=np.float64)
        split = chronological_split(signal)

        signal[0] = 100.0

        self.assertEqual(split.train[0], 0.0)

    def test_split_rejects_no_test_partition(self) -> None:
        with self.assertRaisesRegex(ValueError, "leave a test split"):
            chronological_split(
                np.arange(10),
                train_fraction=0.8,
                validation_fraction=0.2,
            )


if __name__ == "__main__":
    unittest.main()
