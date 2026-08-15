import unittest

from rc_photonics.hardware import (
    HardwareImpairmentModel,
    HardwareImpairments,
)


class HardwareImpairmentTests(unittest.TestCase):
    def test_zero_impairments_are_transparent(self) -> None:
        model = HardwareImpairmentModel(HardwareImpairments())

        self.assertEqual(model.perturb_feedback(0.4), 0.4)
        self.assertEqual(model.perturb_drive(0.7), 0.7)
        self.assertEqual(model.perturb_state(0.6), 0.6)

    def test_seeded_noise_is_reproducible_after_reset(self) -> None:
        model = HardwareImpairmentModel(
            HardwareImpairments(internal_noise_std=0.1, seed=9)
        )
        first = [model.perturb_state(0.5) for _ in range(10)]
        model.reset()
        second = [model.perturb_state(0.5) for _ in range(10)]

        self.assertEqual(first, second)

    def test_quantization_uses_requested_levels(self) -> None:
        model = HardwareImpairmentModel(
            HardwareImpairments(quantization_bits=2)
        )

        self.assertAlmostEqual(model.perturb_state(0.6), 2.0 / 3.0)

    def test_feedback_attenuation(self) -> None:
        model = HardwareImpairmentModel(
            HardwareImpairments(feedback_attenuation=0.25)
        )

        self.assertAlmostEqual(model.perturb_feedback(0.8), 0.6)

    def test_invalid_configuration_is_rejected(self) -> None:
        invalid = (
            HardwareImpairments(internal_noise_std=-1.0),
            HardwareImpairments(feedback_attenuation=1.0),
            HardwareImpairments(quantization_bits=0),
            HardwareImpairments(drift_std=-0.1),
        )
        for config in invalid:
            with self.subTest(config=config):
                with self.assertRaises(ValueError):
                    HardwareImpairmentModel(config)


if __name__ == "__main__":
    unittest.main()
