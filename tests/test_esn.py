import unittest

import numpy as np

from rc_photonics.esn import ESNConfig, EchoStateNetwork


class EchoStateNetworkTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = ESNConfig(
            n_nodes=20,
            input_dim=2,
            spectral_radius=0.8,
            leak_rate=0.4,
            connectivity=0.3,
            seed=7,
        )
        self.inputs = np.column_stack(
            (np.linspace(-1.0, 1.0, 30), np.ones(30))
        )

    def test_seeded_networks_are_deterministic(self) -> None:
        first = EchoStateNetwork(self.config).collect_states(self.inputs)
        second = EchoStateNetwork(self.config).collect_states(self.inputs)

        np.testing.assert_array_equal(first, second)

    def test_state_shape_and_spectral_radius(self) -> None:
        network = EchoStateNetwork(self.config)
        states = network.collect_states(self.inputs)

        self.assertEqual(states.shape, (30, 20))
        self.assertEqual(states.dtype, np.float64)
        self.assertAlmostEqual(network.spectral_radius, 0.8, places=10)
        self.assertTrue(np.all(np.isfinite(states)))

    def test_reset_reproduces_state_sequence(self) -> None:
        network = EchoStateNetwork(self.config)
        first = network.collect_states(self.inputs, reset=True)
        network.collect_states(self.inputs, reset=False)
        second = network.collect_states(self.inputs, reset=True)

        np.testing.assert_array_equal(first, second)

    def test_future_inputs_do_not_change_past_states(self) -> None:
        changed = self.inputs.copy()
        changed[15:, 0] += 1_000.0

        original_states = EchoStateNetwork(self.config).collect_states(self.inputs)
        changed_states = EchoStateNetwork(self.config).collect_states(changed)

        np.testing.assert_array_equal(original_states[:15], changed_states[:15])

    def test_invalid_configuration_is_rejected(self) -> None:
        invalid_configs = (
            ESNConfig(n_nodes=0),
            ESNConfig(spectral_radius=0.0),
            ESNConfig(leak_rate=1.1),
            ESNConfig(connectivity=0.0),
        )

        for config in invalid_configs:
            with self.subTest(config=config):
                with self.assertRaises(ValueError):
                    EchoStateNetwork(config)

    def test_invalid_input_shape_is_rejected(self) -> None:
        network = EchoStateNetwork(self.config)

        with self.assertRaisesRegex(ValueError, "input_dim"):
            network.collect_states(np.ones((10, 1)))


if __name__ == "__main__":
    unittest.main()
