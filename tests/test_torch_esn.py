import unittest

import numpy as np

from rc_photonics.esn import ESNConfig, EchoStateNetwork
from rc_photonics.readout import fit_ridge_readout

try:
    import torch

    from rc_photonics.torch_esn import (
        TorchESNAdapter,
        TorchEchoStateNetwork,
        fit_torch_ridge_readout,
    )
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "optional PyTorch dependency is not installed")
class TorchEchoStateNetworkTests(unittest.TestCase):
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

    def test_states_match_numpy_reference(self) -> None:
        expected = EchoStateNetwork(self.config).collect_states(self.inputs)
        actual = (
            TorchEchoStateNetwork(self.config)
            .collect_states(self.inputs)
            .detach()
            .cpu()
            .numpy()
        )

        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    def test_fixed_matrices_are_buffers_not_parameters(self) -> None:
        model = TorchEchoStateNetwork(self.config)

        self.assertEqual(tuple(model.parameters()), tuple())
        self.assertIn("input_weights", model.state_dict())
        self.assertIn("recurrent_weights", model.state_dict())
        self.assertAlmostEqual(model.spectral_radius, 0.8, places=10)

    def test_forward_supports_input_gradients(self) -> None:
        model = TorchEchoStateNetwork(self.config)
        inputs = torch.tensor(
            self.inputs,
            dtype=torch.float64,
            requires_grad=True,
        )

        model(inputs).square().mean().backward()

        self.assertIsNotNone(inputs.grad)
        self.assertTrue(bool(torch.all(torch.isfinite(inputs.grad))))
        self.assertGreater(float(torch.linalg.vector_norm(inputs.grad)), 0.0)

    def test_reset_and_causality(self) -> None:
        model = TorchEchoStateNetwork(self.config)
        first = model.collect_states(self.inputs, reset=True).detach().numpy()
        model.collect_states(self.inputs, reset=False)
        second = model.collect_states(self.inputs, reset=True).detach().numpy()
        changed = self.inputs.copy()
        changed[15:, 0] += 1_000.0
        changed_states = (
            model.collect_states(changed, reset=True).detach().numpy()
        )

        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(first[:15], changed_states[:15])

    def test_numpy_adapter_matches_reference(self) -> None:
        expected = EchoStateNetwork(self.config).collect_states(self.inputs)
        actual = TorchESNAdapter(self.config).collect_states(self.inputs)

        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    def test_torch_ridge_matches_numpy_readout(self) -> None:
        states = EchoStateNetwork(self.config).collect_states(self.inputs)
        targets = np.sin(np.linspace(0.0, 2.0, states.shape[0]))
        expected = fit_ridge_readout(
            states,
            targets,
            regularization=1e-3,
        ).predict(states)
        torch_states = torch.tensor(states, dtype=torch.float64)
        readout = fit_torch_ridge_readout(
            torch_states,
            torch.tensor(targets, dtype=torch.float64),
            regularization=1e-3,
        )
        actual = readout(torch_states).detach().numpy()

        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
