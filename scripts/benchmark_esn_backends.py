"""Compare NumPy and optional PyTorch ESN state generation."""

import argparse
import statistics
import time

import numpy as np

from rc_photonics.esn import ESNConfig, EchoStateNetwork
from rc_photonics.signals import generate_mackey_glass
from rc_photonics.torch_esn import TorchEchoStateNetwork, torch


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=6_000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    arguments = _arguments()
    if arguments.samples <= 0 or arguments.repeats <= 0:
        raise ValueError("samples and repeats must be positive")
    config = ESNConfig(
        n_nodes=100,
        spectral_radius=0.95,
        leak_rate=0.2,
        input_scaling=0.5,
        seed=42,
    )
    signal = generate_mackey_glass(arguments.samples)
    inputs = np.column_stack((signal, np.ones(signal.size)))
    numpy_model = EchoStateNetwork(config)
    torch_model = TorchEchoStateNetwork(config, device=arguments.device)
    torch_inputs = torch.tensor(
        inputs,
        dtype=torch.float64,
        device=arguments.device,
    )

    numpy_times: list[float] = []
    torch_times: list[float] = []
    numpy_states = numpy_model.collect_states(inputs)
    with torch.no_grad():
        torch_states = torch_model.collect_states(torch_inputs)

    for _ in range(arguments.repeats):
        started = time.perf_counter()
        numpy_states = numpy_model.collect_states(inputs, reset=True)
        numpy_times.append(time.perf_counter() - started)

        started = time.perf_counter()
        with torch.no_grad():
            torch_states = torch_model.collect_states(torch_inputs, reset=True)
            if torch_inputs.is_cuda:
                torch.cuda.synchronize(torch_inputs.device)
        torch_times.append(time.perf_counter() - started)

    torch_numpy = torch_states.detach().cpu().numpy()
    maximum_difference = float(np.max(np.abs(numpy_states - torch_numpy)))
    numpy_median = statistics.median(numpy_times)
    torch_median = statistics.median(torch_times)
    print(f"samples={arguments.samples}")
    print(f"nodes={config.n_nodes}")
    print(f"device={arguments.device}")
    print(f"numpy_median_seconds={numpy_median:.6f}")
    print(f"torch_median_seconds={torch_median:.6f}")
    print(f"torch_to_numpy_ratio={torch_median / numpy_median:.6f}")
    print(f"maximum_state_difference={maximum_difference:.3e}")


if __name__ == "__main__":
    main()
