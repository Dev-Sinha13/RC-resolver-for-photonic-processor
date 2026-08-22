"""Optional PyTorch implementation of the digital echo-state network."""

from numbers import Real
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

try:
    import torch
    from torch import Tensor, nn
except ModuleNotFoundError as error:  # pragma: no cover - environment dependent
    raise ModuleNotFoundError(
        "PyTorch support requires the optional dependency: "
        "pip install 'rc-photonics[torch]'"
    ) from error

from rc_photonics.esn import ESNConfig, EchoStateNetwork


def _floating_dtype(dtype: torch.dtype) -> torch.dtype:
    if not dtype.is_floating_point:
        raise ValueError("dtype must be a floating-point torch dtype")
    return dtype


class TorchEchoStateNetwork(nn.Module):
    """PyTorch ESN initialized exactly from the NumPy reference weights.

    Reservoir matrices are registered as non-trainable buffers. ``forward``
    remains differentiable with respect to its inputs while the fixed weights
    stay outside gradient-based optimization.
    """

    def __init__(
        self,
        config: ESNConfig | None = None,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        self.config = config or ESNConfig()
        self.config.validate()
        selected_dtype = _floating_dtype(dtype)
        reference = EchoStateNetwork(self.config)

        def tensor(values: NDArray[np.float64]) -> Tensor:
            return torch.as_tensor(
                values,
                dtype=selected_dtype,
                device=device,
            ).clone()

        self.register_buffer("input_weights", tensor(reference.input_weights))
        self.register_buffer(
            "recurrent_weights",
            tensor(reference.recurrent_weights),
        )
        self.register_buffer("bias", tensor(reference.bias))
        self.register_buffer(
            "_state",
            torch.zeros(
                self.config.n_nodes,
                dtype=selected_dtype,
                device=device,
            ),
            persistent=False,
        )

    @property
    def state(self) -> Tensor:
        return self._state.detach().clone()

    @property
    def spectral_radius(self) -> float:
        eigenvalues = torch.linalg.eigvals(self.recurrent_weights)
        return float(torch.max(torch.abs(eigenvalues)).detach().cpu())

    def reset(self) -> None:
        with torch.no_grad():
            self._state.zero_()

    def _inputs_tensor(self, inputs: ArrayLike | Tensor) -> Tensor:
        if isinstance(inputs, Tensor):
            return inputs.to(
                device=self.input_weights.device,
                dtype=self.input_weights.dtype,
            )
        return torch.as_tensor(
            np.asarray(inputs),
            device=self.input_weights.device,
            dtype=self.input_weights.dtype,
        )

    def step(self, input_vector: ArrayLike | Tensor) -> Tensor:
        """Advance one stateful inference step."""
        values = self._inputs_tensor(input_vector)
        if values.ndim != 1 or values.shape != (self.config.input_dim,):
            raise ValueError("input_vector must match input_dim")
        if not bool(torch.all(torch.isfinite(values))):
            raise ValueError("input_vector must contain only finite values")

        candidate = torch.tanh(
            self.input_weights @ values
            + self.recurrent_weights @ self._state
            + self.bias
        )
        state = (
            (1.0 - self.config.leak_rate) * self._state
            + self.config.leak_rate * candidate
        )
        with torch.no_grad():
            self._state.copy_(state.detach())
        return state

    def forward(
        self,
        inputs: ArrayLike | Tensor,
        *,
        reset: bool = True,
    ) -> Tensor:
        """Collect causal states with shape ``(samples, reservoir_size)``."""
        values = self._inputs_tensor(inputs)
        if values.ndim != 2 or values.shape[1] != self.config.input_dim:
            raise ValueError("inputs must have shape (n_samples, input_dim)")
        if values.shape[0] == 0:
            raise ValueError("inputs must not be empty")
        if not bool(torch.all(torch.isfinite(values))):
            raise ValueError("inputs must contain only finite values")

        state = torch.zeros_like(self._state) if reset else self._state.clone()
        states: list[Tensor] = []
        for input_vector in values:
            candidate = torch.tanh(
                self.input_weights @ input_vector
                + self.recurrent_weights @ state
                + self.bias
            )
            state = (
                (1.0 - self.config.leak_rate) * state
                + self.config.leak_rate * candidate
            )
            states.append(state)

        with torch.no_grad():
            self._state.copy_(state.detach())
        return torch.stack(states)

    def collect_states(
        self,
        inputs: ArrayLike | Tensor,
        *,
        reset: bool = True,
    ) -> Tensor:
        return self.forward(inputs, reset=reset)


class TorchESNAdapter:
    """Expose PyTorch ESN states through the project's NumPy protocol."""

    def __init__(
        self,
        config: ESNConfig | None = None,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        self.model = TorchEchoStateNetwork(
            config,
            device=device,
            dtype=dtype,
        )

    @property
    def config(self) -> ESNConfig:
        return self.model.config

    def reset(self) -> None:
        self.model.reset()

    def collect_states(
        self,
        inputs: ArrayLike,
        *,
        reset: bool = True,
    ) -> NDArray[np.float64]:
        with torch.no_grad():
            states = self.model.collect_states(inputs, reset=reset)
        return np.asarray(states.detach().cpu(), dtype=np.float64)


class TorchRidgeReadout(nn.Module):
    """Fixed affine PyTorch readout fitted by closed-form ridge regression."""

    def __init__(self, weights: Tensor, intercept: Tensor, penalty: float) -> None:
        super().__init__()
        if weights.ndim != 1 or weights.numel() == 0:
            raise ValueError("weights must be a non-empty one-dimensional tensor")
        if intercept.numel() != 1:
            raise ValueError("intercept must contain one value")
        self.register_buffer("weights", weights.detach().clone())
        self.register_buffer("intercept", intercept.detach().reshape(()).clone())
        self.penalty = float(penalty)

    def forward(self, states: Tensor) -> Tensor:
        if states.ndim != 2 or states.shape[1] != self.weights.numel():
            raise ValueError("states must match the trained state count")
        return states @ self.weights + self.intercept


def fit_torch_ridge_readout(
    states: Tensor,
    targets: Tensor,
    *,
    regularization: float,
) -> TorchRidgeReadout:
    """Fit a differentiable-runtime readout with an unregularized intercept."""
    if states.ndim != 2 or states.shape[0] == 0 or states.shape[1] == 0:
        raise ValueError("states must be a non-empty two-dimensional tensor")
    if targets.ndim != 1 or targets.shape[0] != states.shape[0]:
        raise ValueError("targets must contain one value per state row")
    if not states.dtype.is_floating_point or not targets.dtype.is_floating_point:
        raise ValueError("states and targets must use floating-point dtypes")
    if not bool(torch.all(torch.isfinite(states))) or not bool(
        torch.all(torch.isfinite(targets))
    ):
        raise ValueError("states and targets must contain only finite values")
    if (
        isinstance(regularization, bool)
        or not isinstance(regularization, Real)
        or not np.isfinite(regularization)
        or regularization < 0.0
    ):
        raise ValueError("regularization must be finite and non-negative")

    targets = targets.to(device=states.device, dtype=states.dtype)
    state_mean = torch.mean(states, dim=0)
    target_mean = torch.mean(targets)
    centered_states = states - state_mean
    centered_targets = targets - target_mean
    penalty = float(regularization)
    if penalty == 0.0:
        weights = torch.linalg.lstsq(
            centered_states,
            centered_targets.unsqueeze(1),
        ).solution.squeeze(1)
    else:
        gram = centered_states.T @ centered_states
        right_hand_side = centered_states.T @ centered_targets
        weights = torch.linalg.solve(
            gram
            + penalty
            * torch.eye(
                states.shape[1],
                device=states.device,
                dtype=states.dtype,
            ),
            right_hand_side,
        )
    intercept = target_mean - state_mean @ weights
    return TorchRidgeReadout(weights, intercept, penalty)


def torch_environment() -> dict[str, Any]:
    """Return concise backend information for reports and diagnostics."""
    return {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
    }
