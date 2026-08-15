"""Conventional leaky echo-state network reservoir."""

from dataclasses import dataclass
from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class ESNConfig:
    """Fixed hyperparameters for a seeded echo-state reservoir."""

    n_nodes: int = 100
    input_dim: int = 2
    spectral_radius: float = 0.9
    leak_rate: float = 0.3
    input_scaling: float = 1.0
    bias_scaling: float = 0.1
    connectivity: float = 0.1
    seed: int = 42

    def validate(self) -> None:
        for name, value in (
            ("n_nodes", self.n_nodes),
            ("input_dim", self.input_dim),
        ):
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise ValueError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if not np.isfinite(self.spectral_radius) or self.spectral_radius <= 0.0:
            raise ValueError("spectral_radius must be finite and positive")
        if not np.isfinite(self.leak_rate) or not 0.0 < self.leak_rate <= 1.0:
            raise ValueError("leak_rate must be in (0, 1]")
        if not np.isfinite(self.input_scaling) or self.input_scaling < 0.0:
            raise ValueError("input_scaling must be finite and non-negative")
        if not np.isfinite(self.bias_scaling) or self.bias_scaling < 0.0:
            raise ValueError("bias_scaling must be finite and non-negative")
        if not np.isfinite(self.connectivity) or not 0.0 < self.connectivity <= 1.0:
            raise ValueError("connectivity must be in (0, 1]")


class EchoStateNetwork:
    """A deterministic, fixed-weight leaky recurrent reservoir."""

    def __init__(self, config: ESNConfig | None = None) -> None:
        self.config = config or ESNConfig()
        self.config.validate()
        rng = np.random.default_rng(self.config.seed)
        self._input_weights = rng.uniform(
            -self.config.input_scaling,
            self.config.input_scaling,
            size=(self.config.n_nodes, self.config.input_dim),
        )
        self._bias = rng.uniform(
            -self.config.bias_scaling,
            self.config.bias_scaling,
            size=self.config.n_nodes,
        )
        recurrent = rng.uniform(
            -1.0,
            1.0,
            size=(self.config.n_nodes, self.config.n_nodes),
        )
        connectivity_mask = (
            rng.random(recurrent.shape) < self.config.connectivity
        )
        recurrent *= connectivity_mask
        radius = float(np.max(np.abs(np.linalg.eigvals(recurrent))))
        if radius == 0.0:
            recurrent[0, 0] = 1.0
            radius = 1.0
        self._recurrent_weights = np.asarray(
            recurrent * (self.config.spectral_radius / radius),
            dtype=np.float64,
        )
        self._state = np.zeros(self.config.n_nodes, dtype=np.float64)

    @property
    def state(self) -> NDArray[np.float64]:
        return self._state.copy()

    @property
    def input_weights(self) -> NDArray[np.float64]:
        return self._input_weights.copy()

    @property
    def recurrent_weights(self) -> NDArray[np.float64]:
        return self._recurrent_weights.copy()

    @property
    def spectral_radius(self) -> float:
        return float(
            np.max(np.abs(np.linalg.eigvals(self._recurrent_weights)))
        )

    def reset(self) -> None:
        self._state.fill(0.0)

    def step(self, input_vector: ArrayLike) -> NDArray[np.float64]:
        values = np.asarray(input_vector, dtype=np.float64)
        if values.ndim != 1 or values.shape != (self.config.input_dim,):
            raise ValueError("input_vector must match input_dim")
        if not np.all(np.isfinite(values)):
            raise ValueError("input_vector must contain only finite values")

        candidate = np.tanh(
            self._input_weights @ values
            + self._recurrent_weights @ self._state
            + self._bias
        )
        self._state = (
            (1.0 - self.config.leak_rate) * self._state
            + self.config.leak_rate * candidate
        )
        return self.state

    def collect_states(
        self,
        inputs: ArrayLike,
        *,
        reset: bool = True,
    ) -> NDArray[np.float64]:
        values = np.asarray(inputs, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.config.input_dim:
            raise ValueError("inputs must have shape (n_samples, input_dim)")
        if values.shape[0] == 0:
            raise ValueError("inputs must not be empty")
        if not np.all(np.isfinite(values)):
            raise ValueError("inputs must contain only finite values")
        if reset:
            self.reset()

        states = np.empty(
            (values.shape[0], self.config.n_nodes),
            dtype=np.float64,
        )
        for index, input_vector in enumerate(values):
            states[index] = self.step(input_vector)
        return states
