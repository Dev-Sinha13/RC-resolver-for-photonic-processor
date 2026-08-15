"""Time-multiplexed photonic delay-reservoir simulation."""

from dataclasses import dataclass
from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rc_photonics.hardware import HardwareImpairmentModel, HardwareImpairments


@dataclass(frozen=True)
class PhotonicDelayConfig:
    """Parameters of the discrete virtual-node delay approximation."""

    n_virtual_nodes: int = 50
    input_dim: int = 2
    feedback_gain: float = 0.8
    leak_rate: float = 0.3
    input_scaling: float = 1.0
    phase_bias: float = 0.2
    seed: int = 42

    def validate(self) -> None:
        for name, value in (
            ("n_virtual_nodes", self.n_virtual_nodes),
            ("input_dim", self.input_dim),
        ):
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise ValueError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if not np.isfinite(self.feedback_gain) or self.feedback_gain < 0.0:
            raise ValueError("feedback_gain must be finite and non-negative")
        if not np.isfinite(self.leak_rate) or not 0.0 < self.leak_rate <= 1.0:
            raise ValueError("leak_rate must be in (0, 1]")
        if not np.isfinite(self.input_scaling) or self.input_scaling < 0.0:
            raise ValueError("input_scaling must be finite and non-negative")
        if not np.isfinite(self.phase_bias):
            raise ValueError("phase_bias must be finite")


class PhotonicDelayReservoir:
    """A scalar nonlinear delay node sampled as virtual reservoir nodes.

    The ``sin²`` response approximates a Mach-Zehnder intensity modulator. A
    leaky serial coupling represents the finite response time of the physical
    node, while the previous round-trip states provide delayed feedback.
    """

    def __init__(
        self,
        config: PhotonicDelayConfig | None = None,
        *,
        impairments: HardwareImpairments | None = None,
    ) -> None:
        self.config = config or PhotonicDelayConfig()
        self.config.validate()
        rng = np.random.default_rng(self.config.seed)
        self._input_mask = rng.choice(
            np.array([-1.0, 1.0]),
            size=(self.config.n_virtual_nodes, self.config.input_dim),
        )
        self._state = np.zeros(
            self.config.n_virtual_nodes,
            dtype=np.float64,
        )
        self.impairments = impairments or HardwareImpairments()
        self._hardware = HardwareImpairmentModel(self.impairments)

    @property
    def state(self) -> NDArray[np.float64]:
        return self._state.copy()

    @property
    def input_mask(self) -> NDArray[np.float64]:
        return self._input_mask.copy()

    def reset(self) -> None:
        self._state.fill(0.0)
        self._hardware.reset()

    def step(self, input_vector: ArrayLike) -> NDArray[np.float64]:
        values = np.asarray(input_vector, dtype=np.float64)
        if values.ndim != 1 or values.shape != (self.config.input_dim,):
            raise ValueError("input_vector must match input_dim")
        if not np.all(np.isfinite(values)):
            raise ValueError("input_vector must contain only finite values")

        delayed_states = self._state.copy()
        updated_states = np.empty_like(delayed_states)
        serial_state = float(delayed_states[-1])
        for node_index in range(self.config.n_virtual_nodes):
            feedback = self._hardware.perturb_feedback(
                float(delayed_states[node_index])
            )
            masked_input = float(self._input_mask[node_index] @ values)
            drive = (
                self.config.feedback_gain * feedback
                + self.config.input_scaling * masked_input
                + self.config.phase_bias
            )
            drive = self._hardware.perturb_drive(drive)
            nonlinear_state = self._hardware.perturb_state(
                float(np.sin(drive) ** 2)
            )
            serial_state = (
                (1.0 - self.config.leak_rate) * serial_state
                + self.config.leak_rate * nonlinear_state
            )
            updated_states[node_index] = serial_state

        self._state = updated_states
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
            (values.shape[0], self.config.n_virtual_nodes),
            dtype=np.float64,
        )
        for index, input_vector in enumerate(values):
            states[index] = self.step(input_vector)
        return states
