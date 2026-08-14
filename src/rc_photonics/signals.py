"""Deterministic nonlinear signals used by restoration experiments."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class MackeyGlassParameters:
    """Parameters of the Mackey–Glass delay differential equation.

    The simulated equation is

        dx/dt = beta * x(t - tau) / (1 + x(t - tau)**power) - gamma * x(t).

    Euler integration is sufficient for the initial experimental baseline. A
    later milestone can add a higher-order solver and quantify discretization
    error without changing the public data interface.
    """

    beta: float = 0.2
    gamma: float = 0.1
    tau: float = 17.0
    power: float = 10.0
    dt: float = 0.1
    initial_value: float = 1.2

    def validate(self) -> None:
        if self.beta <= 0:
            raise ValueError("beta must be positive")
        if self.gamma <= 0:
            raise ValueError("gamma must be positive")
        if self.tau <= 0:
            raise ValueError("tau must be positive")
        if self.power <= 0:
            raise ValueError("power must be positive")
        if self.dt <= 0:
            raise ValueError("dt must be positive")
        if self.initial_value <= 0:
            raise ValueError("initial_value must be positive")

        delay_steps = self.tau / self.dt
        if not np.isclose(delay_steps, round(delay_steps), rtol=0.0, atol=1e-9):
            raise ValueError("tau must be an integer multiple of dt")


def generate_mackey_glass(
    n_samples: int,
    *,
    washout: int = 1_000,
    parameters: MackeyGlassParameters | None = None,
) -> NDArray[np.float64]:
    """Generate a Mackey–Glass trajectory with causal Euler integration.

    Args:
        n_samples: Number of returned samples after the transient is removed.
        washout: Number of initial integrated samples to discard.
        parameters: Optional equation and integration parameters.

    Returns:
        A one-dimensional ``float64`` array of length ``n_samples``.
    """

    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if washout < 0:
        raise ValueError("washout cannot be negative")

    params = parameters or MackeyGlassParameters()
    params.validate()

    delay_steps = int(round(params.tau / params.dt))
    total_length = delay_steps + washout + n_samples
    trajectory = np.full(total_length, params.initial_value, dtype=np.float64)

    for index in range(delay_steps, total_length - 1):
        current = trajectory[index]
        delayed = trajectory[index - delay_steps]
        derivative = (
            params.beta * delayed / (1.0 + delayed**params.power)
            - params.gamma * current
        )
        trajectory[index + 1] = current + params.dt * derivative

    start = delay_steps + washout
    return trajectory[start : start + n_samples].copy()
