"""Shared ridge readout for digital and photonic reservoir states."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _as_state_matrix(states: ArrayLike) -> NDArray[np.float64]:
    values = np.asarray(states, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("states must be two-dimensional")
    if values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("states must have non-zero dimensions")
    if not np.all(np.isfinite(values)):
        raise ValueError("states must contain only finite values")
    return values


def _as_targets(
    targets: ArrayLike,
    *,
    expected_samples: int,
) -> NDArray[np.float64]:
    values = np.asarray(targets, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("targets must be one-dimensional")
    if values.size != expected_samples:
        raise ValueError("states and targets must contain the same samples")
    if not np.all(np.isfinite(values)):
        raise ValueError("targets must contain only finite values")
    return values


def _validate_regularization(regularization: float) -> float:
    if (
        isinstance(regularization, bool)
        or not np.isfinite(regularization)
        or regularization < 0.0
    ):
        raise ValueError("regularization must be finite and non-negative")
    return float(regularization)


@dataclass(frozen=True)
class RidgeReadout:
    """A trained affine mapping from reservoir states to a scalar output."""

    weights: NDArray[np.float64]
    intercept: float
    regularization: float

    def __post_init__(self) -> None:
        weights = np.asarray(self.weights, dtype=np.float64)
        if weights.ndim != 1 or weights.size == 0:
            raise ValueError("weights must be a non-empty one-dimensional array")
        if not np.all(np.isfinite(weights)):
            raise ValueError("weights must contain only finite values")
        if not np.isfinite(self.intercept):
            raise ValueError("intercept must be finite")
        regularization = _validate_regularization(self.regularization)
        object.__setattr__(self, "weights", weights.copy())
        object.__setattr__(self, "intercept", float(self.intercept))
        object.__setattr__(self, "regularization", regularization)

    def predict(self, states: ArrayLike) -> NDArray[np.float64]:
        """Predict one output for each reservoir-state vector."""
        state_matrix = _as_state_matrix(states)
        if state_matrix.shape[1] != self.weights.size:
            raise ValueError("state count must match the trained weights")
        return np.asarray(
            state_matrix @ self.weights + self.intercept,
            dtype=np.float64,
        )


def fit_ridge_readout(
    states: ArrayLike,
    targets: ArrayLike,
    *,
    regularization: float,
) -> RidgeReadout:
    """Fit state weights with an unregularized intercept."""
    penalty = _validate_regularization(regularization)
    state_matrix = _as_state_matrix(states)
    target_values = _as_targets(
        targets,
        expected_samples=state_matrix.shape[0],
    )
    state_mean = np.mean(state_matrix, axis=0)
    target_mean = float(np.mean(target_values))
    centered_states = state_matrix - state_mean
    centered_targets = target_values - target_mean

    if penalty == 0.0:
        weights = np.linalg.lstsq(
            centered_states,
            centered_targets,
            rcond=None,
        )[0]
    else:
        gram_matrix = centered_states.T @ centered_states
        right_hand_side = centered_states.T @ centered_targets
        weights = np.linalg.solve(
            gram_matrix + penalty * np.eye(state_matrix.shape[1]),
            right_hand_side,
        )

    intercept = target_mean - float(state_mean @ weights)
    return RidgeReadout(
        weights=np.asarray(weights, dtype=np.float64),
        intercept=intercept,
        regularization=penalty,
    )
