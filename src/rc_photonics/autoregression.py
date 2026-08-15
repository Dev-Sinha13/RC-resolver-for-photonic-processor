"""Trainable causal autoregressive ridge baseline."""

from dataclasses import dataclass
from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _as_signal(signal: ArrayLike, *, name: str) -> NDArray[np.float64]:
    values = np.asarray(signal, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if values.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values")
    return values


def _validate_n_lags(n_lags: int) -> int:
    if isinstance(n_lags, bool) or not isinstance(n_lags, Integral):
        raise ValueError("n_lags must be an integer")
    if n_lags <= 0:
        raise ValueError("n_lags must be positive")
    return int(n_lags)


def _lag_matrix(
    signal: NDArray[np.float64],
    n_lags: int,
) -> NDArray[np.float64]:
    """Return rows ordered from the most recent to oldest lag."""
    windows = np.lib.stride_tricks.sliding_window_view(signal, n_lags)
    return np.asarray(windows[:-1, ::-1], dtype=np.float64)


@dataclass(frozen=True)
class AutoregressiveRidge:
    """A fitted linear predictor over a fixed number of past samples."""

    coefficients: NDArray[np.float64]
    intercept: float
    n_lags: int
    regularization: float

    def predict(self, signal: ArrayLike) -> NDArray[np.float64]:
        """Predict samples ``n_lags`` onward from observed past values."""
        values = _as_signal(signal, name="signal")
        if values.size <= self.n_lags:
            raise ValueError("signal must contain more samples than n_lags")
        features = _lag_matrix(values, self.n_lags)
        return np.asarray(
            features @ self.coefficients + self.intercept,
            dtype=np.float64,
        )

    def restore_missing(
        self,
        signal: ArrayLike,
        observation_mask: ArrayLike,
    ) -> NDArray[np.float64]:
        """Recursively forecast through missing samples using past outputs."""
        values = _as_signal(signal, name="signal")
        mask = np.asarray(observation_mask)
        if mask.dtype != np.bool_:
            raise ValueError("observation_mask must contain Boolean values")
        if mask.ndim != 1 or mask.shape != values.shape:
            raise ValueError("observation_mask must match the signal shape")

        missing_before_history = np.flatnonzero(~mask[: self.n_lags])
        if missing_before_history.size:
            raise ValueError("missing samples require at least n_lags of history")

        restored = values.copy()
        for index in range(self.n_lags, values.size):
            if mask[index]:
                continue
            history = restored[index - self.n_lags : index][::-1]
            restored[index] = float(history @ self.coefficients + self.intercept)
        return restored


@dataclass(frozen=True)
class CurrentSampleRidge:
    """A fitted affine mapping from the current observation to its target."""

    coefficient: float
    intercept: float
    regularization: float

    def predict(self, observations: ArrayLike) -> NDArray[np.float64]:
        values = _as_signal(observations, name="observations")
        return np.asarray(
            self.coefficient * values + self.intercept,
            dtype=np.float64,
        )


def fit_current_sample_ridge(
    observations: ArrayLike,
    targets: ArrayLike,
    *,
    regularization: float,
) -> CurrentSampleRidge:
    """Fit the no-memory current-sample regression baseline."""
    if not np.isfinite(regularization) or regularization < 0.0:
        raise ValueError("regularization must be finite and non-negative")

    observation_values = _as_signal(observations, name="observations")
    target_values = _as_signal(targets, name="targets")
    if observation_values.shape != target_values.shape:
        raise ValueError("observations and targets must have the same shape")

    observation_mean = float(np.mean(observation_values))
    target_mean = float(np.mean(target_values))
    centered_observations = observation_values - observation_mean
    centered_targets = target_values - target_mean
    denominator = float(centered_observations @ centered_observations)
    denominator += float(regularization)
    coefficient = (
        0.0
        if denominator == 0.0
        else float(centered_observations @ centered_targets) / denominator
    )
    intercept = target_mean - coefficient * observation_mean
    return CurrentSampleRidge(
        coefficient=coefficient,
        intercept=intercept,
        regularization=float(regularization),
    )


def fit_autoregressive_ridge(
    observations: ArrayLike,
    targets: ArrayLike | None = None,
    *,
    n_lags: int,
    regularization: float,
) -> AutoregressiveRidge:
    """Fit a causal ridge predictor with an unregularized intercept.

    If ``targets`` is omitted, the model learns one-step prediction of the
    observation sequence. Supplying clean targets while observations are noisy
    trains a causal denoising baseline.
    """
    validated_n_lags = _validate_n_lags(n_lags)
    if not np.isfinite(regularization) or regularization < 0.0:
        raise ValueError("regularization must be finite and non-negative")

    observation_values = _as_signal(observations, name="observations")
    target_values = (
        observation_values
        if targets is None
        else _as_signal(targets, name="targets")
    )
    if observation_values.shape != target_values.shape:
        raise ValueError("observations and targets must have the same shape")
    if observation_values.size <= validated_n_lags:
        raise ValueError("training data must contain more samples than n_lags")

    features = _lag_matrix(observation_values, validated_n_lags)
    responses = target_values[validated_n_lags:]
    feature_mean = np.mean(features, axis=0)
    response_mean = float(np.mean(responses))
    centered_features = features - feature_mean
    centered_responses = responses - response_mean

    if regularization == 0.0:
        coefficients = np.linalg.lstsq(
            centered_features,
            centered_responses,
            rcond=None,
        )[0]
    else:
        gram_matrix = centered_features.T @ centered_features
        right_hand_side = centered_features.T @ centered_responses
        coefficients = np.linalg.solve(
            gram_matrix
            + float(regularization) * np.eye(validated_n_lags),
            right_hand_side,
        )

    intercept = response_mean - float(feature_mean @ coefficients)
    return AutoregressiveRidge(
        coefficients=np.asarray(coefficients, dtype=np.float64),
        intercept=intercept,
        n_lags=validated_n_lags,
        regularization=float(regularization),
    )
