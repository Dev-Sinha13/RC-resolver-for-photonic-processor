"""Causal classical baselines for signal restoration."""

from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _as_signal(signal: ArrayLike) -> NDArray[np.float64]:
    """Return a validated, non-empty, one-dimensional signal."""
    values = np.asarray(signal, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("signal must be one-dimensional")
    if values.size == 0:
        raise ValueError("signal must not be empty")
    if not np.all(np.isfinite(values)):
        raise ValueError("signal must contain only finite values")
    return values


def _as_observation_mask(
    observation_mask: ArrayLike,
    *,
    expected_shape: tuple[int, ...],
) -> NDArray[np.bool_]:
    """Return a validated Boolean mask matching a signal shape."""
    mask = np.asarray(observation_mask)
    if mask.dtype != np.bool_:
        raise ValueError("observation_mask must contain Boolean values")
    if mask.ndim != 1 or mask.shape != expected_shape:
        raise ValueError("observation_mask must match the signal shape")
    return np.asarray(mask, dtype=np.bool_)


def _validate_window_size(window_size: int) -> int:
    if isinstance(window_size, bool) or not isinstance(window_size, Integral):
        raise ValueError("window_size must be an integer")
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    return int(window_size)


def identity_restoration(signal: ArrayLike) -> NDArray[np.float64]:
    """Return an independent copy of the observed signal."""
    return _as_signal(signal).copy()


def causal_moving_average(
    signal: ArrayLike,
    *,
    window_size: int,
) -> NDArray[np.float64]:
    """Average the current and preceding samples within a fixed window.

    At the beginning of the sequence, all available samples are used until a
    complete window exists. No future observation contributes to any output.
    """
    validated_window_size = _validate_window_size(window_size)
    values = _as_signal(signal)
    cumulative_sum = np.concatenate(
        (np.array([0.0], dtype=np.float64), np.cumsum(values, dtype=np.float64))
    )
    ends = np.arange(1, values.size + 1)
    starts = np.maximum(0, ends - validated_window_size)
    window_sums = cumulative_sum[ends] - cumulative_sum[starts]
    window_lengths = ends - starts
    return np.asarray(window_sums / window_lengths, dtype=np.float64)


def causal_median_filter(
    signal: ArrayLike,
    *,
    window_size: int,
) -> NDArray[np.float64]:
    """Return a causal rolling median robust to isolated impulses."""
    validated_window_size = _validate_window_size(window_size)
    values = _as_signal(signal)
    restored = np.empty_like(values)
    for index in range(values.size):
        start = max(0, index - validated_window_size + 1)
        restored[index] = float(np.median(values[start : index + 1]))
    return restored


def last_observation_carried_forward(
    signal: ArrayLike,
    observation_mask: ArrayLike,
) -> NDArray[np.float64]:
    """Fill missing samples with the most recent available observation."""
    values = _as_signal(signal)
    mask = _as_observation_mask(observation_mask, expected_shape=values.shape)
    restored = values.copy()
    last_observation: float | None = None

    for index, is_observed in enumerate(mask):
        if is_observed:
            last_observation = float(values[index])
        elif last_observation is None:
            raise ValueError("the first sample must be observed")
        else:
            restored[index] = last_observation

    return restored


def causal_masked_moving_average(
    signal: ArrayLike,
    observation_mask: ArrayLike,
    *,
    window_size: int,
) -> NDArray[np.float64]:
    """Average the most recent observed values without using future samples.

    ``window_size`` counts available observations rather than elapsed time.
    Missing samples are never added to the history, so a fill value such as
    zero cannot bias the result.
    """
    validated_window_size = _validate_window_size(window_size)
    values = _as_signal(signal)
    mask = _as_observation_mask(observation_mask, expected_shape=values.shape)
    restored = np.empty_like(values)
    recent_observations: list[float] = []

    for index, is_observed in enumerate(mask):
        if is_observed:
            recent_observations.append(float(values[index]))
            if len(recent_observations) > validated_window_size:
                recent_observations.pop(0)
        if not recent_observations:
            raise ValueError("the first sample must be observed")
        restored[index] = float(np.mean(recent_observations))

    return restored
