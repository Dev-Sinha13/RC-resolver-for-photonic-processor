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
    if isinstance(window_size, bool) or not isinstance(window_size, Integral):
        raise ValueError("window_size must be an integer")
    if window_size <= 0:
        raise ValueError("window_size must be positive")

    values = _as_signal(signal)
    cumulative_sum = np.concatenate(
        (np.array([0.0], dtype=np.float64), np.cumsum(values, dtype=np.float64))
    )
    ends = np.arange(1, values.size + 1)
    starts = np.maximum(0, ends - int(window_size))
    window_sums = cumulative_sum[ends] - cumulative_sum[starts]
    window_lengths = ends - starts
    return np.asarray(window_sums / window_lengths, dtype=np.float64)
