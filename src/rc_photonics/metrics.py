"""Evaluation metrics for signal-restoration experiments."""

import numpy as np
from numpy.typing import ArrayLike
from numpy.typing import NDArray


def _validate_pair(
    target: ArrayLike,
    prediction: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return a validated pair of finite, non-empty 1-D arrays."""
    target_array = np.asarray(target, dtype=np.float64)
    prediction_array = np.asarray(prediction, dtype=np.float64)

    if target_array.ndim != 1 or prediction_array.ndim != 1:
        raise ValueError("target and prediction must be one-dimensional")
    if target_array.size == 0 or prediction_array.size == 0:
        raise ValueError("target and prediction must not be empty")
    if target_array.shape != prediction_array.shape:
        raise ValueError("target and prediction must have the same shape")
    if not np.all(np.isfinite(target_array)):
        raise ValueError("target must contain only finite values")
    if not np.all(np.isfinite(prediction_array)):
        raise ValueError("prediction must contain only finite values")

    return target_array, prediction_array


def mean_squared_error(target: ArrayLike, prediction: ArrayLike) -> float:
    """Return the mean squared error for two finite 1-D sequences.

    Both sequences must be non-empty and have the same shape.
    """
    target_array, prediction_array = _validate_pair(target, prediction)
    squared_errors = np.square(target_array - prediction_array)
    return float(np.mean(squared_errors))


def normalized_mean_squared_error(
    target: ArrayLike,
    prediction: ArrayLike,
) -> float:
    """Return MSE divided by the target's population variance.

    The population variance convention is ``numpy.var(target, ddof=0)``.
    A constant target has zero variance, so NMSE is undefined and must raise
    ``ValueError``.
    """
    target_array, prediction_array = _validate_pair(target, prediction)
    target_variance = float(np.var(target_array, ddof=0))

    if target_variance <= 0.0:
        raise ValueError("target variance must be greater than zero")

    squared_errors = np.square(target_array - prediction_array)
    mse = float(np.mean(squared_errors))
    return mse / target_variance
