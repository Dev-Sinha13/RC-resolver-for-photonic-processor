"""Leakage-safe dataset utilities for temporal experiments."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ChronologicalSplit:
    """Contiguous train, validation, and test partitions."""

    train: NDArray[np.float64]
    validation: NDArray[np.float64]
    test: NDArray[np.float64]


def chronological_split(
    signal: NDArray[np.floating],
    *,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
) -> ChronologicalSplit:
    """Split a signal without shuffling or leaking future observations."""

    values = np.asarray(signal, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("signal must be one-dimensional")
    if not np.all(np.isfinite(values)):
        raise ValueError("signal must contain only finite values")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between zero and one")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one")
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("train and validation fractions must leave a test split")

    train_end = int(values.size * train_fraction)
    validation_end = train_end + int(values.size * validation_fraction)
    if train_end == 0 or validation_end == train_end or validation_end == values.size:
        raise ValueError("signal is too short for the requested split fractions")

    return ChronologicalSplit(
        train=values[:train_end].copy(),
        validation=values[train_end:validation_end].copy(),
        test=values[validation_end:].copy(),
    )
