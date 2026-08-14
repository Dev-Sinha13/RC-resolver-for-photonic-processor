"""Controlled observation corruption for restoration experiments."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class CorruptionResult:
    """A corrupted signal and metadata describing the affected samples.

    ``observation_mask`` is false only where no measurement is available. A
    noisy measurement remains an observation, even when ``corruption_mask`` is
    true. This distinction will let a future reservoir distinguish an actual
    zero-valued measurement from a missing value.
    """

    values: NDArray[np.float64]
    observation_mask: NDArray[np.bool_]
    corruption_mask: NDArray[np.bool_]


def _as_signal(signal: NDArray[np.floating]) -> NDArray[np.float64]:
    values = np.asarray(signal, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("signal must be one-dimensional")
    if not np.all(np.isfinite(values)):
        raise ValueError("signal must contain only finite values")
    return values


def add_gaussian_noise(
    signal: NDArray[np.floating],
    *,
    standard_deviation: float,
    seed: int | None = None,
) -> CorruptionResult:
    """Add independent zero-mean Gaussian measurement noise."""

    if standard_deviation < 0:
        raise ValueError("standard_deviation cannot be negative")

    values = _as_signal(signal)
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, standard_deviation, size=values.shape)
    corrupted = values + noise
    changed = np.full(values.shape, standard_deviation > 0, dtype=np.bool_)
    return CorruptionResult(
        values=corrupted,
        observation_mask=np.ones(values.shape, dtype=np.bool_),
        corruption_mask=changed,
    )


def add_impulse_noise(
    signal: NDArray[np.floating],
    *,
    probability: float,
    magnitude: float,
    seed: int | None = None,
) -> CorruptionResult:
    """Add fixed-magnitude impulses with independently sampled random signs."""

    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be between zero and one")
    if magnitude < 0:
        raise ValueError("magnitude cannot be negative")

    values = _as_signal(signal)
    rng = np.random.default_rng(seed)
    selected = rng.random(values.shape) < probability
    signs = rng.choice(np.array([-1.0, 1.0]), size=values.shape)
    impulses = selected * signs * magnitude
    changed = selected & (magnitude > 0)
    return CorruptionResult(
        values=values + impulses,
        observation_mask=np.ones(values.shape, dtype=np.bool_),
        corruption_mask=changed,
    )


def mask_interval(
    signal: NDArray[np.floating],
    *,
    start: int,
    length: int,
    fill_value: float = 0.0,
) -> CorruptionResult:
    """Replace one contiguous interval and mark it as unobserved."""

    values = _as_signal(signal)
    if start < 0:
        raise ValueError("start cannot be negative")
    if length <= 0:
        raise ValueError("length must be positive")
    if start + length > values.size:
        raise ValueError("masked interval extends beyond the signal")
    if not np.isfinite(fill_value):
        raise ValueError("fill_value must be finite")

    corrupted = values.copy()
    corrupted[start : start + length] = fill_value
    observed = np.ones(values.shape, dtype=np.bool_)
    observed[start : start + length] = False
    changed = ~observed
    return CorruptionResult(
        values=corrupted,
        observation_mask=observed,
        corruption_mask=changed,
    )
