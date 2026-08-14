"""Reproducible end-to-end experiments for restoration baselines."""

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from rc_photonics.baselines import causal_moving_average, identity_restoration
from rc_photonics.corruption import add_gaussian_noise
from rc_photonics.datasets import chronological_split
from rc_photonics.metrics import normalized_mean_squared_error
from rc_photonics.signals import generate_mackey_glass


@dataclass(frozen=True)
class GaussianBaselineResult:
    """Test-set scores for one Gaussian measurement-noise level."""

    noise_standard_deviation: float
    selected_window_size: int
    identity_nmse: float
    moving_average_nmse: float


def run_gaussian_baseline_experiment(
    *,
    n_samples: int = 6_000,
    noise_levels: Iterable[float] = (0.02, 0.05, 0.1, 0.2),
    moving_average_windows: Iterable[int] = (1, 3, 5, 9, 15, 25),
    seed: int = 42,
) -> tuple[GaussianBaselineResult, ...]:
    """Select causal moving-average windows and evaluate them on held-out data.

    Window sizes are selected using only the chronological validation split.
    The selected window is then evaluated once on a separately corrupted test
    split. Independent deterministic noise draws are used for the two splits.
    """
    levels = tuple(float(level) for level in noise_levels)
    windows = tuple(moving_average_windows)

    if not levels:
        raise ValueError("noise_levels must not be empty")
    if not windows:
        raise ValueError("moving_average_windows must not be empty")
    if any(not np.isfinite(level) or level < 0.0 for level in levels):
        raise ValueError("noise levels must be finite and non-negative")

    clean = generate_mackey_glass(n_samples)
    split = chronological_split(clean)
    results: list[GaussianBaselineResult] = []

    for index, noise_level in enumerate(levels):
        validation_observation = add_gaussian_noise(
            split.validation,
            standard_deviation=noise_level,
            seed=seed + 2 * index,
        ).values

        validation_scores = {
            window: normalized_mean_squared_error(
                split.validation,
                causal_moving_average(
                    validation_observation,
                    window_size=window,
                ),
            )
            for window in windows
        }
        selected_window = min(
            windows,
            key=lambda window: (validation_scores[window], window),
        )

        test_observation = add_gaussian_noise(
            split.test,
            standard_deviation=noise_level,
            seed=seed + 2 * index + 1,
        ).values
        identity_prediction = identity_restoration(test_observation)
        moving_average_prediction = causal_moving_average(
            test_observation,
            window_size=selected_window,
        )

        results.append(
            GaussianBaselineResult(
                noise_standard_deviation=noise_level,
                selected_window_size=int(selected_window),
                identity_nmse=normalized_mean_squared_error(
                    split.test,
                    identity_prediction,
                ),
                moving_average_nmse=normalized_mean_squared_error(
                    split.test,
                    moving_average_prediction,
                ),
            )
        )

    return tuple(results)
