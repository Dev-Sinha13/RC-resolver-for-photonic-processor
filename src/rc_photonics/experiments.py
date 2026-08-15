"""Reproducible end-to-end experiments for restoration baselines."""

from dataclasses import dataclass
from numbers import Integral
from typing import Iterable

import numpy as np

from rc_photonics.autoregression import (
    AutoregressiveRidge,
    fit_autoregressive_ridge,
    fit_current_sample_ridge,
)
from rc_photonics.baselines import (
    causal_masked_moving_average,
    causal_moving_average,
    identity_restoration,
    last_observation_carried_forward,
)
from rc_photonics.corruption import add_gaussian_noise, mask_interval
from rc_photonics.datasets import chronological_split
from rc_photonics.metrics import mean_squared_error, normalized_mean_squared_error
from rc_photonics.signals import generate_mackey_glass


@dataclass(frozen=True)
class GaussianBaselineResult:
    """Test-set denoising scores for one Gaussian noise level."""

    noise_standard_deviation: float
    selected_window_size: int
    selected_current_regularization: float
    selected_ar_lags: int
    selected_ar_regularization: float
    identity_nmse: float
    moving_average_nmse: float
    current_sample_nmse: float
    autoregressive_nmse: float


@dataclass(frozen=True)
class MissingGapResult:
    """Average gap-only scores for one missing-interval length.

    Each NMSE divides gap MSE by the variance of the complete clean test
    split. The fixed denominator makes scores comparable across gap lengths.
    """

    gap_length: int
    selected_window_size: int
    selected_ar_lags: int
    selected_ar_regularization: float
    carried_forward_nmse: float
    masked_moving_average_nmse: float
    autoregressive_nmse: float


def _positive_integer_candidates(
    values: Iterable[int],
    *,
    name: str,
) -> tuple[int, ...]:
    candidates = tuple(values)
    if not candidates:
        raise ValueError(f"{name} must not be empty")
    if any(
        isinstance(value, bool)
        or not isinstance(value, Integral)
        or value <= 0
        for value in candidates
    ):
        raise ValueError(f"{name} must contain positive integers")
    return tuple(int(value) for value in candidates)


def _regularization_candidates(
    values: Iterable[float],
    *,
    name: str,
) -> tuple[float, ...]:
    candidates = tuple(float(value) for value in values)
    if not candidates:
        raise ValueError(f"{name} must not be empty")
    if any(not np.isfinite(value) or value < 0.0 for value in candidates):
        raise ValueError(f"{name} must contain finite non-negative values")
    return candidates


def _aligned_ar_prediction(
    model: AutoregressiveRidge,
    signal: np.ndarray,
    *,
    evaluation_start: int,
) -> np.ndarray:
    predictions = model.predict(signal)
    return predictions[evaluation_start - model.n_lags :]


def _fit_ar_candidates(
    observations: np.ndarray,
    targets: np.ndarray,
    *,
    lags: tuple[int, ...],
    regularizations: tuple[float, ...],
) -> tuple[AutoregressiveRidge, ...]:
    return tuple(
        fit_autoregressive_ridge(
            observations,
            targets,
            n_lags=n_lags,
            regularization=regularization,
        )
        for n_lags in lags
        for regularization in regularizations
    )


def run_gaussian_baseline_experiment(
    *,
    n_samples: int = 6_000,
    noise_levels: Iterable[float] = (0.02, 0.05, 0.1, 0.2),
    moving_average_windows: Iterable[int] = (1, 3, 5, 9, 15, 25),
    current_regularizations: Iterable[float] = (0.0, 1e-3, 0.1, 1.0),
    autoregressive_lags: Iterable[int] = (5, 20, 50, 100, 200),
    autoregressive_regularizations: Iterable[float] = (1e-6, 1e-3, 0.1),
    seed: int = 42,
) -> tuple[GaussianBaselineResult, ...]:
    """Tune classical denoisers on validation data and evaluate held-out data."""
    levels = tuple(float(level) for level in noise_levels)
    if not levels:
        raise ValueError("noise_levels must not be empty")
    if any(not np.isfinite(level) or level < 0.0 for level in levels):
        raise ValueError("noise levels must be finite and non-negative")

    windows = _positive_integer_candidates(
        moving_average_windows,
        name="moving_average_windows",
    )
    current_penalties = _regularization_candidates(
        current_regularizations,
        name="current_regularizations",
    )
    ar_lags = _positive_integer_candidates(
        autoregressive_lags,
        name="autoregressive_lags",
    )
    ar_penalties = _regularization_candidates(
        autoregressive_regularizations,
        name="autoregressive_regularizations",
    )
    evaluation_start = max(ar_lags)

    clean = generate_mackey_glass(n_samples)
    split = chronological_split(clean)
    if split.train.size <= evaluation_start:
        raise ValueError("training split is too short for autoregressive_lags")
    if split.validation.size <= evaluation_start or split.test.size <= evaluation_start:
        raise ValueError("evaluation splits are too short for autoregressive_lags")

    results: list[GaussianBaselineResult] = []
    for index, noise_level in enumerate(levels):
        train_observation = add_gaussian_noise(
            split.train,
            standard_deviation=noise_level,
            seed=seed + 3 * index,
        ).values
        validation_observation = add_gaussian_noise(
            split.validation,
            standard_deviation=noise_level,
            seed=seed + 3 * index + 1,
        ).values
        test_observation = add_gaussian_noise(
            split.test,
            standard_deviation=noise_level,
            seed=seed + 3 * index + 2,
        ).values
        validation_target = split.validation[evaluation_start:]
        test_target = split.test[evaluation_start:]

        window_scores = {
            window: normalized_mean_squared_error(
                validation_target,
                causal_moving_average(
                    validation_observation,
                    window_size=window,
                )[evaluation_start:],
            )
            for window in windows
        }
        selected_window = min(
            windows,
            key=lambda window: (window_scores[window], window),
        )

        current_models = tuple(
            fit_current_sample_ridge(
                train_observation,
                split.train,
                regularization=regularization,
            )
            for regularization in current_penalties
        )
        current_scores = {
            model.regularization: normalized_mean_squared_error(
                validation_target,
                model.predict(validation_observation)[evaluation_start:],
            )
            for model in current_models
        }
        selected_current = min(
            current_models,
            key=lambda model: (
                current_scores[model.regularization],
                model.regularization,
            ),
        )

        ar_models = _fit_ar_candidates(
            train_observation,
            split.train,
            lags=ar_lags,
            regularizations=ar_penalties,
        )
        ar_scores = {
            (model.n_lags, model.regularization): normalized_mean_squared_error(
                validation_target,
                _aligned_ar_prediction(
                    model,
                    validation_observation,
                    evaluation_start=evaluation_start,
                ),
            )
            for model in ar_models
        }
        selected_ar = min(
            ar_models,
            key=lambda model: (
                ar_scores[(model.n_lags, model.regularization)],
                model.n_lags,
                model.regularization,
            ),
        )

        results.append(
            GaussianBaselineResult(
                noise_standard_deviation=noise_level,
                selected_window_size=selected_window,
                selected_current_regularization=selected_current.regularization,
                selected_ar_lags=selected_ar.n_lags,
                selected_ar_regularization=selected_ar.regularization,
                identity_nmse=normalized_mean_squared_error(
                    test_target,
                    identity_restoration(test_observation)[evaluation_start:],
                ),
                moving_average_nmse=normalized_mean_squared_error(
                    test_target,
                    causal_moving_average(
                        test_observation,
                        window_size=selected_window,
                    )[evaluation_start:],
                ),
                current_sample_nmse=normalized_mean_squared_error(
                    test_target,
                    selected_current.predict(test_observation)[evaluation_start:],
                ),
                autoregressive_nmse=normalized_mean_squared_error(
                    test_target,
                    _aligned_ar_prediction(
                        selected_ar,
                        test_observation,
                        evaluation_start=evaluation_start,
                    ),
                ),
            )
        )

    return tuple(results)


def _gap_starts(
    *,
    signal_length: int,
    gap_length: int,
    n_repeats: int,
    minimum_history: int,
) -> tuple[int, ...]:
    latest_start = signal_length - gap_length
    if latest_start < minimum_history:
        raise ValueError("gap length leaves insufficient causal history")
    if n_repeats <= 0:
        raise ValueError("n_repeats must be positive")
    starts = np.linspace(
        minimum_history,
        latest_start,
        num=n_repeats,
        dtype=np.int64,
    )
    return tuple(int(start) for start in np.unique(starts))


def _gap_score(
    clean: np.ndarray,
    prediction: np.ndarray,
    *,
    start: int,
    length: int,
    normalization_variance: float,
) -> float:
    gap_slice = slice(start, start + length)
    return (
        mean_squared_error(clean[gap_slice], prediction[gap_slice])
        / normalization_variance
    )


def run_missing_gap_experiment(
    *,
    n_samples: int = 6_000,
    gap_lengths: Iterable[int] = (5, 10, 20, 40, 80),
    n_repeats: int = 5,
    moving_average_windows: Iterable[int] = (1, 3, 5, 10, 20, 40),
    autoregressive_lags: Iterable[int] = (5, 20, 50, 100, 200),
    autoregressive_regularizations: Iterable[float] = (1e-6, 1e-3, 0.1),
) -> tuple[MissingGapResult, ...]:
    """Evaluate causal reconstruction as contiguous missing gaps grow."""
    if (
        isinstance(n_repeats, bool)
        or not isinstance(n_repeats, Integral)
        or n_repeats <= 0
    ):
        raise ValueError("n_repeats must be a positive integer")
    validated_n_repeats = int(n_repeats)
    gaps = _positive_integer_candidates(gap_lengths, name="gap_lengths")
    windows = _positive_integer_candidates(
        moving_average_windows,
        name="moving_average_windows",
    )
    ar_lags = _positive_integer_candidates(
        autoregressive_lags,
        name="autoregressive_lags",
    )
    ar_penalties = _regularization_candidates(
        autoregressive_regularizations,
        name="autoregressive_regularizations",
    )
    maximum_lag = max(ar_lags)

    clean = generate_mackey_glass(n_samples)
    split = chronological_split(clean)
    ar_models = _fit_ar_candidates(
        split.train,
        split.train,
        lags=ar_lags,
        regularizations=ar_penalties,
    )
    validation_variance = float(np.var(split.validation, ddof=0))
    test_variance = float(np.var(split.test, ddof=0))
    results: list[MissingGapResult] = []

    for gap_length in gaps:
        validation_starts = _gap_starts(
            signal_length=split.validation.size,
            gap_length=gap_length,
            n_repeats=validated_n_repeats,
            minimum_history=maximum_lag,
        )
        test_starts = _gap_starts(
            signal_length=split.test.size,
            gap_length=gap_length,
            n_repeats=validated_n_repeats,
            minimum_history=maximum_lag,
        )

        window_scores: dict[int, list[float]] = {window: [] for window in windows}
        ar_scores: dict[tuple[int, float], list[float]] = {
            (model.n_lags, model.regularization): [] for model in ar_models
        }
        for start in validation_starts:
            corruption = mask_interval(
                split.validation,
                start=start,
                length=gap_length,
            )
            for window in windows:
                prediction = causal_masked_moving_average(
                    corruption.values,
                    corruption.observation_mask,
                    window_size=window,
                )
                window_scores[window].append(
                    _gap_score(
                        split.validation,
                        prediction,
                        start=start,
                        length=gap_length,
                        normalization_variance=validation_variance,
                    )
                )
            for model in ar_models:
                prediction = model.restore_missing(
                    corruption.values,
                    corruption.observation_mask,
                )
                ar_scores[(model.n_lags, model.regularization)].append(
                    _gap_score(
                        split.validation,
                        prediction,
                        start=start,
                        length=gap_length,
                        normalization_variance=validation_variance,
                    )
                )

        selected_window = min(
            windows,
            key=lambda window: (float(np.mean(window_scores[window])), window),
        )
        selected_ar = min(
            ar_models,
            key=lambda model: (
                float(np.mean(ar_scores[(model.n_lags, model.regularization)])),
                model.n_lags,
                model.regularization,
            ),
        )

        carried_scores: list[float] = []
        masked_average_scores: list[float] = []
        selected_ar_scores: list[float] = []
        for start in test_starts:
            corruption = mask_interval(
                split.test,
                start=start,
                length=gap_length,
            )
            predictions = (
                last_observation_carried_forward(
                    corruption.values,
                    corruption.observation_mask,
                ),
                causal_masked_moving_average(
                    corruption.values,
                    corruption.observation_mask,
                    window_size=selected_window,
                ),
                selected_ar.restore_missing(
                    corruption.values,
                    corruption.observation_mask,
                ),
            )
            score_lists = (
                carried_scores,
                masked_average_scores,
                selected_ar_scores,
            )
            for prediction, score_list in zip(predictions, score_lists):
                score_list.append(
                    _gap_score(
                        split.test,
                        prediction,
                        start=start,
                        length=gap_length,
                        normalization_variance=test_variance,
                    )
                )

        results.append(
            MissingGapResult(
                gap_length=gap_length,
                selected_window_size=selected_window,
                selected_ar_lags=selected_ar.n_lags,
                selected_ar_regularization=selected_ar.regularization,
                carried_forward_nmse=float(np.mean(carried_scores)),
                masked_moving_average_nmse=float(
                    np.mean(masked_average_scores)
                ),
                autoregressive_nmse=float(np.mean(selected_ar_scores)),
            )
        )

    return tuple(results)
