"""Common reservoir training, selection, and held-out evaluation routines."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rc_photonics.corruption import add_gaussian_noise, add_impulse_noise, mask_interval
from rc_photonics.datasets import ChronologicalSplit, chronological_split
from rc_photonics.esn import ESNConfig, EchoStateNetwork
from rc_photonics.hardware import HardwareImpairments
from rc_photonics.metrics import mean_squared_error, normalized_mean_squared_error
from rc_photonics.photonic_delay import (
    PhotonicDelayConfig,
    PhotonicDelayReservoir,
)
from rc_photonics.readout import RidgeReadout, fit_ridge_readout
from rc_photonics.signals import generate_mackey_glass


class StateReservoir(Protocol):
    """The state-collection interface shared by both reservoir types."""

    def collect_states(
        self,
        inputs: ArrayLike,
        *,
        reset: bool = True,
    ) -> NDArray[np.float64]: ...


@dataclass(frozen=True)
class ReservoirCandidate:
    """A named factory used during validation-only model selection."""

    name: str
    factory: Callable[[], StateReservoir]

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("candidate name must not be empty")


@dataclass
class ReservoirRestorer:
    """A fixed reservoir coupled to a fitted shared ridge readout."""

    reservoir: StateReservoir
    readout: RidgeReadout

    def predict(
        self,
        observations: ArrayLike,
        observation_mask: ArrayLike,
    ) -> NDArray[np.float64]:
        inputs = make_reservoir_inputs(observations, observation_mask)
        states = self.reservoir.collect_states(inputs, reset=True)
        return self.readout.predict(states)


@dataclass(frozen=True)
class ReservoirGaussianResult:
    noise_standard_deviation: float
    selected_candidate: str
    selected_regularization: float
    identity_nmse: float
    nmse: float


@dataclass(frozen=True)
class ReservoirGapResult:
    gap_length: int
    selected_candidate: str
    selected_regularization: float
    nmse: float


@dataclass(frozen=True)
class ReservoirImpulseResult:
    impulse_probability: float
    impulse_magnitude: float
    selected_candidate: str
    selected_regularization: float
    identity_nmse: float
    nmse: float


@dataclass(frozen=True)
class RobustnessResult:
    label: str
    nmse: float


def make_reservoir_inputs(
    observations: ArrayLike,
    observation_mask: ArrayLike,
) -> NDArray[np.float64]:
    """Combine values and missingness into two causal input channels."""
    values = np.asarray(observations, dtype=np.float64)
    mask = np.asarray(observation_mask)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("observations must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(values)):
        raise ValueError("observations must contain only finite values")
    if mask.dtype != np.bool_ or mask.ndim != 1 or mask.shape != values.shape:
        raise ValueError("observation_mask must be Boolean and match observations")
    return np.column_stack((values, mask.astype(np.float64)))


def fit_reservoir_restorer(
    reservoir: StateReservoir,
    observations: ArrayLike,
    observation_mask: ArrayLike,
    targets: ArrayLike,
    *,
    regularization: float,
    washout: int,
    training_selector: ArrayLike | None = None,
) -> ReservoirRestorer:
    """Collect states and fit the common readout on eligible samples."""
    inputs = make_reservoir_inputs(observations, observation_mask)
    target_values = np.asarray(targets, dtype=np.float64)
    if target_values.ndim != 1 or target_values.shape != inputs.shape[:1]:
        raise ValueError("targets must match the observation sequence")
    if not np.all(np.isfinite(target_values)):
        raise ValueError("targets must contain only finite values")
    if isinstance(washout, bool) or not isinstance(washout, int) or washout < 0:
        raise ValueError("washout must be a non-negative integer")
    if washout >= target_values.size:
        raise ValueError("washout must leave training samples")

    eligible = np.arange(target_values.size) >= washout
    if training_selector is not None:
        selector = np.asarray(training_selector)
        if selector.dtype != np.bool_ or selector.shape != target_values.shape:
            raise ValueError("training_selector must be Boolean and match targets")
        eligible &= selector
    if not np.any(eligible):
        raise ValueError("no samples remain for readout training")

    states = reservoir.collect_states(inputs, reset=True)
    readout = fit_ridge_readout(
        states[eligible],
        target_values[eligible],
        regularization=regularization,
    )
    return ReservoirRestorer(reservoir=reservoir, readout=readout)


_DEFAULT_ESN_SETTINGS = (
    (50, 0.7, 0.3, 0.5),
    (50, 0.9, 0.5, 1.0),
    (100, 0.7, 0.3, 1.0),
    (100, 0.95, 0.2, 0.5),
)


def default_esn_candidates() -> tuple[ReservoirCandidate, ...]:
    """Return a deliberately small, reproducible ESN search space."""
    candidates: list[ReservoirCandidate] = []
    for n_nodes, radius, leak, input_scale in _DEFAULT_ESN_SETTINGS:
        config = ESNConfig(
            n_nodes=n_nodes,
            spectral_radius=radius,
            leak_rate=leak,
            input_scaling=input_scale,
            seed=42,
        )
        name = (
            f"esn_n{n_nodes}_rho{radius:g}_leak{leak:g}_"
            f"in{input_scale:g}"
        )
        candidates.append(
            ReservoirCandidate(
                name=name,
                factory=lambda config=config: EchoStateNetwork(config),
            )
        )
    return tuple(candidates)


def default_torch_esn_candidates() -> tuple[ReservoirCandidate, ...]:
    """Return the same ESN search space using the optional PyTorch backend."""
    from rc_photonics.torch_esn import TorchESNAdapter

    candidates: list[ReservoirCandidate] = []
    for n_nodes, radius, leak, input_scale in _DEFAULT_ESN_SETTINGS:
        config = ESNConfig(
            n_nodes=n_nodes,
            spectral_radius=radius,
            leak_rate=leak,
            input_scaling=input_scale,
            seed=42,
        )
        name = (
            f"torch_esn_n{n_nodes}_rho{radius:g}_leak{leak:g}"
            f"_in{input_scale:g}"
        )
        candidates.append(
            ReservoirCandidate(
                name=name,
                factory=lambda config=config: TorchESNAdapter(config),
            )
        )
    return tuple(candidates)


def default_photonic_candidates() -> tuple[ReservoirCandidate, ...]:
    """Return a small search space for the delay-reservoir approximation."""
    settings = (
        (50, 0.5, 0.2, 0.5, 0.2),
        (50, 0.8, 0.5, 1.0, 0.2),
        (100, 0.5, 0.5, 0.5, np.pi / 4.0),
        (100, 0.8, 0.2, 1.0, np.pi / 4.0),
    )
    candidates: list[ReservoirCandidate] = []
    for n_nodes, feedback, leak, input_scale, phase in settings:
        config = PhotonicDelayConfig(
            n_virtual_nodes=n_nodes,
            feedback_gain=feedback,
            leak_rate=leak,
            input_scaling=input_scale,
            phase_bias=phase,
            seed=42,
        )
        name = (
            f"photonic_n{n_nodes}_fb{feedback:g}_leak{leak:g}_"
            f"in{input_scale:g}_phase{phase:.3f}"
        )
        candidates.append(
            ReservoirCandidate(
                name=name,
                factory=lambda config=config: PhotonicDelayReservoir(config),
            )
        )
    return tuple(candidates)


def _validated_penalties(values: Iterable[float]) -> tuple[float, ...]:
    penalties = tuple(float(value) for value in values)
    if not penalties:
        raise ValueError("readout_regularizations must not be empty")
    if any(not np.isfinite(value) or value < 0.0 for value in penalties):
        raise ValueError("readout regularizations must be finite and non-negative")
    return penalties


def run_reservoir_gaussian_experiment(
    candidates: Iterable[ReservoirCandidate],
    *,
    n_samples: int = 6_000,
    noise_levels: Iterable[float] = (0.02, 0.05, 0.1, 0.2),
    readout_regularizations: Iterable[float] = (1e-6, 1e-3, 0.1),
    washout: int = 200,
    seed: int = 42,
) -> tuple[ReservoirGaussianResult, ...]:
    """Tune reservoir/readout pairs on validation and score test data once."""
    candidate_values = tuple(candidates)
    if not candidate_values:
        raise ValueError("candidates must not be empty")
    levels = tuple(float(level) for level in noise_levels)
    if not levels or any(
        not np.isfinite(level) or level < 0.0 for level in levels
    ):
        raise ValueError("noise_levels must be finite and non-negative")
    penalties = _validated_penalties(readout_regularizations)
    clean = generate_mackey_glass(n_samples)
    split = chronological_split(clean)
    observed_mask_train = np.ones(split.train.shape, dtype=np.bool_)
    observed_mask_validation = np.ones(split.validation.shape, dtype=np.bool_)
    observed_mask_test = np.ones(split.test.shape, dtype=np.bool_)
    results: list[ReservoirGaussianResult] = []

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

        best_score = np.inf
        best_name = ""
        best_penalty = 0.0
        best_restorer: ReservoirRestorer | None = None
        for candidate in candidate_values:
            reservoir = candidate.factory()
            train_inputs = make_reservoir_inputs(
                train_observation,
                observed_mask_train,
            )
            validation_inputs = make_reservoir_inputs(
                validation_observation,
                observed_mask_validation,
            )
            train_states = reservoir.collect_states(train_inputs, reset=True)
            validation_states = reservoir.collect_states(
                validation_inputs,
                reset=True,
            )
            for penalty in penalties:
                readout = fit_ridge_readout(
                    train_states[washout:],
                    split.train[washout:],
                    regularization=penalty,
                )
                score = normalized_mean_squared_error(
                    split.validation[washout:],
                    readout.predict(validation_states)[washout:],
                )
                if (score, candidate.name, penalty) < (
                    best_score,
                    best_name or candidate.name,
                    best_penalty,
                ):
                    best_score = score
                    best_name = candidate.name
                    best_penalty = penalty
                    best_restorer = ReservoirRestorer(reservoir, readout)

        if best_restorer is None:
            raise RuntimeError("reservoir selection failed")
        test_prediction = best_restorer.predict(
            test_observation,
            observed_mask_test,
        )
        results.append(
            ReservoirGaussianResult(
                noise_standard_deviation=noise_level,
                selected_candidate=best_name,
                selected_regularization=best_penalty,
                identity_nmse=normalized_mean_squared_error(
                    split.test[washout:],
                    test_observation[washout:],
                ),
                nmse=normalized_mean_squared_error(
                    split.test[washout:],
                    test_prediction[washout:],
                ),
            )
        )
    return tuple(results)


def run_reservoir_impulse_experiment(
    candidates: Iterable[ReservoirCandidate],
    *,
    n_samples: int = 6_000,
    impulse_probabilities: Iterable[float] = (0.01, 0.05, 0.1, 0.2),
    impulse_magnitude: float = 0.5,
    readout_regularizations: Iterable[float] = (1e-6, 1e-3, 0.1),
    washout: int = 200,
    seed: int = 84,
) -> tuple[ReservoirImpulseResult, ...]:
    """Tune reservoir impulse denoisers on validation and score test data."""
    candidate_values = tuple(candidates)
    if not candidate_values:
        raise ValueError("candidates must not be empty")
    probabilities = tuple(float(value) for value in impulse_probabilities)
    if not probabilities or any(
        not np.isfinite(value) or not 0.0 <= value <= 1.0
        for value in probabilities
    ):
        raise ValueError("impulse probabilities must be finite and in [0, 1]")
    penalties = _validated_penalties(readout_regularizations)
    split = chronological_split(generate_mackey_glass(n_samples))
    masks = tuple(
        np.ones(partition.shape, dtype=np.bool_)
        for partition in (split.train, split.validation, split.test)
    )
    results: list[ReservoirImpulseResult] = []

    for index, probability in enumerate(probabilities):
        observations = (
            add_impulse_noise(
                split.train,
                probability=probability,
                magnitude=impulse_magnitude,
                seed=seed + 3 * index,
            ).values,
            add_impulse_noise(
                split.validation,
                probability=probability,
                magnitude=impulse_magnitude,
                seed=seed + 3 * index + 1,
            ).values,
            add_impulse_noise(
                split.test,
                probability=probability,
                magnitude=impulse_magnitude,
                seed=seed + 3 * index + 2,
            ).values,
        )
        best: tuple[float, str, float, ReservoirRestorer] | None = None
        for candidate in candidate_values:
            reservoir = candidate.factory()
            train_states = reservoir.collect_states(
                make_reservoir_inputs(observations[0], masks[0]),
                reset=True,
            )
            validation_states = reservoir.collect_states(
                make_reservoir_inputs(observations[1], masks[1]),
                reset=True,
            )
            for penalty in penalties:
                readout = fit_ridge_readout(
                    train_states[washout:],
                    split.train[washout:],
                    regularization=penalty,
                )
                score = normalized_mean_squared_error(
                    split.validation[washout:],
                    readout.predict(validation_states)[washout:],
                )
                choice = (
                    score,
                    candidate.name,
                    penalty,
                    ReservoirRestorer(reservoir, readout),
                )
                if best is None or choice[:3] < best[:3]:
                    best = choice
        if best is None:
            raise RuntimeError("reservoir selection failed")
        prediction = best[3].predict(observations[2], masks[2])
        results.append(
            ReservoirImpulseResult(
                impulse_probability=probability,
                impulse_magnitude=float(impulse_magnitude),
                selected_candidate=best[1],
                selected_regularization=best[2],
                identity_nmse=normalized_mean_squared_error(
                    split.test[washout:],
                    observations[2][washout:],
                ),
                nmse=normalized_mean_squared_error(
                    split.test[washout:],
                    prediction[washout:],
                ),
            )
        )
    return tuple(results)


def run_reservoir_on_split(
    split: ChronologicalSplit,
    candidates: Iterable[ReservoirCandidate],
    *,
    noise_standard_deviation: float = 0.1,
    readout_regularizations: Iterable[float] = (1e-6, 1e-3, 0.1),
    washout: int = 200,
    seed: int = 42,
) -> ReservoirGaussianResult:
    """Tune and evaluate a reservoir on an externally prepared clean split."""
    candidate_values = tuple(candidates)
    if not candidate_values:
        raise ValueError("candidates must not be empty")
    penalties = _validated_penalties(readout_regularizations)
    observations = (
        add_gaussian_noise(
            split.train,
            standard_deviation=noise_standard_deviation,
            seed=seed,
        ).values,
        add_gaussian_noise(
            split.validation,
            standard_deviation=noise_standard_deviation,
            seed=seed + 1,
        ).values,
        add_gaussian_noise(
            split.test,
            standard_deviation=noise_standard_deviation,
            seed=seed + 2,
        ).values,
    )
    masks = tuple(
        np.ones(partition.shape, dtype=np.bool_)
        for partition in (split.train, split.validation, split.test)
    )
    best: tuple[float, str, float, ReservoirRestorer] | None = None
    for candidate in candidate_values:
        reservoir = candidate.factory()
        train_states = reservoir.collect_states(
            make_reservoir_inputs(observations[0], masks[0]),
            reset=True,
        )
        validation_states = reservoir.collect_states(
            make_reservoir_inputs(observations[1], masks[1]),
            reset=True,
        )
        for penalty in penalties:
            readout = fit_ridge_readout(
                train_states[washout:],
                split.train[washout:],
                regularization=penalty,
            )
            score = normalized_mean_squared_error(
                split.validation[washout:],
                readout.predict(validation_states)[washout:],
            )
            choice = (
                score,
                candidate.name,
                penalty,
                ReservoirRestorer(reservoir, readout),
            )
            if best is None or choice[:3] < best[:3]:
                best = choice
    if best is None:
        raise RuntimeError("reservoir selection failed")
    prediction = best[3].predict(observations[2], masks[2])
    return ReservoirGaussianResult(
        noise_standard_deviation=noise_standard_deviation,
        selected_candidate=best[1],
        selected_regularization=best[2],
        identity_nmse=normalized_mean_squared_error(
            split.test[washout:],
            observations[2][washout:],
        ),
        nmse=normalized_mean_squared_error(
            split.test[washout:],
            prediction[washout:],
        ),
    )


def _gap_starts(
    signal_length: int,
    gap_length: int,
    *,
    minimum_history: int,
    n_repeats: int,
) -> tuple[int, ...]:
    latest = signal_length - gap_length
    if latest < minimum_history:
        raise ValueError("gap length leaves insufficient history")
    return tuple(
        int(value)
        for value in np.unique(
            np.linspace(minimum_history, latest, n_repeats, dtype=np.int64)
        )
    )


def _multiple_gap_corruption(
    clean: NDArray[np.float64],
    *,
    gap_length: int,
    washout: int,
    n_gaps: int,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    starts = _gap_starts(
        clean.size,
        gap_length,
        minimum_history=washout,
        n_repeats=n_gaps,
    )
    values = clean.copy()
    mask = np.ones(clean.shape, dtype=np.bool_)
    for start in starts:
        values[start : start + gap_length] = 0.0
        mask[start : start + gap_length] = False
    return values, mask


def run_reservoir_gap_experiment(
    candidates: Iterable[ReservoirCandidate],
    *,
    n_samples: int = 6_000,
    gap_lengths: Iterable[int] = (5, 10, 20, 40, 80),
    readout_regularizations: Iterable[float] = (1e-6, 1e-3, 0.1),
    washout: int = 200,
    n_repeats: int = 5,
) -> tuple[ReservoirGapResult, ...]:
    """Tune gap restorers on validation and report fixed-denominator NMSE."""
    candidate_values = tuple(candidates)
    if not candidate_values:
        raise ValueError("candidates must not be empty")
    gaps = tuple(int(length) for length in gap_lengths)
    if not gaps or any(length <= 0 for length in gaps):
        raise ValueError("gap_lengths must contain positive integers")
    penalties = _validated_penalties(readout_regularizations)
    if n_repeats <= 0:
        raise ValueError("n_repeats must be positive")

    split = chronological_split(generate_mackey_glass(n_samples))
    validation_variance = float(np.var(split.validation, ddof=0))
    test_variance = float(np.var(split.test, ddof=0))
    results: list[ReservoirGapResult] = []

    for gap_length in gaps:
        train_values, train_mask = _multiple_gap_corruption(
            split.train,
            gap_length=gap_length,
            washout=washout,
            n_gaps=max(8, n_repeats),
        )
        validation_starts = _gap_starts(
            split.validation.size,
            gap_length,
            minimum_history=washout,
            n_repeats=n_repeats,
        )

        best_score = np.inf
        best_name = ""
        best_penalty = 0.0
        best_candidate: ReservoirCandidate | None = None
        for candidate in candidate_values:
            reservoir = candidate.factory()
            train_states = reservoir.collect_states(
                make_reservoir_inputs(train_values, train_mask),
                reset=True,
            )
            for penalty in penalties:
                readout = fit_ridge_readout(
                    train_states[~train_mask],
                    split.train[~train_mask],
                    regularization=penalty,
                )
                scores: list[float] = []
                for start in validation_starts:
                    corruption = mask_interval(
                        split.validation,
                        start=start,
                        length=gap_length,
                    )
                    states = reservoir.collect_states(
                        make_reservoir_inputs(
                            corruption.values,
                            corruption.observation_mask,
                        ),
                        reset=True,
                    )
                    prediction = readout.predict(states)
                    gap_slice = slice(start, start + gap_length)
                    scores.append(
                        mean_squared_error(
                            split.validation[gap_slice],
                            prediction[gap_slice],
                        )
                        / validation_variance
                    )
                score = float(np.mean(scores))
                if (score, candidate.name, penalty) < (
                    best_score,
                    best_name or candidate.name,
                    best_penalty,
                ):
                    best_score = score
                    best_name = candidate.name
                    best_penalty = penalty
                    best_candidate = candidate

        if best_candidate is None:
            raise RuntimeError("reservoir selection failed")
        selected_reservoir = best_candidate.factory()
        selected_restorer = fit_reservoir_restorer(
            selected_reservoir,
            train_values,
            train_mask,
            split.train,
            regularization=best_penalty,
            washout=washout,
            training_selector=~train_mask,
        )
        test_scores: list[float] = []
        for start in _gap_starts(
            split.test.size,
            gap_length,
            minimum_history=washout,
            n_repeats=n_repeats,
        ):
            corruption = mask_interval(
                split.test,
                start=start,
                length=gap_length,
            )
            prediction = selected_restorer.predict(
                corruption.values,
                corruption.observation_mask,
            )
            gap_slice = slice(start, start + gap_length)
            test_scores.append(
                mean_squared_error(
                    split.test[gap_slice],
                    prediction[gap_slice],
                )
                / test_variance
            )
        results.append(
            ReservoirGapResult(
                gap_length=gap_length,
                selected_candidate=best_name,
                selected_regularization=best_penalty,
                nmse=float(np.mean(test_scores)),
            )
        )
    return tuple(results)


def run_photonic_robustness_experiment(
    base_config: PhotonicDelayConfig,
    impairment_cases: Iterable[tuple[str, HardwareImpairments]],
    *,
    n_samples: int = 6_000,
    noise_standard_deviation: float = 0.1,
    readout_regularization: float = 1e-3,
    washout: int = 200,
    seed: int = 42,
) -> tuple[RobustnessResult, ...]:
    """Train an ideal photonic readout and test it under non-ideal dynamics."""
    cases = tuple(impairment_cases)
    if not cases:
        raise ValueError("impairment_cases must not be empty")
    if len({label for label, _ in cases}) != len(cases):
        raise ValueError("impairment case labels must be unique")

    split = chronological_split(generate_mackey_glass(n_samples))
    train_observation = add_gaussian_noise(
        split.train,
        standard_deviation=noise_standard_deviation,
        seed=seed,
    ).values
    test_observation = add_gaussian_noise(
        split.test,
        standard_deviation=noise_standard_deviation,
        seed=seed + 1,
    ).values
    train_mask = np.ones(split.train.shape, dtype=np.bool_)
    test_mask = np.ones(split.test.shape, dtype=np.bool_)
    ideal = PhotonicDelayReservoir(base_config)
    ideal_states = ideal.collect_states(
        make_reservoir_inputs(train_observation, train_mask),
        reset=True,
    )
    readout = fit_ridge_readout(
        ideal_states[washout:],
        split.train[washout:],
        regularization=readout_regularization,
    )

    results: list[RobustnessResult] = []
    test_inputs = make_reservoir_inputs(test_observation, test_mask)
    for label, impairments in cases:
        impaired = PhotonicDelayReservoir(
            base_config,
            impairments=impairments,
        )
        prediction = readout.predict(
            impaired.collect_states(test_inputs, reset=True)
        )
        results.append(
            RobustnessResult(
                label=label,
                nmse=normalized_mean_squared_error(
                    split.test[washout:],
                    prediction[washout:],
                ),
            )
        )
    return tuple(results)
