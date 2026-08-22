"""Causal OOK equalization with classical, digital, and photonic models."""

from dataclasses import dataclass, replace
from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rc_photonics.esn import ESNConfig, EchoStateNetwork
from rc_photonics.metrics import normalized_mean_squared_error
from rc_photonics.optical_channel import (
    OpticalLinkConfig,
    OpticalTransmission,
    bit_error_rate,
    generate_ook_bits,
    hard_decisions,
    select_binary_threshold,
    simulate_ook_link,
)
from rc_photonics.photonic_delay import PhotonicDelayConfig, PhotonicDelayReservoir
from rc_photonics.readout import fit_ridge_readout


@dataclass(frozen=True)
class EqualizerScore:
    name: str
    bit_error_rate: float
    nmse: float
    threshold: float


@dataclass(frozen=True)
class OpticalEqualizationResult:
    config: OpticalLinkConfig
    raw: EqualizerScore
    feed_forward: EqualizerScore
    esn: EqualizerScore
    photonic: EqualizerScore
    test_bits: NDArray[np.int8]
    test_received_samples: NDArray[np.float64]
    test_predictions: dict[str, NDArray[np.float64]]
    test_transmission: OpticalTransmission


def causal_tap_matrix(values: ArrayLike, *, n_taps: int) -> NDArray[np.float64]:
    """Return current-and-past causal samples for a feed-forward equalizer."""
    observations = np.asarray(values, dtype=np.float64)
    if observations.ndim != 1 or observations.size == 0:
        raise ValueError("values must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(observations)):
        raise ValueError("values must contain only finite samples")
    if isinstance(n_taps, bool) or not isinstance(n_taps, Integral) or n_taps <= 0:
        raise ValueError("n_taps must be a positive integer")
    states = np.zeros((observations.size, int(n_taps)), dtype=np.float64)
    for lag in range(int(n_taps)):
        states[lag:, lag] = observations[: observations.size - lag]
    return states


def _aligned_raw(
    values: NDArray[np.float64],
    bits: NDArray[np.int8],
    *,
    delay: int,
    start: int,
) -> tuple[NDArray[np.float64], NDArray[np.int8]]:
    first_observation = max(start, delay)
    decision_indices = np.arange(first_observation, values.shape[0])
    target_indices = decision_indices - delay
    # A raw receiver can buffer its original symbol sample until the allowed
    # decision time. Latency must not silently compare symbol t + delay with t.
    return values[target_indices], bits[target_indices]


def _decision_rows(
    values: NDArray[np.float64],
    transmission: OpticalTransmission,
    *,
    delay: int,
    start: int,
) -> tuple[NDArray[np.float64], NDArray[np.int8]]:
    first_observation = max(start, delay)
    observation_symbols = np.arange(first_observation, transmission.bits.size)
    target_symbols = observation_symbols - delay
    samples_per_symbol = (
        transmission.detector_waveform.size // transmission.bits.size
    )
    waveform_indices = np.rint(
        transmission.sample_positions_ui[observation_symbols]
        * samples_per_symbol
    ).astype(np.int64)
    waveform_indices = np.clip(waveform_indices, 0, values.shape[0] - 1)
    return (
        values[waveform_indices],
        transmission.bits[target_symbols],
    )


def _score(
    name: str,
    validation_values: NDArray[np.float64],
    validation_bits: NDArray[np.int8],
    test_values: NDArray[np.float64],
    test_bits: NDArray[np.int8],
) -> EqualizerScore:
    threshold = select_binary_threshold(validation_bits, validation_values)
    decisions = hard_decisions(test_values, threshold=threshold)
    return EqualizerScore(
        name=name,
        bit_error_rate=bit_error_rate(test_bits, decisions),
        nmse=normalized_mean_squared_error(
            test_bits.astype(np.float64),
            test_values,
        ),
        threshold=threshold,
    )


def _waveform_inputs(
    values: NDArray[np.float64],
    *,
    input_dim: int,
) -> NDArray[np.float64]:
    """Put the scalar detector waveform on one channel and zero unused ones."""
    inputs = np.zeros((values.size, input_dim), dtype=np.float64)
    inputs[:, 0] = values
    return inputs


def run_optical_equalization_experiment(
    config: OpticalLinkConfig | None = None,
    *,
    n_train_bits: int = 4_000,
    n_validation_bits: int = 2_000,
    n_test_bits: int = 4_000,
    ffe_taps: int = 17,
    decision_delay_symbols: int = 1,
    washout: int = 200,
    regularization: float = 1e-3,
    esn_config: ESNConfig | None = None,
    photonic_config: PhotonicDelayConfig | None = None,
    seed: int = 4_200,
) -> OpticalEqualizationResult:
    """Run a leakage-free OOK equalization comparison on independent bits."""
    parameters = config or OpticalLinkConfig()
    parameters.validate()
    for name, value in (
        ("n_train_bits", n_train_bits),
        ("n_validation_bits", n_validation_bits),
        ("n_test_bits", n_test_bits),
        ("ffe_taps", ffe_taps),
        ("washout", washout),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if (
        isinstance(decision_delay_symbols, bool)
        or not isinstance(decision_delay_symbols, Integral)
        or decision_delay_symbols < 0
    ):
        raise ValueError("decision_delay_symbols must be a non-negative integer")
    minimum_size = max(washout, decision_delay_symbols) + 2
    if min(n_train_bits, n_validation_bits, n_test_bits) <= minimum_size:
        raise ValueError("every partition must leave samples after washout and delay")

    bit_partitions = (
        generate_ook_bits(n_train_bits, seed=seed),
        generate_ook_bits(n_validation_bits, seed=seed + 1),
        generate_ook_bits(n_test_bits, seed=seed + 2),
    )
    transmissions = tuple(
        simulate_ook_link(
            bits,
            replace(parameters, seed=parameters.seed + index),
        )
        for index, bits in enumerate(bit_partitions)
    )
    train_waveform, validation_waveform, test_waveform = (
        transmission.detector_waveform for transmission in transmissions
    )
    training_mean = float(np.mean(train_waveform))
    training_std = float(np.std(train_waveform))
    if training_std <= 0.0:
        raise ValueError("training receiver samples must have non-zero variance")
    normalized_waveforms = tuple(
        (waveform - training_mean) / training_std
        for waveform in (train_waveform, validation_waveform, test_waveform)
    )
    normalized_samples = tuple(
        (transmission.sampled_values - training_mean) / training_std
        for transmission in transmissions
    )
    train_bits, validation_bits, test_bits = bit_partitions
    evaluation_start = max(washout, ffe_taps - 1)
    delay = int(decision_delay_symbols)

    raw_train, raw_train_bits = _aligned_raw(
        normalized_samples[0],
        train_bits,
        delay=delay,
        start=evaluation_start,
    )
    raw_readout = fit_ridge_readout(
        raw_train.reshape(-1, 1),
        raw_train_bits.astype(np.float64),
        regularization=regularization,
    )
    raw_validation_samples, raw_validation_bits = _aligned_raw(
        normalized_samples[1],
        validation_bits,
        delay=delay,
        start=evaluation_start,
    )
    raw_test_samples, aligned_test_bits = _aligned_raw(
        normalized_samples[2],
        test_bits,
        delay=delay,
        start=evaluation_start,
    )
    raw_validation = raw_readout.predict(raw_validation_samples.reshape(-1, 1))
    raw_test = raw_readout.predict(raw_test_samples.reshape(-1, 1))
    raw_score = _score(
        "No temporal equalization",
        raw_validation,
        raw_validation_bits,
        raw_test,
        aligned_test_bits,
    )

    tap_states = tuple(
        causal_tap_matrix(values, n_taps=ffe_taps)
        for values in normalized_waveforms
    )
    ffe_train, ffe_train_bits = _decision_rows(
        tap_states[0],
        transmissions[0],
        delay=delay,
        start=evaluation_start,
    )
    ffe_readout = fit_ridge_readout(
        ffe_train,
        ffe_train_bits.astype(np.float64),
        regularization=regularization,
    )
    ffe_validation_states, ffe_validation_bits = _decision_rows(
        tap_states[1],
        transmissions[1],
        delay=delay,
        start=evaluation_start,
    )
    ffe_test_states, _ = _decision_rows(
        tap_states[2],
        transmissions[2],
        delay=delay,
        start=evaluation_start,
    )
    ffe_validation = ffe_readout.predict(ffe_validation_states)
    ffe_test = ffe_readout.predict(ffe_test_states)
    ffe_score = _score(
        "17-tap FFE" if ffe_taps == 17 else f"{ffe_taps}-tap FFE",
        ffe_validation,
        ffe_validation_bits,
        ffe_test,
        aligned_test_bits,
    )

    digital = EchoStateNetwork(
        esn_config
        or ESNConfig(
            n_nodes=100,
            input_dim=1,
            spectral_radius=0.95,
            leak_rate=0.2,
            input_scaling=0.5,
            seed=42,
        )
    )
    digital_states = tuple(
        digital.collect_states(
            _waveform_inputs(values, input_dim=digital.config.input_dim),
            reset=True,
        )
        for values in normalized_waveforms
    )
    esn_train, esn_train_bits = _decision_rows(
        digital_states[0],
        transmissions[0],
        delay=delay,
        start=evaluation_start,
    )
    esn_readout = fit_ridge_readout(
        esn_train,
        esn_train_bits.astype(np.float64),
        regularization=regularization,
    )
    esn_validation_states, esn_validation_bits = _decision_rows(
        digital_states[1],
        transmissions[1],
        delay=delay,
        start=evaluation_start,
    )
    esn_test_states, _ = _decision_rows(
        digital_states[2],
        transmissions[2],
        delay=delay,
        start=evaluation_start,
    )
    esn_validation = esn_readout.predict(esn_validation_states)
    esn_test = esn_readout.predict(esn_test_states)
    esn_score = _score(
        "Digital ESN",
        esn_validation,
        esn_validation_bits,
        esn_test,
        aligned_test_bits,
    )

    photonic = PhotonicDelayReservoir(
        photonic_config
        or PhotonicDelayConfig(
            n_virtual_nodes=100,
            input_dim=1,
            feedback_gain=0.5,
            leak_rate=0.5,
            input_scaling=0.25,
            phase_bias=np.pi / 4.0,
            seed=42,
        )
    )
    photonic_states = tuple(
        photonic.collect_states(
            _waveform_inputs(values, input_dim=photonic.config.input_dim),
            reset=True,
        )
        for values in normalized_waveforms
    )
    photonic_train, photonic_train_bits = _decision_rows(
        photonic_states[0],
        transmissions[0],
        delay=delay,
        start=evaluation_start,
    )
    photonic_readout = fit_ridge_readout(
        photonic_train,
        photonic_train_bits.astype(np.float64),
        regularization=regularization,
    )
    photonic_validation_states, photonic_validation_bits = _decision_rows(
        photonic_states[1],
        transmissions[1],
        delay=delay,
        start=evaluation_start,
    )
    photonic_test_states, _ = _decision_rows(
        photonic_states[2],
        transmissions[2],
        delay=delay,
        start=evaluation_start,
    )
    photonic_validation = photonic_readout.predict(photonic_validation_states)
    photonic_test = photonic_readout.predict(photonic_test_states)
    photonic_score = _score(
        "Photonic delay reservoir",
        photonic_validation,
        photonic_validation_bits,
        photonic_test,
        aligned_test_bits,
    )

    return OpticalEqualizationResult(
        config=parameters,
        raw=raw_score,
        feed_forward=ffe_score,
        esn=esn_score,
        photonic=photonic_score,
        test_bits=aligned_test_bits.copy(),
        test_received_samples=raw_test.copy(),
        test_predictions={
            "FFE": ffe_test.copy(),
            "ESN": esn_test.copy(),
            "Photonic": photonic_test.copy(),
        },
        test_transmission=transmissions[2],
    )
