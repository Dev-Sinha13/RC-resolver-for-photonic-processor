"""Physics-based OOK transmitter, fibre, and direct-detection simulation."""

from dataclasses import dataclass
from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike, NDArray


SPEED_OF_LIGHT_M_S = 299_792_458.0


@dataclass(frozen=True)
class OpticalLinkConfig:
    """Parameters for a single-channel intensity-modulated optical link."""

    symbol_rate_gbaud: float = 10.0
    samples_per_symbol: int = 8
    fibre_length_km: float = 25.0
    attenuation_db_per_km: float = 0.2
    dispersion_ps_nm_km: float = 16.7
    nonlinear_coefficient_per_w_km: float = 1.3
    launch_power_dbm: float = 10.0
    wavelength_nm: float = 1550.0
    transmitter_bandwidth_ghz: float | None = 7.5
    receiver_bandwidth_ghz: float | None = 7.5
    detector_snr_db: float | None = 18.0
    timing_jitter_std_ui: float = 0.02
    ssfm_steps: int = 32
    guard_symbols: int = 64
    seed: int = 42

    def validate(self) -> None:
        if not np.isfinite(self.symbol_rate_gbaud) or self.symbol_rate_gbaud <= 0:
            raise ValueError("symbol_rate_gbaud must be finite and positive")
        for name, value in (
            ("samples_per_symbol", self.samples_per_symbol),
            ("ssfm_steps", self.ssfm_steps),
            ("guard_symbols", self.guard_symbols),
        ):
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise ValueError(f"{name} must be an integer")
        if self.samples_per_symbol < 2:
            raise ValueError("samples_per_symbol must be at least two")
        if self.ssfm_steps <= 0:
            raise ValueError("ssfm_steps must be positive")
        if self.guard_symbols < 1:
            raise ValueError("guard_symbols must be positive")
        for name, value in (
            ("fibre_length_km", self.fibre_length_km),
            ("attenuation_db_per_km", self.attenuation_db_per_km),
            ("nonlinear_coefficient_per_w_km", self.nonlinear_coefficient_per_w_km),
            ("timing_jitter_std_ui", self.timing_jitter_std_ui),
        ):
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not np.isfinite(self.dispersion_ps_nm_km):
            raise ValueError("dispersion_ps_nm_km must be finite")
        if not np.isfinite(self.launch_power_dbm):
            raise ValueError("launch_power_dbm must be finite")
        if not np.isfinite(self.wavelength_nm) or self.wavelength_nm <= 0.0:
            raise ValueError("wavelength_nm must be finite and positive")
        nyquist_ghz = (
            0.5 * self.symbol_rate_gbaud * self.samples_per_symbol
        )
        for name, value in (
            ("transmitter_bandwidth_ghz", self.transmitter_bandwidth_ghz),
            ("receiver_bandwidth_ghz", self.receiver_bandwidth_ghz),
        ):
            if value is not None and (
                not np.isfinite(value) or value <= 0.0 or value >= nyquist_ghz
            ):
                raise ValueError(
                    f"{name} must be positive and below the sampled Nyquist rate"
                )
        if self.detector_snr_db is not None and not np.isfinite(
            self.detector_snr_db
        ):
            raise ValueError("detector_snr_db must be finite or None")

    @property
    def symbol_period_s(self) -> float:
        return 1.0 / (self.symbol_rate_gbaud * 1e9)

    @property
    def sample_period_s(self) -> float:
        return self.symbol_period_s / self.samples_per_symbol

    @property
    def launch_power_w(self) -> float:
        return 1e-3 * 10.0 ** (self.launch_power_dbm / 10.0)


@dataclass(frozen=True)
class OpticalTransmission:
    """Paired transmitted bits and received optical/electrical waveforms."""

    bits: NDArray[np.int8]
    transmitted_power: NDArray[np.float64]
    received_power: NDArray[np.float64]
    detector_waveform: NDArray[np.float64]
    sampled_values: NDArray[np.float64]
    sample_positions_ui: NDArray[np.float64]


def generate_ook_bits(n_bits: int, *, seed: int) -> NDArray[np.int8]:
    """Generate a reproducible independent OOK bit sequence."""
    if isinstance(n_bits, bool) or not isinstance(n_bits, Integral) or n_bits <= 0:
        raise ValueError("n_bits must be a positive integer")
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=int(n_bits), dtype=np.int8)


def _validated_bits(bits: ArrayLike) -> NDArray[np.int8]:
    values = np.asarray(bits)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("bits must be a non-empty one-dimensional array")
    if not np.all((values == 0) | (values == 1)):
        raise ValueError("bits must contain only zero and one")
    return np.asarray(values, dtype=np.int8)


def _causal_low_pass(
    values: NDArray[np.complex128] | NDArray[np.float64],
    *,
    bandwidth_hz: float | None,
    sample_period_s: float,
) -> NDArray[np.complex128] | NDArray[np.float64]:
    if bandwidth_hz is None:
        return values.copy()
    coefficient = 1.0 - np.exp(-2.0 * np.pi * bandwidth_hz * sample_period_s)
    output = np.empty_like(values)
    output[0] = values[0]
    for index in range(1, values.size):
        output[index] = output[index - 1] + coefficient * (
            values[index] - output[index - 1]
        )
    return output


def _dispersion_beta2_s2_m(config: OpticalLinkConfig) -> float:
    dispersion_s_m2 = config.dispersion_ps_nm_km * 1e-6
    wavelength_m = config.wavelength_nm * 1e-9
    return float(
        -dispersion_s_m2
        * wavelength_m**2
        / (2.0 * np.pi * SPEED_OF_LIGHT_M_S)
    )


def _propagate_ssfm(
    field: NDArray[np.complex128],
    config: OpticalLinkConfig,
) -> NDArray[np.complex128]:
    length_m = config.fibre_length_km * 1_000.0
    if length_m == 0.0:
        return field.copy()
    step_m = length_m / config.ssfm_steps
    angular_frequency = 2.0 * np.pi * np.fft.fftfreq(
        field.size,
        d=config.sample_period_s,
    )
    beta2 = _dispersion_beta2_s2_m(config)
    attenuation_power_per_m = (
        config.attenuation_db_per_km * np.log(10.0) / 10.0 / 1_000.0
    )
    gamma_per_w_m = config.nonlinear_coefficient_per_w_km / 1_000.0
    half_step = np.exp(
        (
            -0.5 * attenuation_power_per_m
            + 0.5j * beta2 * angular_frequency**2
        )
        * (0.5 * step_m)
    )
    propagated = field.copy()
    for _ in range(config.ssfm_steps):
        propagated = np.fft.ifft(np.fft.fft(propagated) * half_step)
        if gamma_per_w_m > 0.0:
            propagated *= np.exp(
                1j * gamma_per_w_m * np.abs(propagated) ** 2 * step_m
            )
        propagated = np.fft.ifft(np.fft.fft(propagated) * half_step)
    return np.asarray(propagated, dtype=np.complex128)


def simulate_ook_link(
    bits: ArrayLike,
    config: OpticalLinkConfig | None = None,
) -> OpticalTransmission:
    """Transmit OOK bits through a nonlinear dispersive fibre and receiver."""
    parameters = config or OpticalLinkConfig()
    parameters.validate()
    target_bits = _validated_bits(bits)
    rng = np.random.default_rng(parameters.seed)
    guards = rng.integers(
        0,
        2,
        size=2 * parameters.guard_symbols,
        dtype=np.int8,
    )
    full_bits = np.concatenate(
        (
            guards[: parameters.guard_symbols],
            target_bits,
            guards[parameters.guard_symbols :],
        )
    )
    levels = np.repeat(full_bits.astype(np.float64), parameters.samples_per_symbol)
    field = np.asarray(levels, dtype=np.complex128)
    field = np.asarray(
        _causal_low_pass(
            field,
            bandwidth_hz=(
                None
                if parameters.transmitter_bandwidth_ghz is None
                else parameters.transmitter_bandwidth_ghz * 1e9
            ),
            sample_period_s=parameters.sample_period_s,
        ),
        dtype=np.complex128,
    )
    # Define launch power at the fibre input, after transmitter filtering.
    field *= np.sqrt(
        parameters.launch_power_w / float(np.mean(np.abs(field) ** 2))
    )
    transmitted_power_full = np.abs(field) ** 2
    received_field = _propagate_ssfm(field, parameters)
    received_power_full = np.abs(received_field) ** 2
    detector_full = received_power_full / parameters.launch_power_w
    detector_full = np.asarray(
        _causal_low_pass(
            detector_full,
            bandwidth_hz=(
                None
                if parameters.receiver_bandwidth_ghz is None
                else parameters.receiver_bandwidth_ghz * 1e9
            ),
            sample_period_s=parameters.sample_period_s,
        ),
        dtype=np.float64,
    )
    if parameters.detector_snr_db is not None:
        reference_mean_square = 2.0
        noise_std = np.sqrt(
            reference_mean_square / (10.0 ** (parameters.detector_snr_db / 10.0))
        )
        detector_full += rng.normal(0.0, noise_std, size=detector_full.size)

    start = parameters.guard_symbols * parameters.samples_per_symbol
    stop = start + target_bits.size * parameters.samples_per_symbol
    transmitted_power = transmitted_power_full[start:stop].copy()
    received_power = received_power_full[start:stop].copy()
    detector = detector_full[start:stop].copy()
    nominal_centres = (
        np.arange(target_bits.size, dtype=np.float64)
        * parameters.samples_per_symbol
        + 0.5 * (parameters.samples_per_symbol - 1)
    )
    jitter_samples = rng.normal(
        0.0,
        parameters.timing_jitter_std_ui * parameters.samples_per_symbol,
        size=target_bits.size,
    )
    sample_positions = np.clip(
        nominal_centres + jitter_samples,
        0.0,
        detector.size - 1.0,
    )
    sampled = np.interp(
        sample_positions,
        np.arange(detector.size, dtype=np.float64),
        detector,
    )
    return OpticalTransmission(
        bits=target_bits.copy(),
        transmitted_power=np.asarray(transmitted_power, dtype=np.float64),
        received_power=np.asarray(received_power, dtype=np.float64),
        detector_waveform=np.asarray(detector, dtype=np.float64),
        sampled_values=np.asarray(sampled, dtype=np.float64),
        sample_positions_ui=sample_positions / parameters.samples_per_symbol,
    )


def hard_decisions(scores: ArrayLike, *, threshold: float) -> NDArray[np.int8]:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("scores must be a non-empty finite one-dimensional array")
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")
    return np.asarray(values > threshold, dtype=np.int8)


def bit_error_rate(target_bits: ArrayLike, predicted_bits: ArrayLike) -> float:
    target = _validated_bits(target_bits)
    prediction = _validated_bits(predicted_bits)
    if prediction.shape != target.shape:
        raise ValueError("target_bits and predicted_bits must have the same shape")
    return float(np.mean(target != prediction))


def select_binary_threshold(target_bits: ArrayLike, scores: ArrayLike) -> float:
    """Select the exact deterministic BER-minimizing validation threshold."""
    target = _validated_bits(target_bits)
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or values.shape != target.shape:
        raise ValueError("scores must match target_bits")
    if not np.all(np.isfinite(values)):
        raise ValueError("scores must contain only finite values")
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    sorted_targets = target[order]
    errors = int(np.count_nonzero(sorted_targets == 0))
    best_errors = errors
    best_threshold = float(np.nextafter(sorted_values[0], -np.inf))
    index = 0
    while index < sorted_values.size:
        stop = index + 1
        while stop < sorted_values.size and sorted_values[stop] == sorted_values[index]:
            stop += 1
        group = sorted_targets[index:stop]
        errors += int(np.count_nonzero(group == 1))
        errors -= int(np.count_nonzero(group == 0))
        threshold = float(sorted_values[index])
        if errors < best_errors:
            best_errors = errors
            best_threshold = threshold
        index = stop
    return best_threshold
