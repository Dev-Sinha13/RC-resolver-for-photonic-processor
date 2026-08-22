"""Tools for causal signal-restoration experiments."""

from rc_photonics.autoregression import (
    AutoregressiveRidge,
    CurrentSampleRidge,
    fit_autoregressive_ridge,
    fit_current_sample_ridge,
)
from rc_photonics.baselines import (
    causal_masked_moving_average,
    causal_median_filter,
    causal_moving_average,
    identity_restoration,
    last_observation_carried_forward,
)
from rc_photonics.corruption import (
    CorruptionResult,
    add_gaussian_noise,
    add_impulse_noise,
    mask_interval,
)
from rc_photonics.datasets import ChronologicalSplit, chronological_split
from rc_photonics.esn import ESNConfig, EchoStateNetwork
from rc_photonics.hardware import HardwareImpairments
from rc_photonics.metrics import (
    mean_squared_error,
    normalized_mean_squared_error,
)
from rc_photonics.experiments import (
    GaussianBaselineResult,
    ImpulseBaselineResult,
    MissingGapResult,
    run_gaussian_baseline_experiment,
    run_impulse_baseline_experiment,
    run_missing_gap_experiment,
)
from rc_photonics.model_evaluation import (
    ReservoirCandidate,
    ReservoirGapResult,
    ReservoirGaussianResult,
    ReservoirImpulseResult,
    RobustnessResult,
    default_esn_candidates,
    default_photonic_candidates,
    default_torch_esn_candidates,
    fit_reservoir_restorer,
    make_reservoir_inputs,
    run_photonic_robustness_experiment,
    run_reservoir_gap_experiment,
    run_reservoir_gaussian_experiment,
    run_reservoir_impulse_experiment,
    run_reservoir_on_split,
)
from rc_photonics.photonic_delay import (
    PhotonicDelayConfig,
    PhotonicDelayReservoir,
)
from rc_photonics.optical_channel import (
    OpticalLinkConfig,
    OpticalTransmission,
    bit_error_rate,
    generate_ook_bits,
    hard_decisions,
    select_binary_threshold,
    simulate_ook_link,
)
from rc_photonics.optical_experiment import (
    EqualizerScore,
    OpticalEqualizationResult,
    causal_tap_matrix,
    run_optical_equalization_experiment,
)
from rc_photonics.readout import RidgeReadout, fit_ridge_readout
from rc_photonics.sensor_data import (
    PreparedSensorData,
    SensorSeries,
    download_uci_air_quality,
    load_uci_air_quality,
    prepare_sensor_series,
)
from rc_photonics.signals import MackeyGlassParameters, generate_mackey_glass

__all__ = [
    "AutoregressiveRidge",
    "ChronologicalSplit",
    "CorruptionResult",
    "CurrentSampleRidge",
    "ESNConfig",
    "EchoStateNetwork",
    "GaussianBaselineResult",
    "HardwareImpairments",
    "ImpulseBaselineResult",
    "MackeyGlassParameters",
    "MissingGapResult",
    "PhotonicDelayConfig",
    "PhotonicDelayReservoir",
    "OpticalEqualizationResult",
    "OpticalLinkConfig",
    "OpticalTransmission",
    "EqualizerScore",
    "PreparedSensorData",
    "ReservoirCandidate",
    "ReservoirGapResult",
    "ReservoirGaussianResult",
    "ReservoirImpulseResult",
    "RidgeReadout",
    "RobustnessResult",
    "SensorSeries",
    "add_gaussian_noise",
    "add_impulse_noise",
    "causal_masked_moving_average",
    "causal_median_filter",
    "causal_moving_average",
    "chronological_split",
    "causal_tap_matrix",
    "default_esn_candidates",
    "default_photonic_candidates",
    "default_torch_esn_candidates",
    "download_uci_air_quality",
    "fit_autoregressive_ridge",
    "fit_current_sample_ridge",
    "fit_reservoir_restorer",
    "fit_ridge_readout",
    "generate_mackey_glass",
    "generate_ook_bits",
    "hard_decisions",
    "identity_restoration",
    "last_observation_carried_forward",
    "load_uci_air_quality",
    "make_reservoir_inputs",
    "mask_interval",
    "mean_squared_error",
    "normalized_mean_squared_error",
    "bit_error_rate",
    "prepare_sensor_series",
    "run_gaussian_baseline_experiment",
    "run_impulse_baseline_experiment",
    "run_missing_gap_experiment",
    "run_photonic_robustness_experiment",
    "run_reservoir_gap_experiment",
    "run_reservoir_gaussian_experiment",
    "run_reservoir_impulse_experiment",
    "run_reservoir_on_split",
    "run_optical_equalization_experiment",
    "select_binary_threshold",
    "simulate_ook_link",
]
