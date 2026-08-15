"""Tools for causal signal-restoration experiments."""

from rc_photonics.autoregression import (
    AutoregressiveRidge,
    CurrentSampleRidge,
    fit_autoregressive_ridge,
    fit_current_sample_ridge,
)
from rc_photonics.baselines import (
    causal_masked_moving_average,
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
from rc_photonics.metrics import (
    mean_squared_error,
    normalized_mean_squared_error,
)
from rc_photonics.experiments import (
    GaussianBaselineResult,
    MissingGapResult,
    run_gaussian_baseline_experiment,
    run_missing_gap_experiment,
)
from rc_photonics.signals import MackeyGlassParameters, generate_mackey_glass

__all__ = [
    "AutoregressiveRidge",
    "ChronologicalSplit",
    "CorruptionResult",
    "CurrentSampleRidge",
    "GaussianBaselineResult",
    "MackeyGlassParameters",
    "MissingGapResult",
    "add_gaussian_noise",
    "add_impulse_noise",
    "causal_masked_moving_average",
    "causal_moving_average",
    "chronological_split",
    "fit_autoregressive_ridge",
    "fit_current_sample_ridge",
    "generate_mackey_glass",
    "identity_restoration",
    "last_observation_carried_forward",
    "mask_interval",
    "mean_squared_error",
    "normalized_mean_squared_error",
    "run_gaussian_baseline_experiment",
    "run_missing_gap_experiment",
]
