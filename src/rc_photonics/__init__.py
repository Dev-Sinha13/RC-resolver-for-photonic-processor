"""Tools for causal signal-restoration experiments."""

from rc_photonics.baselines import causal_moving_average, identity_restoration
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
    run_gaussian_baseline_experiment,
)
from rc_photonics.signals import MackeyGlassParameters, generate_mackey_glass

__all__ = [
    "ChronologicalSplit",
    "CorruptionResult",
    "GaussianBaselineResult",
    "MackeyGlassParameters",
    "add_gaussian_noise",
    "add_impulse_noise",
    "causal_moving_average",
    "chronological_split",
    "generate_mackey_glass",
    "identity_restoration",
    "mask_interval",
    "mean_squared_error",
    "normalized_mean_squared_error",
    "run_gaussian_baseline_experiment",
]
