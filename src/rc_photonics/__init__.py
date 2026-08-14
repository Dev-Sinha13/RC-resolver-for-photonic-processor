"""Tools for causal signal-restoration experiments."""

from rc_photonics.corruption import (
    CorruptionResult,
    add_gaussian_noise,
    add_impulse_noise,
    mask_interval,
)
from rc_photonics.datasets import ChronologicalSplit, chronological_split
from rc_photonics.signals import MackeyGlassParameters, generate_mackey_glass

__all__ = [
    "ChronologicalSplit",
    "CorruptionResult",
    "MackeyGlassParameters",
    "add_gaussian_noise",
    "add_impulse_noise",
    "chronological_split",
    "generate_mackey_glass",
    "mask_interval",
]
