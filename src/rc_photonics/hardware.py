"""Simulated photonic noise, attenuation, quantization, and drift."""

from dataclasses import dataclass
from numbers import Integral

import numpy as np


@dataclass(frozen=True)
class HardwareImpairments:
    """Independent non-idealities applied inside photonic state updates."""

    internal_noise_std: float = 0.0
    feedback_attenuation: float = 0.0
    quantization_bits: int | None = None
    drift_std: float = 0.0
    timing_jitter_std: float = 0.0
    seed: int = 1_337

    def validate(self) -> None:
        for name, value in (
            ("internal_noise_std", self.internal_noise_std),
            ("drift_std", self.drift_std),
            ("timing_jitter_std", self.timing_jitter_std),
        ):
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if (
            not np.isfinite(self.feedback_attenuation)
            or not 0.0 <= self.feedback_attenuation < 1.0
        ):
            raise ValueError("feedback_attenuation must be in [0, 1)")
        if self.quantization_bits is not None:
            if (
                isinstance(self.quantization_bits, bool)
                or not isinstance(self.quantization_bits, Integral)
                or self.quantization_bits <= 0
            ):
                raise ValueError("quantization_bits must be a positive integer")


class HardwareImpairmentModel:
    """Stateful, seeded realization of a hardware-imperfection configuration."""

    def __init__(self, config: HardwareImpairments | None = None) -> None:
        self.config = config or HardwareImpairments()
        self.config.validate()
        self._rng = np.random.default_rng(self.config.seed)
        self._previous_drive = 0.0

    def reset(self) -> None:
        self._rng = np.random.default_rng(self.config.seed)
        self._previous_drive = 0.0

    def perturb_feedback(self, value: float) -> float:
        """Apply deterministic attenuation and seeded multiplicative drift."""
        drift = 0.0
        if self.config.drift_std > 0.0:
            drift = float(self._rng.normal(0.0, self.config.drift_std))
        return float(
            value
            * (1.0 - self.config.feedback_attenuation)
            * (1.0 + drift)
        )

    def perturb_drive(self, value: float) -> float:
        """Approximate timing jitter as a first-order drive-phase error."""
        perturbed = float(value)
        if self.config.timing_jitter_std > 0.0:
            slope = float(value) - self._previous_drive
            offset = float(self._rng.normal(0.0, self.config.timing_jitter_std))
            perturbed += offset * slope
        self._previous_drive = float(value)
        return perturbed

    def perturb_state(self, value: float) -> float:
        """Apply detector noise and optional uniform state quantization."""
        perturbed = float(value)
        if self.config.internal_noise_std > 0.0:
            perturbed += float(
                self._rng.normal(0.0, self.config.internal_noise_std)
            )
        perturbed = float(np.clip(perturbed, 0.0, 1.0))
        if self.config.quantization_bits is not None:
            levels = 2 ** int(self.config.quantization_bits) - 1
            perturbed = round(perturbed * levels) / levels
        return float(perturbed)
