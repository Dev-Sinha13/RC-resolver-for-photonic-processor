# Implemented architecture

All original V1 milestones are complete. The project is organized so both
reservoirs use the same readout, input representation, splits, selection rule,
and metrics.

```text
src/rc_photonics/
├── signals.py             Mackey–Glass generation
├── corruption.py          Gaussian noise, impulses, gaps, and masks
├── datasets.py            chronological train/validation/test splits
├── metrics.py             MSE and variance-normalized NMSE
├── baselines.py           causal identity, mean, median, and gap baselines
├── autoregression.py      memoryless and autoregressive ridge models
├── experiments.py         fixed baseline benchmarks
├── readout.py             shared ridge readout
├── esn.py                 digital echo-state network
├── model_evaluation.py    common reservoir fitting and selection
├── photonic_delay.py      virtual-node photonic delay reservoir
├── hardware.py            simulated physical impairments
├── sensor_data.py         UCI Air Quality loading and preprocessing
├── reporting.py           Markdown, CSV, confidence intervals, and SVG
└── reservoir_cli.py       reproducible experiment entry points
```

## Data flow

1. Generate or load a clean chronological signal.
2. Corrupt only the observations and add an observation-mask channel.
3. Transform `[value, mask]` into fixed reservoir states.
4. Discard the initial washout states.
5. Fit only a ridge-regression readout on training states.
6. Select the candidate and readout penalty on validation data.
7. Score the selected system once on held-out test data.

The reservoir weights are never trained. This keeps the comparison focused on
the temporal representation produced by the ESN or delay loop.

## Completed milestones

- **Shared readout:** centered ridge regression with an unregularized
  intercept and shape/finite-value validation.
- **Digital ESN:** seeded sparse recurrent matrix, spectral-radius scaling,
  leaky updates, state reset, washout, and causal collection.
- **Common evaluation:** validation-only selection for Gaussian, impulse,
  gap, external-sensor, and hardware-robustness experiments.
- **Photonic delay reservoir:** deterministic input mask, delayed feedback,
  time-multiplexed virtual nodes, leaky coupling, and `sin²` nonlinearity.
- **Hardware robustness:** independently controllable internal noise,
  attenuation, quantization, drift, and timing jitter.
- **Real sensor data:** official UCI Air Quality download, missing-marker
  handling, longest-contiguous-run selection, chronological splitting, and
  training-only standardization.
- **Reporting and automation:** console commands, scripts, Markdown tables,
  CSV/SVG figures, tests, and Python 3.11/3.12 CI.

## Sensible research extensions

V1 is complete as a software prototype. Follow-up research should focus on
repeated-seed confidence intervals, memory-capacity measurements, improved
closed-loop gap training, broader real datasets, and calibration to measured
photonic-device parameters. Those are scientific extensions rather than
missing plumbing.
