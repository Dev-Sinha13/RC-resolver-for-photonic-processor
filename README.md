# RC Photonics

This project investigates causal restoration of noisy and missing nonlinear
time-series data using a simulated, time-multiplexed photonic delay reservoir.

The current foundation intentionally contains no reservoir implementation. It
establishes:

- reproducible Mackey–Glass signal generation;
- chronological train/validation/test splitting;
- Gaussian, impulse, and missing-interval corruption;
- explicit observation masks for missing samples; and
- variance-normalized restoration metrics;
- causal identity and moving-average baselines; and
- a reproducible Gaussian-denoising experiment with validation-only
  hyperparameter selection.

## Current milestone

Create a local environment and install the package from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
```

Then run the tests:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Run the Gaussian baseline experiment from Python with:

```python
from rc_photonics import run_gaussian_baseline_experiment

for result in run_gaussian_baseline_experiment():
    print(result)
```

The next milestone will add mask-aware missing-value baselines before
implementing a trainable autoregressive model and digital echo-state network.

## Experimental rule

The validation split may be used for model and hyperparameter selection. The
test split must remain untouched until the final evaluation of an experiment.
