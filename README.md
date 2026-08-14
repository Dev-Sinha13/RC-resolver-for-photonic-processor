# RC Photonics

This project investigates causal restoration of noisy and missing nonlinear
time-series data using a simulated, time-multiplexed photonic delay reservoir.

The first milestone intentionally contains no reservoir implementation. It
establishes the experimental foundation:

- reproducible Mackey–Glass signal generation;
- chronological train/validation/test splitting;
- Gaussian, impulse, and missing-interval corruption;
- explicit observation masks for missing samples; and
- unit tests for numerical and reproducibility guarantees.

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

The next milestone will add causal classical baselines before implementing a
digital echo-state network.

## Experimental rule

The validation split may be used for model and hyperparameter selection. The
test split must remain untouched until the final evaluation of an experiment.
