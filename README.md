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
- causal identity, moving-average, missing-value, current-sample regression,
  and autoregressive ridge baselines; and
- reproducible Gaussian-denoising and missing-gap experiments with
  validation-only hyperparameter selection.

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

Run the complete pre-reservoir benchmark suite with:

```powershell
.\.venv\Scripts\rc-photonics-baselines.exe
```

The source-checkout alternative is
`.\.venv\Scripts\python.exe scripts\run_baseline_experiments.py`.

The fixed benchmark rules are documented in
[`docs/experimental_protocol.md`](docs/experimental_protocol.md). The next
model milestone is a conventional digital echo-state network. The photonic
delay reservoir comes only after the ESN passes the same benchmark interface.

## Experimental rule

The validation split may be used for model and hyperparameter selection. The
test split must remain untouched until the final evaluation of an experiment.

## Repository checks

Pull requests run the complete test suite and benchmark entry point on Python
3.11 and 3.12. In GitHub's `main` branch protection settings, require the
`Python 3.11` and `Python 3.12` checks from the `Tests` workflow before merging
so repository auto-merge waits for successful verification.
