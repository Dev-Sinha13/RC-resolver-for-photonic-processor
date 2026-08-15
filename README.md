# RC Photonics

RC Photonics is a reproducible research prototype for causal restoration of
noisy and missing nonlinear time-series data. It compares classical filters,
autoregression, a digital echo-state network (ESN), and a simulated
time-multiplexed photonic delay reservoir under exactly the same chronological
evaluation protocol.

The implementation includes:

- deterministic Mackey–Glass signal generation and corruption;
- Gaussian-noise, impulse-noise, and missing-interval tasks;
- causal classical and autoregressive baselines;
- a seeded digital ESN and shared ridge readout;
- a virtual-node photonic delay reservoir with a `sin²` nonlinearity;
- simulated noise, attenuation, quantization, drift, and timing jitter;
- a real-data workflow using the UCI Air Quality dataset; and
- CSV/SVG result generation with a 74-test unit suite.

## Install and verify

From the repository root on Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The package has one runtime dependency: NumPy. Python 3.11 and 3.12 are tested
in continuous integration.

## Run the experiments

```powershell
.\.venv\Scripts\rc-photonics-baselines.exe
.\.venv\Scripts\rc-photonics-esn.exe
.\.venv\Scripts\rc-photonics-photonic.exe
.\.venv\Scripts\rc-photonics-robustness.exe
.\.venv\Scripts\rc-photonics-sensor.exe
.\.venv\Scripts\rc-photonics-figures.exe
```

The sensor command downloads the official UCI dataset on first use. Generated
data and results are ignored by Git. Equivalent source-checkout scripts are in
[`scripts/`](scripts/).

The photonic and figure workflows deliberately evaluate all fixed candidates
and can take several minutes. Candidate spaces are recorded in [`configs/`](configs/).

## What the current results say

On the fixed Mackey–Glass benchmark, the ESN beats autoregression at every
tested Gaussian noise level. It is also effective on sparse impulse noise, but
a causal median filter remains a very strong specialized baseline. For missing
intervals, autoregression is much better than either reservoir; the first
photonic design loses useful state during autonomous gaps.

On the UCI sensor run, the ESN slightly improves the corrupted-input NMSE
(`0.016038` to `0.014949`), while the photonic model makes it worse
(`0.022058`). These results demonstrate a working experimental system, not a
claim that the present photonic configuration is superior. Full deterministic
results and limitations are in [`docs/results.md`](docs/results.md).

## Scientific scope

This is a discrete-time, phenomenological delay-reservoir simulation. The
`sin²` response, feedback, attenuation, noise, quantization, drift, and timing
jitter are physically motivated, but the code is not an electromagnetic or
device-level Maxwell-equation simulator. Hardware claims require calibration
against a specific photonic platform and repeated-seed experiments.

The fixed evaluation rules are in
[`docs/experimental_protocol.md`](docs/experimental_protocol.md), and the
implemented architecture is summarized in
[`docs/implementation_roadmap.md`](docs/implementation_roadmap.md). Primary
reservoir-computing and dataset sources are collected in
[`docs/references.md`](docs/references.md).

## Evaluation rule

Training data fits model parameters, validation data selects candidates, and
the test split is used only for final scoring. Every estimate at time `t` may
depend only on observations at or before `t`.
