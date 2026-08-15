# Pre-reservoir experimental protocol

This protocol is fixed before implementing either the digital echo-state
network or the photonic delay reservoir. Both reservoir models must use the
same data, splits, causal information, hyperparameter-selection rules, and
metrics as the classical baselines.

## Signal and partitions

- Generate one deterministic 6,000-sample Mackey–Glass trajectory after the
  configured transient washout.
- Preserve temporal order with contiguous 60% training, 20% validation, and
  20% test partitions.
- Fit model parameters on training data only.
- Select hyperparameters using validation data only.
- Evaluate the selected configuration once on the test partition.

## Causality rule

An estimate at time `t` may depend only on observations at or before `t`.
Autoregressive forecasts exclude the current observation and use samples
strictly before `t`. During a missing interval, predicted values are fed back
recursively without access to the clean target or future observations.

## Gaussian-denoising task

- Add independent, deterministic Gaussian noise draws to the train,
  validation, and test partitions.
- Evaluate standard deviations `0.02`, `0.05`, `0.1`, and `0.2`.
- Compare identity processing, current-sample ridge regression, causal moving
  average, and lagged ridge autoregression.
- Use the same evaluation start for every model, determined by the largest
  autoregressive lag candidate.
- Report variance-normalized test NMSE.

## Missing-interval task

- Mask contiguous gaps of `5`, `10`, `20`, `40`, and `80` samples.
- Evaluate multiple deterministic positions for each gap length.
- Compare last observation carried forward, mask-aware causal moving average,
  and recursive ridge autoregression.
- Average error only over missing samples.
- Normalize every gap MSE by the variance of the complete corresponding clean
  split. This fixed denominator permits comparison across gap lengths.

## Reservoir entry criteria

Reservoir implementation begins only after:

- all baseline and experiment tests pass;
- both benchmark runs are deterministic;
- validation choices and test scores are separated;
- all baselines pass explicit no-future-information tests; and
- the benchmark command runs from a clean installation.
