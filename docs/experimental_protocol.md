# Experimental protocol

This protocol is shared by the classical baselines, digital echo-state
network, and simulated photonic delay reservoir.

## Signal and partitions

- Generate one deterministic 6,000-sample Mackey–Glass trajectory after its
  transient washout.
- Preserve temporal order with contiguous 60% training, 20% validation, and
  20% test partitions.
- Fit parameters using training data only.
- Select candidates and ridge penalties using validation data only.
- Evaluate the selected candidate on the test partition.
- Use deterministic, independent corruption seeds for each partition.

The real-sensor experiment applies the same 60/20/20 chronological split and
computes standardization statistics from its training partition only.

The optical experiment instead generates independent seeded OOK bitstreams for
training, validation, and test. It normalizes the oversampled detector waveform
using training statistics only. A validation bitstream selects each decision
threshold; the held-out test bitstream is evaluated once.

## Causality rule

An estimate at time `t` may depend only on observations at or before `t`.
Autoregressive forecasts exclude the current observation. During a missing
interval, predictions are fed back recursively without access to the clean
target or future observations. Reservoir inputs contain two channels:
`[corrupted value, observation mask]`.

For optical equalization, the receiver has one detector-waveform input channel.
The default decision has one symbol of fixed latency: receiver state at symbol
`t + 1` estimates transmitted symbol `t`. Every equalizer uses current or past
receiver samples only. The raw receiver buffers symbol `t`'s original sample
for the same latency rather than incorrectly comparing symbol `t + 1` with
symbol `t`.

## Tasks

### Gaussian denoising

- Noise standard deviations: `0.02`, `0.05`, `0.1`, and `0.2`.
- Baselines: identity, causal moving average, current-sample regression, and
  lagged autoregression.
- Reservoir target: the clean value at the current time.

### Impulse denoising

- Impulse probabilities: `0.01`, `0.05`, `0.1`, and `0.2`.
- Baselines: identity, causal median filter, and autoregression.
- Impulse values and positions are independently seeded by partition.

### Missing-interval restoration

- Gap lengths: `5`, `10`, `20`, `40`, and `80` samples.
- Multiple deterministic positions are evaluated for each length.
- Baselines: last observation carried forward, mask-aware causal moving
  average, and recursive autoregression.
- Reservoir readouts are trained on artificial missing positions so the target
  cannot be solved by simply copying observed values.
- Error is averaged only over missing samples.

### Optical OOK equalization

- Modulation: on-off keying at 10 Gbaud and 8 samples per symbol.
- Fibre: 25 km, 0.2 dB/km attenuation, 16.7 ps/(nm·km) dispersion, and
  1.3 (W·km)⁻¹ Kerr coefficient.
- Propagation: 32-step symmetric split-step Fourier method with guard symbols.
- Receiver: direct detection, 7.5 GHz bandwidth, 18 dB detector SNR, and
  0.02-unit-interval timing jitter.
- Models: raw sampling, a causal 17-tap feed-forward equalizer, digital ESN,
  and simulated photonic delay reservoir.
- Primary metric: bit-error rate after selecting a scalar threshold on the
  validation bitstream. NMSE is a secondary readout diagnostic.
- Dataset size: no download is needed; the default seeded experiment generates
  4,000 training, 2,000 validation, and 4,000 test bits in memory.

## Reservoir selection

ESN candidates vary node count, spectral radius, leak rate, and input scaling.
Photonic candidates vary virtual-node count, feedback gain, leak rate, input
scaling, and phase bias. Both use the same ridge penalties. Complete fixed
candidate spaces and seeds are stored in [`../configs/`](../configs/).

Every candidate is reset before each state-collection pass. The initial state
washout is excluded from fitting and scoring. The exact same selected readout
is used when evaluating simulated photonic hardware impairment; it is not
retrained to hide degradation.

## Metrics

The reported normalized mean squared error is

```text
NMSE = mean((target - prediction)^2) / variance(clean reference)
```

For Gaussian and impulse experiments, every model uses the same evaluation
start and test-reference variance. For gaps, the numerator uses only missing
positions while the denominator remains the variance of the complete clean
split. This fixed denominator makes different gap lengths comparable.

Lower is better. An NMSE of zero is perfect; a value around one is comparable
to predicting the reference mean. NMSE is undefined for a constant reference,
which the implementation rejects.

## Reproducibility and interpretation

All matrices, corruption, positions, and impairments are seeded. The checked-in
results are deterministic single-seed benchmark values. The reporting module
supports confidence intervals, but publication-grade claims should rerun the
full experiment across multiple independent initialization and corruption
seeds.
