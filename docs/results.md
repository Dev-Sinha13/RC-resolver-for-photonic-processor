# Project benchmark results

These are deterministic engineering-verification results, not confidence
intervals or claims about physical hardware. The signal-restoration tables use
the fixed 6,000-sample Mackey–Glass protocol; the optical section uses its own
independent-bit communication protocol. Lower NMSE and BER are better.

## Optical OOK equalization

This separate communication experiment generated independent seeded OOK
bitstreams and propagated them at 10 Gbaud through 25 km of simulated fibre.
The channel included 0.2 dB/km attenuation, 16.7 ps/(nm·km) chromatic
dispersion, 1.3 (W·km)⁻¹ Kerr nonlinearity, 7.5 GHz transmitter and receiver
bandwidth, 18 dB detector SNR, and 0.02 UI sampling jitter. A 32-step symmetric
split-step Fourier solver operated at eight samples per symbol.

Training, validation, and test contained 4,000, 2,000, and 4,000 independently
generated bits. After the fixed washout and one-symbol causal receiver latency,
3,800 test decisions remained. Thresholds were fitted on validation data only.

| Receiver | Test BER | Bit errors | Test NMSE |
| --- | ---: | ---: | ---: |
| No temporal equalization | 0.033684 | 128 | 0.233769 |
| 17-tap FFE | **0.000263** | **1** | 0.059293 |
| Digital ESN | **0.000263** | **1** | **0.028519** |
| Photonic delay reservoir | 0.008684 | 33 | 0.144666 |

The photonic delay reservoir removed 95 of the raw receiver's 128 bit errors,
a 74.2% BER reduction. It remained behind the linear FFE and digital ESN, so
the experiment supports feasibility but not superiority. The FFE's especially
strong BER also shows that this fixed link is dominated by impairments that a
short linear-memory receiver can largely reverse. More nonlinear regimes,
repeated seeds, parameter selection performed strictly on validation, and
comparison with measured waveforms are required before making broader claims.

The full run used seed `2026` and completed in 58.14 seconds on the development
CPU. The raw receiver includes a training-fitted scalar amplitude calibration
and validation-fitted threshold, but no temporal taps. Generated CSV and SVG
files are deliberately ignored by Git and can be reproduced with:

```powershell
.\.venv\Scripts\rc-photonics-optical.exe --seed 2026
```

## PyTorch backend verification

The optional PyTorch ESN uses the exact seeded matrices from the NumPy
reference and produced identical six-decimal NMSE values and candidate choices
for every Gaussian, impulse, and missing-gap condition below. Focused tests
also verified state parity, causal behavior, reset behavior, input gradients,
registered non-trainable buffers, and agreement between the NumPy and PyTorch
closed-form ridge readouts.

On the development CPU, the complete benchmark took 119.532 seconds with
PyTorch 2.13.0 CPU and 54.525 seconds with NumPy. This small sequential workload
therefore does not benefit from PyTorch on CPU. The backend is intended for
future batched optical waveforms, GPU execution, and differentiable
hardware-aware experiments rather than as a claim of immediate acceleration.

A focused 6,000-sample, 100-node state-generation benchmark measured median
times of 0.489471 seconds for NumPy and 1.136686 seconds for PyTorch CPU, a
2.322× ratio. The maximum absolute difference between the two state matrices
was `5.551e-16`, which is at floating-point rounding scale.

## Gaussian denoising

| Noise σ | Identity | Moving average | Autoregression | ESN | Photonic |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.02 | 0.006865 | 0.002000 | 0.000926 | 0.000707 | 0.001739 |
| 0.05 | 0.042066 | 0.006883 | 0.004813 | 0.004446 | 0.006861 |
| 0.10 | 0.174032 | 0.024403 | 0.014334 | 0.011720 | 0.025178 |
| 0.20 | 0.706382 | 0.058996 | 0.037812 | 0.032127 | 0.092020 |

The ESN improves on autoregression at every tested level. The photonic model
does remove noise, but it trails the ESN and is close to a causal moving
average in this V1 configuration.

## Missing intervals

| Gap samples | Carried forward | Autoregression | ESN | Photonic |
| ---: | ---: | ---: | ---: | ---: |
| 5 | 0.002935 | <0.000001 | 0.002461 | 0.259413 |
| 10 | 0.009604 | <0.000001 | 0.012945 | 1.699307 |
| 20 | 0.030931 | 0.000005 | 0.055112 | 1.684836 |
| 40 | 0.081846 | 0.000157 | 0.238981 | 1.552219 |
| 80 | 0.211857 | 0.019200 | 0.542724 | 1.294124 |

The smooth, deterministic Mackey–Glass trajectory strongly favors a fitted
autoregressive model. The ESN degrades with gap length, and the photonic delay
reservoir rapidly loses an informative autonomous state. This is the clearest
limitation discovered by the benchmark.

## Impulse denoising

| Impulse probability | Identity | Causal median | Autoregression | ESN | Photonic |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.01 | 0.049598 | 0.000200 | 0.005497 | 0.000392 | 0.002568 |
| 0.05 | 0.216426 | 0.003247 | 0.021599 | 0.001666 | 0.010880 |
| 0.10 | 0.459906 | 0.003374 | 0.028333 | 0.006162 | 0.049374 |
| 0.20 | 0.955882 | 0.026402 | 0.044492 | 0.025083 | 0.106150 |

The ESN is broadly effective and beats autoregression, while the specialized
median filter is best at three of four corruption levels. The photonic model
helps at low impulse rates but degrades faster.

## Photonic robustness

| Simulated condition | Test NMSE |
| --- | ---: |
| Ideal | 0.037850 |
| Internal noise 0.005 | 0.094037 |
| Internal noise 0.02 | 0.874763 |
| Feedback attenuation 0.1 | 0.121102 |
| Feedback attenuation 0.4 | 1.611486 |
| Quantization 8 bit | 0.041055 |
| Quantization 4 bit | 0.836562 |
| Drift 0.01 | 0.058765 |
| Timing jitter 0.01 | 0.317880 |

The simulated loop is relatively tolerant of 8-bit quantization and modest
drift, but sensitive to strong attenuation, internal noise, low-bit
quantization, and timing jitter. These dimensionless impairment values are
comparative simulator settings; they are not calibrated laboratory units.

## UCI Air Quality sensor

The loader selected the longest valid hourly run of the `PT08.S1(CO)` sensor:
1,778 samples. Gaussian corruption and the fixed chronological protocol gave:

| Model | Corrupted identity NMSE | Restored NMSE |
| --- | ---: | ---: |
| ESN | 0.016038 | 0.014949 |
| Photonic | 0.016038 | 0.022058 |

The ESN gives a small improvement. The photonic model does not generalize as
well in its present form. The dataset source, missing-data rule, units, and
license are documented in [`../data/README.md`](../data/README.md).

## Main conclusion

The project successfully tests the hypothesis rather than assuming it. Fixed
reservoir dynamics plus a trained linear readout can restore temporally
structured data, and the digital ESN is useful for denoising. The current
photonic-delay approximation is not yet competitive, especially when it must
operate autonomously through missing intervals. Improving that memory behavior
is the most valuable next research direction.
