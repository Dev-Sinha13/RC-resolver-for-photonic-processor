# V1 benchmark results

These are deterministic results from the fixed 6,000-sample Mackey–Glass
protocol. Lower NMSE is better. They are engineering verification results from
one seed, not confidence intervals or claims about physical hardware.

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
