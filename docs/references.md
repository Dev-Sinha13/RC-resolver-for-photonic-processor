# Scientific references

The implementation follows the standard reservoir-computing separation
between fixed recurrent dynamics and a trained linear readout, and the
single-node delay-reservoir idea used to emulate many virtual nodes in time.

- H. Jaeger and H. Haas, “Harnessing Nonlinearity: Predicting Chaotic Systems
  and Saving Energy in Wireless Communication,” *Science* 304, 78–80 (2004).
  DOI: [10.1126/science.1091277](https://doi.org/10.1126/science.1091277)
- H. Jaeger et al., “Optimization and applications of echo state networks with
  leaky-integrator neurons,” *Neural Networks* 20, 335–352 (2007).
  DOI: [10.1016/j.neunet.2007.04.016](https://doi.org/10.1016/j.neunet.2007.04.016)
- L. Appeltant et al., “Information processing using a single dynamical node as
  complex system,” *Nature Communications* 2, 468 (2011).
  [Article](https://www.nature.com/articles/ncomms1476)
- Y. Paquot et al., “Optoelectronic Reservoir Computing,” *Scientific Reports*
  2, 287 (2012). [Article](https://www.nature.com/articles/srep00287)
- I. B. Yildiz, H. Jaeger, and S. J. Kiebel, “Re-visiting the echo state
  property,” *Neural Networks* 35, 1–9 (2012). This paper is a useful warning
  that spectral-radius heuristics alone do not prove the echo-state property.
  DOI: [10.1016/j.neunet.2012.07.005](https://doi.org/10.1016/j.neunet.2012.07.005)

The real-data experiment uses:

- S. De Vito, *Air Quality*, UCI Machine Learning Repository (2008), CC BY
  4.0. DOI: [10.24432/C59K5F](https://doi.org/10.24432/C59K5F)

These references motivate the architecture; they do not imply that this
phenomenological simulator reproduces any paper's physical apparatus or that
its dimensionless impairment settings correspond to measured device units.
