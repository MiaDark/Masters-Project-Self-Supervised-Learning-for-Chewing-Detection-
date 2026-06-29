# 0002 — Uncertainty via Monte Carlo Dropout

- **Status:** Accepted
- **Date:** _set when committed_

## Context

In a health-monitoring setting a confidently wrong detector is dangerous, and we
specifically need to know how reliability degrades as sensors are removed (the
energy/sensor-selection question). Standard deep classifiers output a score but
not a calibrated measure of predictive uncertainty.

## Decision

Quantify predictive uncertainty with **Monte Carlo Dropout**: keep dropout
active at inference and run multiple stochastic forward passes, using the spread
of predictions as the uncertainty estimate. Report both detection metrics and
calibration (e.g. reliability diagrams, expected calibration error) for each
sensor configuration.

## Consequences

- No change to the architecture or training procedure; modest extra inference
  cost (the forward-pass count `T` is a knob).
- Gives a principled lever for sensor selection: we can ask which sensor subsets
  preserve both accuracy *and* calibrated confidence.
- MC-Dropout estimates can be imperfectly calibrated, so a **deep-ensemble
  reference** should be run as a sanity check where feasible.

## Alternatives considered

- **Deep ensembles** — often better-calibrated but require training and storing
  several models; heavier for the per-configuration sweeps we need. Kept as a
  reference baseline rather than the primary method.
- **Single softmax confidence** — rejected; not a true uncertainty estimate.

## References

Gal & Ghahramani (2016), *Dropout as a Bayesian Approximation*; see the project
motivation document.