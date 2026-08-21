# 0001 — Self-supervised pretext tasks

- **Status:** Accepted
- **Date:** _set when committed_
- **Implementation status** (added retrospectively): masked signal
  reconstruction is fully implemented and is this project's primary
  result (`docs/ssl_masked.md`). **Cross-sensor prediction was never
  implemented** — tracked as an open item in `docs/ssl_masked.md` §12, not
  to be confused with the cross-sensor *ablation* study (§11 of that
  document), which evaluates the masked-reconstruction model under sensor
  removal rather than building the cross-sensor-prediction pretext task
  this decision specified. Contrastive learning, rejected below, was later
  reconsidered — see decision [004](004_contrastive_pretext_task.md).

## Context

Labelling chewing in continuous, free-living OCOsense recordings is expensive and
error-prone, and supervised deep models trained on small labelled sets tend not
to generalise beyond their training conditions. We need representations that can
be learned from the large pool of *unlabelled* sensor data (chiefly Spectacle)
and then fine-tuned on a small labelled set.

## Decision

Pre-train a multimodal encoder with **two combined pretext tasks**:

1. **Masked signal reconstruction** — hide random temporal spans *and* whole
   sensor channels, and reconstruct them. Time-only masking ignores
   inter-channel structure; masking along the sensor axis forces the encoder to
   learn the coupling between facial OMG channels and inertial motion that is
   characteristic of chewing.
2. **Cross-sensor prediction** — predict one modality from the others. This
   captures shared information across modalities and yields a data-driven measure
   of which sensors are redundant vs complementary, feeding directly into the
   energy-aware sensor-selection analysis.

## Consequences

- Reduces dependence on scarce labels; the largest gains are expected in the
  low-label regime, which is exactly our setting.
- Cross-sensor prediction error doubles as an early, label-free signal of sensor
  redundancy for the energy analysis.
- Adds pre-training cost and hyper-parameters (mask ratio, loss weighting) that
  must be ablated.

## Alternatives considered

- **Contrastive learning** — strong for inertial HAR but depends on
  positive-pair/augmentation choices whose semantic validity for facial sensing
  is poorly understood.
- **Supervised-only** — rejected as the baseline-to-beat, not the method.

## References

See the project motivation document and the SSL-for-HAR literature
(masked-reconstruction and channel-masking work; large-scale wearable SSL
studies).