# 0004 — Add a TF-C-style contrastive pretext task

- **Status:** Accepted
- **Date:** 2026-07-15

## Context

Decision 001 rejected contrastive learning as *the* pretext task, specifically
because "positive-pair/augmentation choices whose semantic validity for facial
sensing is poorly understood." That concern was abstract at the time. Since
then: the dataset has grown to 5 datasets / ~223 participants / tens of
thousands of pretrain windows (`data/processed/index/pretrain_train.parquet`),
and a concrete, well-specified augmentation scheme is available (Time-Frequency
Fusion Contrasting / TF-C: jittering, scaling, permutation, masking in the time
domain; low-pass, phase-shift, remove/add-frequency in the frequency domain).
With real data volume and a specific, citable augmentation scheme in hand, the
original concern is now something to test empirically rather than avoid.

## Decision

Add TF-C-style contrastive learning as a **second** pretext task, combined
with — not replacing — the masked-reconstruction task from decision 001. Both
tasks train the same shared encoder; final loss is a weighted sum of the
masked-reconstruction MSE and a contrastive term.

Augmentations (`chewing_ssl.models.augmentation`) are applied per-batch, at
training time, on top of already-preprocessed, normalized windows — not a
preprocessing step (see `docs/preprocessing.md`). Parameters live in
`configs/augmentation.yaml` for empirical tuning per decision 001's original
"positive-pair choices ... poorly understood" concern: since it's now an open
empirical question, it needs to be easy to sweep, not hardcoded.

## Consequences

- Training loop needs a combined loss and a tunable weighting between the two
  tasks — not built yet; see `augmentation.py`'s module docstring for the
  concrete open question this leaves (the paper's kernel-PCA fusion step
  produces a flattened embedding, not a (C, T) sequence, so it doesn't
  directly fit the existing CNN/Bi-GRU/Transformer encoders without a
  decision on how to reconcile that shape).
- Needs a projection head on the encoder output for the contrastive
  embedding, separate from the reconstruction decoder.
- Adds pretraining compute (extra forward passes for the augmented views) and
  new hyperparameters (per-augmentation strength/probability, kernel-PCA
  settings) that need their own ablation.

## Alternatives considered

- Carries over decision 001's original alternatives (contrastive-only,
  supervised-only — both still rejected for the reasons stated there).
- Full replacement of masked reconstruction with contrastive-only — rejected;
  no reason to discard task 1, which is already implemented and validated
  across all three candidate encoder architectures.

## References

- [Decision 001](001_ssl_pretext_tasks.md) (original masked-reconstruction
  decision; contrastive learning rejection).
- Time-series representation learning via Time-Frequency Fusion Contrasting
  (Frontiers in Artificial Intelligence, 2024).
