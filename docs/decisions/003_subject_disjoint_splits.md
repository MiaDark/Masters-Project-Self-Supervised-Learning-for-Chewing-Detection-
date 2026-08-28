# 0003 — Subject-disjoint splits

- **Status:** Accepted
- **Date:** _set when committed_

## Context

Windows from a single participant are highly correlated (same physiology, fit,
sensor placement, habits). If windows from one person appear in both training and
test sets, the model can memorise person-specific patterns and report inflated
performance that will not hold on new users. This is the most common way wearable
HAR results are silently over-optimistic.

## Decision

All train/validation/test splits are **subject-disjoint**: whole participants are
assigned to exactly one split, never individual windows. The held-out test set is
fixed by participant *before any modelling* to prevent leakage. Evaluation uses
subject-wise cross-validation where practical.


## Consequences

- Reported metrics reflect generalisation to *unseen people*, which is the claim
  that matters for deployment.
- Smaller effective per-split sizes and higher variance; mitigated by
  cross-validation and reporting confidence intervals.
- Splitting logic must operate at the participant level across all three
  datasets despite their differing folder structures.

## Alternatives considered

- **Random window-level splitting** — rejected; causes subject leakage and
  inflated scores.

## References

See the project research plan (cross-cutting practices) and the wearable-HAR
generalisation literature.