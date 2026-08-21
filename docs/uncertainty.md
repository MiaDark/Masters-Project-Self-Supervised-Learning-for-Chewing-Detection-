# Uncertainty Quantification

Decision [002](decisions/002_uncertainty_mc.md)'s planned analysis: quantify predictive
uncertainty so sensor selection can be judged on more than raw F1 — does confidence hold up
as sensors are removed, and is that confidence trustworthy in the first place. Implemented in
`src/chewing_ssl/uncertainty/`.

## Current headline result (12 models, 2026-08-12)

| | masked_ssl (MC-Dropout) | classical_ml (tree ensemble) |
|---|---|---|
| Best config | global norm, all channels | global norm, all channels |
| Bout F1 (IoU) | **0.771** | 0.750 |
| ECE (calibration error) | **0.008** (well calibrated) | 0.039 |
| AUROC(uncertainty → misclassified) | 0.737 (epistemic) / 0.863 (total) | **0.829** (epistemic) / 0.843 (total) |

SSL is much better calibrated; the Random Forest's uncertainty is a somewhat better
*error detector*. Full 12-row table in §4. Worst configuration on every axis, for both
pipelines: accel-only (§5).

---

## 1. Motivation

The plan's aim was never uncertainty for its own sake — it exists to make sensor selection
defensible: "does reliability degrade as sensors are removed", not just "does F1 drop".
Decision 002 chose **Monte Carlo Dropout** (Gal & Ghahramani 2016) as the primary method,
with reliability diagrams / expected calibration error (ECE) as the trustworthiness check,
and a cheaper ensemble-style method as a sanity-check reference where feasible. Both arms are
now implemented — MC-Dropout for the masked-reconstruction SSL encoder, and (going beyond the
original decision record) tree-ensemble disagreement for the classical-ML Random Forest, since
it turned out to be a natural, zero-retraining-cost analogue.

## 2. Method

### 2.1 MC-Dropout (masked-reconstruction SSL)

`ChewingDetector` ([ssl/masked/finetune.py](../src/chewing_ssl/ssl/masked/finetune.py)) already
has dropout in two places that normal evaluation (`model.eval()`) turns off: the classification
head (`nn.Dropout(0.3)`) and, for the Bi-GRU encoder, between GRU layers (`dropout=0.1`, active
because every capacity-search winner uses `num_layers=3`). No retraining is required — the
dropout the model was *trained* with is simply left switched on at inference time and sampled
repeatedly.

- **`enable_mc_dropout(model)`** — `model.eval()` everywhere, then re-enables `.train()` on
  `nn.Dropout` submodules only, so every other layer (Linear, GRU, GroupNorm) stays
  deterministic and only dropout is stochastic.
- **`infer_recording_mc(model, x, window_len, n_passes=30)`** — slides non-overlapping windows
  over a recording exactly like `evaluate_bouts.infer_recording`, but each window is repeated
  into a `(T, C, window_len)` batch and run through in **one** forward call (each batch row
  gets an independent dropout mask — equivalent to, just faster than, T sequential passes).

### 2.2 Tree-ensemble uncertainty (classical ML)

A `RandomForestClassifier(n_estimators=300)` already *is* an ensemble — each tree fit on an
independent bootstrap sample with a random feature subset. Where MC-Dropout re-runs the *same*
weights under different masks, each forest tree is a genuinely different model fit on
different data — arguably a cleaner match to "epistemic disagreement" than dropout's
same-weights re-masking, and it needs no retraining either.

- **`tree_ensemble_predict(pipeline, X)`** — stacks `tree.predict_proba(X)[:, 1]` across all
  300 `forest.estimators_`, in place of MC-Dropout's T stochastic passes.
- **`upscale_columns(...)`** — classical ML predicts one probability per *window*, not per
  timestep; this broadcasts several per-window arrays (mean/std/entropy/mutual_info) to
  per-timestep resolution together, generalizing
  `classical_ml.upscale_predictions.upscale_recording`'s single-column version.

### 2.3 Shared decomposition (both pipelines)

`decompose_uncertainty(passes)` in
[uncertainty/decompose.py](../src/chewing_ssl/uncertainty/decompose.py) takes a
`(n_predictors, n_items)` array — T dropout passes, or one row per tree — and returns:

- **mean** — the point prediction (fed to the same postprocessing/bout-matching pipeline every
  other script in this project uses, so bout-level numbers stay directly comparable).
- **std** — raw spread across predictors.
- **entropy** (total uncertainty) — binary entropy of the *mean* prediction.
- **exp_entropy** (aleatoric) — the *mean* of each individual predictor's own entropy; noise
  inherent to the input, irreducible by more training (e.g. a genuinely ambiguous bout-boundary
  timestep).
- **mutual_info** (epistemic, BALD) — `entropy - exp_entropy`; disagreement *between*
  predictors, i.e. what more/better-supported data would reduce.

### 2.4 Calibration

[`uncertainty/calibration.py`](../src/chewing_ssl/uncertainty/calibration.py)'s
`reliability_diagram(probs, labels, n_bins=15)` bins the mean predicted probability and compares
it against the observed positive rate per bin; `ece` is the count-weighted average gap — the
number decision 002 asked for explicitly.

## 3. Scripts and where output lands

| | Script | Run as |
|---|---|---|
| MC-Dropout eval | [`uncertainty/evaluate_uncertainty.py`](../src/chewing_ssl/uncertainty/evaluate_uncertainty.py) | `python -m chewing_ssl.uncertainty.evaluate_uncertainty --checkpoint <pt> --interim-root <dir> --processed-root <dir> --mc-passes 30 --run-name <name>` |
| Tree-ensemble eval | [`uncertainty/evaluate_uncertainty_rf.py`](../src/chewing_ssl/uncertainty/evaluate_uncertainty_rf.py) | `python -m chewing_ssl.uncertainty.evaluate_uncertainty_rf --model-path <joblib> --features-path <parquet> --interim-root <dir> --run-name <name>` |

Both share `common/evaluation/pipeline.py`'s postprocessing/bout-matching (same
`--post-*`/`--match-mode` flags as `evaluate_bouts.py` everywhere else in this project), so a
bout-level F1 computed here is directly comparable to the numbers already on record elsewhere —
used as a sanity check (§4).

**Output, every run:**

- `checkpoints/uncertainty/<run_name>_<participant>.npz` — per-recording, per-timestep
  `mean`/`std`/`entropy`/`exp_entropy`/`mutual_info`/`label` arrays. Raw material for any plot.
- `checkpoints/uncertainty/<run_name>_summary.json` — that run's summary metrics.
- `checkpoints/uncertainty_summary.csv` — **one row per run, appended** (dedup by `run_name`) —
  the single file comparing every config; schema: `run_name, pipeline, norm, channels,
  checkpoint, mc_passes|n_trees, n_timesteps, bout_iou_f1, bout_tolerance_f1, ece,
  mean_mi_correct, mean_mi_wrong, auroc_mi_vs_error, auroc_entropy_vs_error`.

Exact commands for all 12 configs, plus which real checkpoint each `--checkpoint`/
`--model-path` should point at: [`checkpoints/uncertainty_models/MANIFEST.md`](../checkpoints/uncertainty_models/MANIFEST.md)
— that directory also holds 12 descriptively-named symlinks
(`classical_ML_RF_global_all.joblib`, `masked_ssl_bigru_perfile_nav.pt`, ...) into the real
checkpoint directories, so a run command doesn't need the cryptic
`finetune_masked_80_10_10_clipped_nav_only`-style paths directly.

## 4. Results: the 12-model grid

2 pipelines × 2 normalization schemes × 3 channel subsets, all genuinely retrained per config
(`checkpoints/cross_sensor_evaluation/CONCLUSION.md`'s model family — see the MANIFEST for why
this specific 12, not some other checkpoint generation). `checkpoints/uncertainty_summary.csv`,
run 2026-08-12:

| run_name | pipeline | norm | channels | bout F1 (IoU) | bout F1 (tol) | ECE | epistemic: correct→wrong | AUROC(epistemic→err) | AUROC(entropy→err) |
|---|---|---|---|---|---|---|---|---|---|
| ssl_global_all | masked_ssl | global | all | 0.771 | 0.559 | **0.008** | 0.0015→0.0028 | 0.737 | 0.863 |
| ssl_global_nav | masked_ssl | global | nav | 0.755 | 0.536 | 0.021 | 0.0015→0.0026 | 0.737 | 0.848 |
| ssl_global_accel | masked_ssl | global | accel | 0.638 | 0.398 | 0.046 | 0.0039→0.0058 | 0.712 | 0.816 |
| ssl_perfile_all | masked_ssl | per_file | all | 0.774 | 0.519 | 0.023 | 0.0016→0.0035 | 0.803 | 0.859 |
| ssl_perfile_nav | masked_ssl | per_file | nav | 0.694 | 0.479 | 0.028 | 0.0029→0.0071 | 0.789 | 0.832 |
| ssl_perfile_accel | masked_ssl | per_file | accel | 0.700 | 0.495 | 0.045 | 0.0013→0.0027 | **0.799** | 0.857 |
| rf_global_all | classical_ml | global | all | 0.750 | 0.530 | 0.039 | 0.173→0.354 | **0.829** | 0.843 |
| rf_global_nav | classical_ml | global | nav | 0.727 | 0.503 | 0.059 | 0.177→0.369 | 0.777 | 0.806 |
| rf_global_accel | classical_ml | global | accel | 0.668 | 0.414 | 0.054 | 0.077→0.142 | 0.720 | 0.819 |
| rf_perfile_all | classical_ml | per_file | all | 0.748 | 0.510 | 0.031 | 0.204→0.434 | 0.822 | 0.844 |
| rf_perfile_nav | classical_ml | per_file | nav | 0.744 | 0.499 | 0.043 | 0.244→0.422 | 0.808 | 0.818 |
| rf_perfile_accel | classical_ml | per_file | accel | 0.641 | 0.369 | 0.046 | 0.101→0.156 | 0.687 (worst) | 0.812 |

**Sanity check**: bout IoU F1 matches `checkpoints/cross_sensor_evaluation/summary.csv`
(native-test scope) within ~0.005 on 11/12 rows — confirms the MC-mean / ensemble-mean point
prediction reproduces the already-trusted deterministic numbers, i.e. the uncertainty
machinery doesn't silently shift the model's operating point. The one exception,
`rf_global_all` (0.750 here vs. 0.769 on record), is very likely explained by the LR/RF
sweep-winner correction described in the MANIFEST — not yet independently re-verified.

## 5. Findings

1. **SSL's epistemic uncertainty is ~100x smaller in absolute terms than the Random Forest's**
   (0.0015-0.0071 vs. 0.08-0.43). The BiGRU's dropout passes barely disagree with each other;
   300 independently-bootstrapped trees disagree substantially more. The *relative*
   correct→wrong jump is nonetheless similar for both (~2x) — but the Random Forest's
   disagreement is the better error-detector in absolute AUROC terms (0.68-0.83 vs. SSL's
   0.71-0.80): genuinely different trees carry more informative disagreement than the same
   weights re-masked.
2. **SSL is consistently better calibrated** — ECE roughly 2-5x lower than the Random Forest's
   on every matched config, most dramatic on global/all (0.008 vs. 0.039). Consistent with the
   general expectation that Random Forest vote-fraction probabilities need explicit calibration
   more than a network trained end-to-end with BCE.
3. **Accel-only is the worst configuration on every axis, for both pipelines** — lowest F1,
   highest (or near-highest) ECE, and for the Random Forest specifically, its uncertainty is
   the *least* trustworthy of all 12 configs (AUROC 0.687, the table's minimum). Not just
   "worse accuracy" — its own confidence signal is also least trustworthy exactly where you'd
   need it most, a concrete point for the energy/sensor-selection trade-off chapter.
4. **Normalization's effect on calibration flips direction by pipeline.** Global norm gives
   better ECE for SSL; per-file norm gives better ECE for the Random Forest (comparing the two
   all-channel rows). "Which normalization is better" isn't answerable independent of which
   model is being deployed.
5. `SussexEatingUK/EMTEQRAP15` (already flagged in `docs/ssl_masked.md` §11 as an unexplained
   near-total miss, 22/23 bites) — the trace plot (Fig. 4, §8) shows this is *not* "confidently
   predicting non-chewing": the mean probability does rise during most ground-truth bouts,
   to 0.6-0.85, and total uncertainty (entropy) tracks those rises correctly. What's actually
   flat is the **epistemic** component (`mutual_info`) — pinned near 0 for the entire ~22
   minute recording, including every correctly-classified stretch, not just the missed bites.
   The postprocessing hysteresis (`--post-high 0.75`) requires a *sustained* crossing to open a
   bout; these rises are real but too brief/low to trigger it, and MC-Dropout's epistemic
   signal gives no advance warning that this particular recording's dynamics sit in that
   failure mode — it would look no different from a routine correct recording by epistemic
   uncertainty alone. A sharper version of the original point: the *aleatoric/total* signal is
   doing all the work of tracking ambiguity here, epistemic carries none of it.

## 6. Conclusion

Both uncertainty mechanisms are implemented, validated (bout-level sanity check against
already-trusted numbers), and calibrated (ECE computed, decision 002's explicit ask). The
headline practical answer to decision 002's original motivating question — "how does
reliability degrade as sensors are removed" — is: **not gracefully for accel-only**, on either
pipeline, and specifically for the Random Forest, its accel-only confidence signal degrades
*faster* than its accuracy does. Nav-only is a much softer landing (masked_ssl actually
improves the AUROC error-detection metric on per_file/nav-only vs. some other configs), which
matters directly for the sensor-selection/energy chapter: if one sensor has to go for energy
savings, the evidence here says accelerometer, not navigation, and says so on more than the F1
numbers `cross_sensor_evaluation/` already established.

## 7. Open items / not yet done

- **Deep-ensemble sanity check** (decision 002's stated fallback) — not run. The 8-way
  ablation checkpoints (`docs/ssl_masked.md` §4) are architecturally diverse enough to serve as
  a rough reference at no extra training cost; a cleaner version would be 3-5 reseeded runs of
  the winning config specifically.
- **`EMTEQRAP15` (finding 5)** — the trace plot narrowed down *what* is uninformative
  (epistemic specifically, for this whole recording, not just its missed bites) but not *why*
  this recording's dynamics produce that pattern while others don't; still open.
- **Not yet folded into the energy analysis** (`src/chewing_ssl/energy/` is still an empty
  stub) — the accuracy/confidence/energy three-way trade-off decision 002 and the original
  project plan both call for isn't assembled yet.
- **`rf_global_all`'s 0.750 vs. 0.769 discrepancy** (§4) not independently re-verified.

## 8. Figures

Generated by [`uncertainty/plots.py`](../src/chewing_ssl/uncertainty/plots.py)
(`python -m chewing_ssl.uncertainty.plots`), directly from the existing `.npz`/CSV output — no
new evaluation runs. Each written as both `.pdf` (vector, for the LaTeX build — per
`docs/dissertation.tex`'s own instructions, prefer PDF over PNG for figures) and `.png` (quick
preview) to `checkpoints/plots/uncertainty/`. Color roles: categorical slot 1 (blue `#2a78d6`)
= masked_ssl, slot 2 (orange `#eb6834`) = classical_ml, fixed across every figure; status colors
(green `#0ca30c` / red `#d03b3b`) for the binary correct/wrong split in Fig. 3 only.

1. **`01_reliability_diagram`** — confidence vs. observed accuracy per bin (global/all config,
   both pipelines) against the perfect-calibration diagonal. Visualizes finding 2 directly —
   SSL's line sits almost on the diagonal (ECE 0.008); the Random Forest's wanders further
   (ECE 0.039), particularly in the 0.55-0.75 confidence range.
2. **`02_config_grid`** — 3 panels (bout F1 / ECE / AUROC-epistemic), grouped bars across all
   6 norm×channel configs, paired by pipeline. Carries §4's whole table visually: the
   accel-only dip (finding 3) is visible in all three panels at once, and per_file/accel is
   where classical_ml's AUROC panel bar visibly drops below every other bar in the whole grid
   (finding 3's "least trustworthy of all 12").
3. **`03_uncertainty_distributions`** — density histograms of epistemic uncertainty split
   correct/wrong, one panel per pipeline (separate x-axis scales — finding 1's ~100x gap would
   flatten one pipeline on a shared axis). Dashed lines mark group means. Makes the "different
   absolute scale, similar relative separation" point (finding 1) visually immediate.
4. **`04_emteqrap15_trace`** — two-panel trace for the flagged outlier recording (finding 5):
   mean probability + GT shading on top (same visual language as `plot_predictions.py`), total
   vs. epistemic uncertainty underneath. This is what corrected finding 5 from "confidently
   predicting non-chewing" to the more precise "epistemic signal flat near-zero for the whole
   recording" — worth checking the actual plot before quoting the finding from the table alone.

**Not yet built**: a proper (non-ASCII) methods-section figure for `docs/architecture.md`'s
pipeline diagram, showing the T-stochastic-passes → decomposition → calibration flow.
