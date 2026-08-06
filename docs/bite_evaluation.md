# Bite-Level Evaluation — Unified Architecture

How postprocessing and bout-level evaluation work across **both** pipelines
(classical ML and SSL masked-reconstruction): why the logic used to be
duplicated five times, how it's consolidated now, and the one concrete
postprocessing improvement this round validated and shipped.

## Prerequisites

`compare_bites.py` (§2) is a **comparison step, not a pipeline** — it needs
each pipeline's own trained model/checkpoint to already exist. Before
running it:

1. **Classical ML's model must already be trained** (`docs/classical_ml.md`'s
   "Commands" section, through `train.py`) — `compare_bites.py` computes
   predictions FRESH from `checkpoints/train_ml/best_model.joblib` itself
   (via `compute_upscaled_recordings`, §5), so there's no separate
   prediction-generation step to run first.
2. **SSL masked-reconstruction's model must already be fine-tuned**
   (`docs/ssl_masked.md` §9.2's commands, or the 8-way ablation described
   in that same file's §4, through `finetune.py`) — specifically,
   `ssl/masked/evaluate_bouts.py` must have been run at least once, since
   that's what writes the per-timestep prediction CSVs to
   `checkpoints/predictions_masked/{run_name}/` that `compare_bites.py` reads
   (SSL inference is comparatively expensive — unlike classical ML's, it
   isn't recomputed fresh inside `compare_bites.py`).

Neither pipeline depends on the other, and neither depends on anything in
this document — you can run them in either order, or in parallel. Only
`compare_bites.py` has to come last.

## How to run

**Current, real commands (80/10/10 split, both pipelines' actual artefact
paths as of 2026-07-31)** — see `docs/dissertation_classical_ml_sections.md`
and `docs/ssl_masked.md` §9 for the full derivation of each step; this is
just the sequence, for direct reuse:

```bash
# 1. Classical ML, start to finish -- no separate "upscale predictions"
#    step; evaluate_bouts.py/plot_predictions.py/compare_bites.py all
#    compute predictions fresh from --model-path.
python -m chewing_ssl.classical_ml.segment \
  --interim-root data/interim_normalized_80_10_10 \
  --splits data/processed_80_10_10/splits.json \
  --out data/processed_80_10_10/ml_segments
python -m chewing_ssl.classical_ml.extract_features \
  --segments-dir data/processed_80_10_10/ml_segments \
  --out data/processed_80_10_10/ml_features
python -m chewing_ssl.classical_ml.train \
  --sweep-config configs/train_ml_rf06_only.yaml \
  --features-dir data/processed_80_10_10/ml_features \
  --out-dir checkpoints/train_ml_80_10_10
python -m chewing_ssl.classical_ml.evaluate_bouts \
  --model-path checkpoints/train_ml_80_10_10/best_model.joblib \
  --features-path data/processed_80_10_10/ml_features/test.parquet \
  --interim-root data/interim_normalized_80_10_10 \
  --run-name rf_4s_80_10_10 --model-type random_forest \
  --summary-csv checkpoints/classical_ml_summary_80_10_10.csv \
  --out-csv checkpoints/ml_bout_ledger_80_10_10.csv

# 2. SSL masked-reconstruction, start to finish
python -m chewing_ssl.ssl.masked.pretrain \
  --architecture bigru --pretext-task span \
  --lr 1e-3 --mask-span-ratio 0.4 --hidden-size 96 --num-layers 3 \
  --processed-root data/processed_80_10_10 \
  --checkpoint-dir checkpoints/pretrain_masked_80_10_10/span_bigru_w200 --device cuda
python -m chewing_ssl.ssl.masked.finetune \
  --checkpoint checkpoints/pretrain_masked_80_10_10/span_bigru_w200/bigru_span_best.pt \
  --processed-root data/processed_80_10_10 \
  --out-dir checkpoints/finetune_masked_80_10_10 --device cuda
python -m chewing_ssl.ssl.masked.evaluate_bouts \
  --checkpoint checkpoints/finetune_masked_80_10_10/span_bigru_w200_best.pt \
  --interim-root data/interim_normalized_80_10_10 \
  --processed-root data/processed_80_10_10 \
  --pred-dir checkpoints/predictions_masked_80_10_10/span_bigru_w200 \
  --summary-csv checkpoints/masked_ablation_summary_80_10_10.csv \
  --out-csv checkpoints/masked_bout_ledger_80_10_10.csv

# 3. Only now, the comparison -- classical ML computed fresh in-process;
#    SSL read back from step 2's saved per-timestep CSVs
python -m chewing_ssl.common.evaluation.compare_bites \
  --interim-root data/interim_normalized_80_10_10 \
  --classical-ml-model-path checkpoints/train_ml_80_10_10/best_model.joblib \
  --classical-ml-features-path data/processed_80_10_10/ml_features/test.parquet \
  --classical-ml-run-name rf_4s_80_10_10 \
  --ssl-pred-dir checkpoints/predictions_masked_80_10_10/span_bigru_w200 \
  --ssl-run-name span_bigru_w200_80_10_10 \
  --out-ledger checkpoints/bite_comparison_ledger_80_10_10.csv \
  --out-summary checkpoints/bite_comparison_summary_80_10_10.csv
```

Results from exactly this run: §2.1 below.

`postprocess.py` and `pipeline.py` are **never invoked directly** — they're
plain library modules (no `main()`), imported by `evaluate_bouts.py`,
`plot_predictions.py`, and `compare_bites.py`. Running any of those three
scripts already exercises the shared postprocessing/evaluation code; there
is no separate step for it. The same is true of
`classical_ml.upscale_predictions` since this round (§5) — it's a library
module now too, not a script.

## 1. The consolidated architecture

Two files hold the entire shared implementation:

**`common/evaluation/postprocess.py`** — signal-processing primitives (all
pre-existing) plus two new pieces:
- `smooth_probs(probs, fs, window_s)` — NaN-aware moving average (§4)
- `PostprocessConfig` — one dataclass holding every postprocessing parameter
  (hysteresis thresholds, gap-fill, duration bounds, envelope-split settings,
  smoothing window)
- `apply_full_postprocessing(probs, envelope, cfg)` — the full recipe
  (smoothing → hysteresis → gap-fill → duration/valley split →
  envelope-split → min-duration) as one call

**`common/evaluation/pipeline.py`** (new file): everything downstream of
postprocessing:
- `evaluate_recording(rec, cfg, ...)`: postprocess one recording, match
  bouts under **both** IoU and tolerance modes always
- `aggregate_dual_mode(results)`: sum tp/fp/fn across recordings, both modes
- `build_bite_ledger(...)` / `ledger_rows_for_recording(...)`: one row per
  GT/predicted bite, tagged detected/missed/false_alarm, with
  `onset_lag_s`/`offset_lag_s`/`duration_delta_s` for matched pairs
- `build_summary_row(...)` / `append_summary_row(...)`: the results-summary
  CSV row, shared schema
- `add_postprocess_args(parser)` / `add_match_args(parser, default_match_mode)`
  / `postprocess_config_from_args(args)`: shared CLI wiring, so every
  script's `--post-high`, `--envelope-prominence`, etc. are defined once,
  not five times

`common/evaluation/bouts.py` (bout extraction + matching) and the console
printing in `reporting.py` were already shared and needed no consolidation —
`reporting.py` gained one addition, `print_worst_offenders()`, generalized
from what used to be classical-ML-only code.

**`common/evaluation/plotting.py`** (new file) — the figure itself: `PALETTE`,
`shade_bout`, `gt_panel`/`pred_panel` (build a panel dict: bouts + which are
detected/missed/false-alarm), and `plot_bout_panels(t, nav_x, nav_y, panels,
suptitle, out_path)` — an N-panel stacked figure sharing one time axis. Both
`plot_predictions.py` scripts now call this (2 panels: GT + one pipeline's
prediction); `compare_bites.py`'s `--plot` (§2) calls it with 3 (GT + both
pipelines' predictions) — the same function either way, just a different
panel count.

**What each script still owns** (irreducibly pipeline-specific):

| Script | Pipeline-specific part |
|---|---|
| `classical_ml/evaluate_bouts.py` | Computing the pre-averaged (overlapping-window) probability trace fresh, via `compute_upscaled_recordings` (§5) |
| `ssl/masked/evaluate_bouts.py` | Loading the model checkpoint and running live sliding-window inference |
| `classical_ml/plot_predictions.py` / `ssl/masked/plot_predictions.py` | The data-loading half only — the figure itself is shared too now (`common/evaluation/plotting.py`, §2) |
| `compare_bites.py` | Nothing beyond the two loaders above — everything else is shared calls |

Every one of these was verified after refactoring to reproduce the exact
pre-refactor numbers on real data (for classical ML, checked directly against
`git show HEAD`'s original script output). All 130 tests still pass.

## 2. Bite-by-bite comparison (`compare_bites.py`)

```bash
python -m chewing_ssl.common.evaluation.compare_bites \
  --classical-ml-model-path checkpoints/train_ml/best_model.joblib \
  --classical-ml-run-name rf_4s \
  --ssl-run-name span_bigru_w200 \
  --out-ledger checkpoints/bite_comparison_ledger.csv \
  --out-summary checkpoints/bite_comparison_summary.csv \
  --plot --plot-out-dir checkpoints/plots/bite_comparison/all
```

Gets each pipeline's per-timestep probability trace (classical ML computed
FRESH from `--classical-ml-model-path`, §5; SSL read back from the
per-timestep CSVs `ssl/masked/evaluate_bouts.py` already saved to
`checkpoints/predictions_masked/{run_name}/`), applies the **identical**
shared postprocessing recipe to both, and produces:

- `checkpoints/bite_comparison_ledger.csv` — every GT and predicted bite from
  **both** pipelines in one table, so you can look at the same ground-truth
  bite next to what each pipeline predicted for it. Columns:
  `pipeline, run_name, dataset, participant, tag, segment, bout_type, status,
  start_s, end_s, duration_s, iou, matched_start_s, matched_end_s,
  onset_lag_s, offset_lag_s, duration_delta_s`.
- `checkpoints/bite_comparison_summary.csv` — one row per pipeline, both
  match modes.
- **`--plot`** (opt-in) — a 3-panel PNG per recording shared by both
  pipelines (ground truth / classical ML / SSL), via `plot_bout_panels`
  (§1) — the comparison neither single-pipeline `plot_predictions.py` can
  produce alone, since neither has access to the other pipeline's
  predictions. The GT panel shows a bout as "detected" if **either**
  pipeline matched it (a bout both pipelines missed is the only thing
  shown as "missed" there) — matching is done positionally between the two
  pipelines' `extract_bouts()` output, since both come from the same
  underlying ground-truth array for that recording.

One parameter is deliberately allowed to differ between the two pipelines
(`--smooth-window-s` for classical ML, `--ssl-smooth-window-s` for SSL) — see
§4 for why. Every other postprocessing parameter is identical for both.

### 2.1 Results, 80/10/10 split (real, verified 2026-07-31)

**Note on match mode**: `compare_bites.py` applies **one** match mode to
both pipelines for direct comparability (default: tolerance) — this
differs from each pipeline's own individual `evaluate_bouts.py` console
display default (classical ML: tolerance; SSL: IoU), so the "detected"
counts in this joint ledger will not numerically match the TP count shown
in either pipeline's own standalone report unless you pass the same
`--match-mode` both places. Both `bout_iou_*` and `bout_tolerance_*`
columns are always computed regardless, in `--out-summary`.

**Headline numbers** (`checkpoints/bite_comparison_summary_80_10_10.csv` —
computed by one script call, one shared postprocessing code path, so this
is the most directly comparable version of these numbers anywhere in the
project's docs):

| Pipeline | n GT | n Pred | IoU F1 | IoU median IoU | Tolerance F1 | Tolerance median IoU |
|---|---|---|---|---|---|---|
| Classical ML (`rf_4s_80_10_10`) | 524 | 556 | 0.744 | 0.622 | 0.509 | 0.698 |
| Masked SSL (`span_bigru_w200_80_10_10`) | 524 | 541 | **0.766** | **0.647** | **0.548** | **0.740** |

**SSL leads on both bout-level metrics under this split** — a reversal from
the original 60/20/20 comparison (§3 below), where classical ML led on
IoU-mode. See `docs/dissertation_classical_ml_sections.md` §5.7 for the
caveats on over-reading this (different test population, classical ML's
own postprocessing re-tuning not yet folded in, etc.) — this document
focuses on the *bite-by-bite* "where do they actually differ" question
instead, which the headline F1 numbers alone can't answer.

**Bite-outcome taxonomy, side by side**
(`checkpoints/bite_comparison_ledger_80_10_10.csv`,
`align_bouts_sequential`, match-mode-independent — see
`docs/dissertation_classical_ml_sections.md` §5.5 for definitions):

| Cluster label | Classical ML | Masked SSL |
|---|---|---|
| match | 172 bites (32.8%) | **269 bites (51.3%)** |
| merge | 38 bites (7.3%) | 48 bites (9.2%) |
| split | 29 bites (5.5%) | 44 bites (8.4%) |
| tangle | **243 bites (46.4%)** | 111 bites (21.2%) |
| missed | 42 bites (8.0%) | 52 bites (9.9%) |
| false_alarm (predicted bouts) | 30 | 22 |

**The two pipelines fail differently, not just "by different amounts."**
SSL converts far more bites into clean 1:1 matches (51.3% vs. 32.8%) and
has a much smaller tangle problem (21.2% vs. 46.4%) — but has a slightly
*higher* miss rate (9.9% vs. 8.0%) and more split events (8.4% vs. 5.5%).
Classical ML's dominant failure mode is dense, boundary-ambiguous bursts
(tangle); SSL's is closer to genuinely losing individual bites outright, or
over-segmenting a real bite into two predicted pieces.

**Per-bite agreement — computed directly from the joint ledger, not
previously reported anywhere**: pivoting the 524 ground-truth bites by
which pipeline(s) detected each one (tolerance mode, both pipelines):

| Outcome | Bites | % of GT |
|---|---|---|
| Both pipelines detected | 212 | 40.5% |
| Classical ML only | 63 | 12.0% |
| Masked SSL only | 80 | 15.3% |
| Neither pipeline detected | 169 | 32.3% |

**This is the single most useful number in this document for anyone
considering an ensemble or fallback strategy**: 143 bites (27.3% of the
total) are caught by exactly one pipeline, not both — meaningfully more
than either pipeline's own individual miss count alone would suggest,
since it shows the two models' errors are only partially overlapping, not
one being a strict subset of the other's mistakes. A simple "detected by
either" union would raise effective recall well above either individual
pipeline's own tolerance-mode recall (classical ML 0.525, SSL 0.557) — to
`(212+63+80)/524 = 0.677` — at the obvious cost of also unioning both
pipelines' false alarms, not analyzed here. **169 bites (32.3%) that
neither pipeline detects at all** is the honest floor on what postprocessing
tuning or an ensemble could ever recover without a better underlying
detector — worth keeping in mind before assuming any postprocessing change
(§3, or the sweeps in `docs/dissertation_classical_ml_sections.md` §4.6 /
`docs/ssl_masked.md` §8) can close that gap.

## 3. The postprocessing improvement: smoothing

**Diagnosis** (from the bite-by-bite ledger, before this change): SSL
produced more, shorter false-alarm bouts than classical ML (median 5.3s vs.
10.4s), while classical ML's errors ran longer but were fewer. The reason:
classical ML's probability trace is implicitly pre-smoothed by
`upscale_predictions.py` averaging overlapping windows; SSL's per-timestep
trace comes straight from the model with no equivalent smoothing, so it
flickers more under a fixed hysteresis threshold.

**Fix**: `smooth_probs` — a short moving average applied to `probs` before
hysteresis, now part of `PostprocessConfig` as `smooth_window_s`.

**Validated empirically**, not assumed — swept window sizes against real
recordings for both pipelines:

| `smooth_window_s` | SSL (`span_bigru_w200`) IoU-F1 | SSL tolerance-F1 | Classical ML (`rf_4s`) IoU-F1 |
|---|---|---|---|
| 0.0 (off) | 0.674 | 0.489 | **0.763** |
| 1.0 | 0.720 | 0.522 | 0.750 (worse) |
| **2.0** | **0.731** | **0.527** | 0.746 (worse) |
| 3.5 | 0.736 | 0.529 | — |
| 5.0 | 0.727 | 0.501 (declining) | — |

Confirms the diagnosis exactly: smoothing helps SSL substantially (mostly by
cutting false alarms: 1259→1036 predicted bouts) and mildly hurts classical
ML (already smoothed; more smoothing just blurs real signal). **Defaults set
accordingly — the one place the two pipelines' shared config deliberately
differs**: `smooth_window_s=0.0` for classical ML, `2.0` for SSL
(`ssl/masked/evaluate_bouts.py`/`plot_predictions.py` override the shared
default; `compare_bites.py` exposes both as separate flags).

**Numbers at the time this fix was validated, original 60/20/20 split**
(`checkpoints/masked_ablation_summary.csv`, `checkpoints/classical_ml_summary.csv`
— superseded by §2.1's 80/10/10 numbers above; kept here since this is
where the smoothing fix itself was originally derived and validated):

| | window test F1 | bout F1, IoU≥0.3 | bout F1, tolerance=5s |
|---|---|---|---|
| Classical ML, 4s (`rf_4s`) | 0.844 | **0.763** | 0.512 |
| Masked SSL, best combo (`span_bigru_w200`) | 0.799 | 0.731 | **0.527** |

The gap at the bout level narrowed substantially on IoU-mode (0.674→0.731,
now within 3 points of classical ML's 0.763) and **SSL now leads on
tolerance-mode F1** (0.527 vs. 0.512) — a genuine change in the comparison's
conclusion at that match mode, not just noise: driven by fewer, more
confident false alarms after smoothing, at a small recall cost. Classical ML
remained ahead on IoU-mode and on window-level F1 under this split.
**§2.1 above has the current, 80/10/10 numbers** — SSL now leads on both
bout-level metrics, not tolerance-mode alone; whether `smooth_window_s=2.0`
is still the optimal SSL value under the new split specifically has been
checked exploratorily (`docs/ssl_masked.md` §8's postprocessing sweep found
2.0 still the best value among {0.0, 1.0, 2.0, 3.0} when re-swept against
the 80/10/10 model), so this default has NOT gone stale, unlike some of the
other postprocessing parameters that sweep re-tuned.

## 4. Reproducibility: predictions are always computed fresh, never cached

Classical ML used to have a separate script, `upscale_predictions.py`,
run once and writing its output to `checkpoints/ml_predictions_upscaled/`
(one parquet per recording + `manifest.csv`). Every downstream script
(`evaluate_bouts.py`, `plot_predictions.py`, `compare_bites.py`) then read
that cached file instead of the model.

**This was a real reproducibility risk**: if you retrained or swapped the
model (`checkpoints/train_ml/best_model.joblib`) without remembering to
re-run `upscale_predictions.py`, every downstream script would silently keep
evaluating the OLD model's predictions — no error, no warning, just quietly
stale numbers. SSL's pipeline could never have this bug, since it always
runs inference fresh from whatever checkpoint it's pointed at.

**Fix**: `chewing_ssl.classical_ml.upscale_predictions` is now a library
module (no CLI, no `main()` — like `postprocess.py`/`pipeline.py`),
exposing `compute_upscaled_recordings(model_path, features_path,
interim_root, window_len, ...)`, which `evaluate_bouts.py`,
`plot_predictions.py`, and `compare_bites.py` all call directly, every run.
There is no cached prediction artifact left to go stale —
`checkpoints/ml_predictions_upscaled/` no longer exists (deleted, ~26MB).

What's still safely cached (and should be — it's data-derived, not
model-derived, so it can't go stale relative to the model): the feature
tables (`data/processed/ml_features/`) and the trained model itself
(`checkpoints/train_ml/best_model.joblib`). Only the *predictions* — the
thing that depends on which model you currently have — stopped being
cached.

Verified after the change: both the 4s and 2s configs reproduce their
pre-change numbers exactly (`bout_iou_f1`/`bout_tolerance_f1` unchanged to
the last decimal), and all 130 tests still pass.

## 5. What this doesn't cover

- Only `span_bigru_w200` has been re-evaluated with the new smoothing
  default. The other 7 masked-reconstruction combinations in
  `checkpoints/masked_ablation_summary.csv` still reflect `smooth_window_s=0.0`
  (the shared default) — re-running `ssl/masked/evaluate_bouts.py` for each
  would pick up `smooth_window_s=2.0` automatically (it's now the script's
  own default) and likely show a similar improvement, but this hasn't been
  done for all 8 yet.
- The other postprocessing ideas raised alongside smoothing (per-recording
  adaptive thresholds, envelope-prominence sensitivity to a recording's own
  noise floor) are not implemented — smoothing was the one validated and
  shipped this round.
- Classical ML's own sweep (`checkpoints/train_ml/sweep_results.csv`) has not
  been re-run against the new shared architecture — its numbers are
  unaffected by this round's changes (its `smooth_window_s` default is 0.0,
  a no-op, so nothing about its own pipeline changed).
- **§2.1's 80/10/10 comparison uses each pipeline's own default
  postprocessing recipe (§4.4 in `docs/dissertation_classical_ml_sections.md`),
  not either pipeline's independently re-tuned recipe** — classical ML's
  postprocessing sweep (`docs/dissertation_classical_ml_sections.md` §4.6)
  found a further tolerance-F1 improvement to 0.533, and SSL's own sweep
  (`docs/ssl_masked.md` §8) found tolerance-F1 up to 0.553 — neither has
  been adopted as the default nor re-run through `compare_bites.py`, so
  §2.1's head-to-head is not yet "each pipeline at its best."
- **The 8-way ablation and the 3-stage hyperparameter tuning pass have not
  been rerun under 80/10/10** for the SSL pipeline (`docs/ssl_masked.md`
  §11) — only the single already-identified best configuration was
  retrained. A different combination might now win under the larger
  training population; unknown.
- **The per-bite agreement analysis in §2.1 (both/only-classical-ML/only-SSL/
  neither) has not been extended to an actual ensemble implementation** —
  it identifies the *opportunity* (27.3% of bites caught by exactly one
  pipeline) but no combination rule (e.g. union of both pipelines'
  detections, or a learned meta-classifier) has been built or evaluated.
- **`SussexEatingUK/EMTEQRAP15`'s near-total SSL miss rate** (22 of 23
  bites missed, `docs/ssl_masked.md` §9.4) — checked directly against
  `checkpoints/ml_bout_ledger_80_10_10.csv`: classical ML detects 11 of the
  same 23 bites on this exact recording, missing 12. **This is a genuine
  SSL-specific weakness on this recording, not a shared "hard recording"
  both pipelines struggle with equally** — classical ML does roughly as
  well here as its own overall average, while SSL nearly completely fails.
  The underlying cause (recording-specific signal characteristics? a
  participant whose eating style the SSL encoder generalizes to poorly?)
  has not been investigated further.
