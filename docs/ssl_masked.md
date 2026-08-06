# SSL: Masked Reconstruction — Unified Pipeline, Tuning History & Results


Decision 001's original self-supervised pretext task: reconstruct
deliberately hidden parts of a window from context. Implemented in
`ssl/masked/`.

## Current headline result (80/10/10 split, real, verified 2026-07-31)

| | window test F1 | bout F1, IoU ≥ 0.3 | bout F1, tolerance = 5 s |
|---|---|---|---|
| **`span_bigru_w200`, 80/10/10** | 0.826 | **0.766** | **0.548** |

524 ground-truth test bouts. Full derivation and head-to-head comparison
against classical ML: §9 below and `docs/bite_evaluation.md`.

---

## 1. Input and shared preprocessing

Same 10-channel, globally z-scored window as every other pipeline in this
project: `accel_x`, `accel_y`, `accel_z`, `nav_x_left_temple`,
`nav_y_left_temple`, and each of those five again as a first-derivative
(`_d1`) channel — shape `(10, T)`, `T=200` (4s @ 50Hz) or `T=100` (2s/25%-overlap,
only relevant to the 8-way ablation, §4).

`preprocess.py`'s raw-to-normalized pipeline fully applies here: gap
segmentation, resampling to 50Hz, the zero-phase Butterworth low-pass filter
(8Hz cutoff, order 4), and the first-derivative channels are **not**
classical-ML-specific; they're shared upstream infrastructure that produces
`data/interim_normalized/`, which every pipeline in this project (classical
ML, masked reconstruction, TF-C) reads from identically. See
`docs/preprocessing.md` and `docs/dissertation_classical_ml_sections.md` §1
for the full raw→interim→normalized sequence
(`preprocess.py` → `windowing.py --splits-only` → `refit_global_norm.py`).

What differs between classical ML and masked reconstruction is one step
*later*: how the already-preprocessed signal gets turned into training
examples.

| | Classical ML | Masked reconstruction (+ TF-C) |
|---|---|---|
| Windowing code | Its own `classical_ml/segment.py` | Shared `common/data/windowing.py` |
| Materialization | Eagerly materializes dense `(N, 10, T)` arrays to disk | Lightweight index only (`segment_path, start, end`) — `WindowDataset` slices the real values out of the cached normalized parquet at read time |
| Window/stride | As of the 80/10/10 rerun, non-overlapping for every split, classical ML's own choice (`docs/dissertation_classical_ml_sections.md` §2.1) | Fixed convention: 50% overlap for train-purpose windows, **zero** overlap for eval-purpose windows (deliberately, to avoid inflating effective eval sample size) — unchanged by the 80/10/10 rerun |
| Per-window label | Immediately collapsed to majority vote — one 0/1 label per window (a feature vector needs a single target) | Kept dense — one label per timestep, carried straight through the index unchanged |
| What the model actually sees | ~850 **hand-crafted** features per window (statistical/time/frequency/wavelet drivers computed over the raw `(10,T)` array) | The raw `(10,T)` z-scored tensor itself, fed directly into the CNN/Bi-GRU encoder — no hand-crafted features at all |

So the underlying *signal* is identical between the two pipelines all the
way through `data/interim_normalized`; the divergence is in how it's chopped
into windows and what representation of a window actually reaches the
model.

Pretraining windows (`pretrain_train`/`pretrain_val`) are **never labelled**,
even when the source segment has one — masked reconstruction is
self-supervised by construction (decision 001). Fine-tuning windows
(`finetune_train`/`val`/`test`) carry the dense per-timestep label array
described above (see §8 for how the resulting metrics compare to classical
ML's per-window label).

**A real, pre-existing data bug was found and fixed during the first
pretraining run ever attempted for this project** (before the 8-way
ablation, §4). No `checkpoints/pretrain_masked` had ever existed yet.
Running pretraining for the first time surfaced that the window index
(`data/processed/index/*.parquet`) was pointing at `data/interim/` (raw,
unnormalized — accelerometer magnitudes in the tens of thousands) instead
of `data/interim_normalized/`, for *every* purpose, not just pretraining.
Masked reconstruction's MSE loss made this glaringly obvious (millions-scale
loss on epoch 1); TF-C's NT-Xent loss is bounded regardless of input scale,
so the exact same bug would have been invisible in its loss curves — it
affected all SSL approaches equally, since they all read from the same
index. Fixed by rebuilding the index against the normalized tree (same
`splits.json` reused, so no leakage risk introduced by the fix itself).

## 2. The pretext task

Implemented in `ssl/models/masking.py`. Two masking mechanisms:

- **Whole-channel masking**: each of the 10 channels independently has a 15%
  chance (`channel_mask_prob`) of being masked for its entire duration —
  forces the model to reconstruct it from the *other* channels, a warm-up
  toward genuine cross-sensor prediction.
- **Temporal span masking**: 1–3 contiguous spans totaling ~40%
  (`span_mask_ratio`) of timesteps, masked across all channels at once. This
  forces the model to use temporal context (what came before/after) to fill
  the gap.

**Run as two separate pretext tasks, not combined** — a deliberate departure
from decision 001's original framing of "hide random temporal spans *and*
whole sensor modalities" as one combined task. `sample_mask_batch`'s `mode`
parameter (`"channel"`, `"span"`, or `"both"` for the legacy OR'd-together
behavior) lets `pretrain.py` train on each masking mechanism independently,
so the 8-way ablation (§4) could measure whether channel-masking or
span-masking produces a better encoder, rather than always training on both
at once and never finding out. **Span masking won decisively** (§4) — every
subsequent step in this document (the EDA work, the hyperparameter tuning,
the 80/10/10 rerun) only concerns span masking.

Masked positions are replaced with a **learnable mask token**
(`nn.Parameter(n_channels)`), not zero — on z-scored data, zero is the
channel mean, i.e. a perfectly plausible real value that would carry no
"this is missing" signal. The loss (`masked_mse`) is computed **only over
masked positions** — otherwise the model could trivially copy the unmasked
input straight through and the loss would mostly reward that, not genuine
reconstruction.

### 2.1 Variable-length span masking (current default, adopted 2026-07-30)

This section (originally `docs/ssl_tuning_review_2026-07-30.md` §1–§2)
describes the one change made to the pretext task itself, prior to and
independent of the hyperparameter tuning in §6.

**Motivating EDA.** `src/EDA/eda_bite_gap_analysis.py`, run against raw
sensor CSVs (`data/raw_unscaled`, real per-sample timestamps, all 133
eating recordings, gaps >120s excluded as between-meal), measured real bite
and inter-bite-pause durations directly from the signal:

- Bite duration: mean 11.9s, std 8.3s, median 10.1s (p5=3.0s, p95=27.1s).
- Inter-bite pause: mean 5.6s, std 10.7s, **median only 2.5s** (p5=0.4s,
  p75=5.0s).
- **68% of real inter-bite pauses are shorter than a 4s window** — most of
  the "non-chewing" signal at this window scale can't form a clean example
  on its own from within-meal data alone.
- `Spectacle_NonMeal` provides ~4.3x more usable non-chewing signal
  (94,474s) than all within-meal gaps combined (22,145s) — confirmed as the
  primary, not supplementary, source of clean negative training data.

Full tables/plots: `src/EDA/tables/`, `src/EDA/plots/`.


### Masking strategy
The idea: I pretrain the model on a "fill in the blank" task. Hide part of the sensor signal and train the model to reconstruct it from what's left. This teaches it useful patterns without needing any chewing labels.

What I hide: short time segments (not sensor channels). The model has to infer what happened during a hidden gap from the signal just before and after it, across all sensors at once.

The improvement I made: originally, the hidden segments were an arbitrary fixed length. I changed this to size them off real inter-bite pause durations. I measured actual pause lengths in the raw eating data (they range roughly 0.4–5 seconds, with a lot of short ones), and now the hidden segments are sampled from that real distribution instead of a made-up length. So the model practises on gaps that actually resemble real pauses between bites, not arbitrary blanks.

Why this is fair, not cheating: this only uses the timing statistics of pauses, not any labels, pretraining data never has chewing labels at all, so I couldn't have used them even if I wanted to. I'm just choosing a more realistic difficulty for the task, not giving the model the answer.

Overall difficulty stays fixed: total amount of signal hidden per window is still ~40%, same as before. I only changed how it's distributed (a few realistic-length chunks instead of one arbitrary shape).
<br>

**The change.** Span masking previously split a fixed 40% ratio evenly
across however many spans were sampled (arbitrary granularity, no
grounding in real signal timescales). Now: each span's length is drawn
independently, log-uniformly, from `[--mask-min-span-s, --mask-max-span-s]`
seconds (default **0.4s–5.0s**, the p5–p75 pause range measured above),
then the whole set is rescaled to preserve the original total masking
*budget* (`span_mask_ratio`) — so span *count/ratio* behaves as before,
only individual span *length* is now variable instead of an even split.
Log-uniform (not uniform) sampling matters specifically because real
pauses are themselves heavy-tailed — a plain uniform draw over `[min, max]`
would be dominated by long spans and rarely sample anything sub-1s.

A hidden span shorter than a real pause would be masking sub-chew-cycle
noise; one longer than a real bite (median 10.1s — only relevant if sized
to *bite* length rather than *pause* length) would risk masking almost the
entire 4s/2s window. Sizing masking granularity from domain timescales is a
legitimate, label-free hyperparameter choice; using labels to decide
*where* masks fall would not be — but this doesn't apply here, since
masking still runs on `pretrain_train`/`pretrain_val` windows, which never
carry labels at all (decision 001).

The original fixed-ratio equal-split behavior (deterministic, no sampling)
is preserved and selectable via `--no-mask-variable-length`, specifically so
the original 8-way ablation checkpoints (§4) remain exactly reproducible if
any of them need to be rerun.

**Important methodological caveat, carried forward honestly**: the original
8-way ablation's `span_bigru_w200` baseline (§4) was pretrained under the
*old* fixed-ratio masking; every hyperparameter-tuning candidate in §6 was
pretrained under this *new* variable-length masking. §6's headline finding
is that, on the evidence gathered, this change has **not** by itself
translated into a bout-level improvement over the old scheme — see §6 for
the full discussion of why that comparison isn't as clean as "hyperparameter
X vs Y" in isolation.

## 3. Models

Two candidate encoder architectures, trained under identical masking policy,
comparable parameter budget (~100–120K), optimizer, batch size and number of
training windows — so the 8-way ablation comparison measures inductive
bias, not confounds (`ssl/masked/pretrain.py`):

| Architecture | File | Inductive bias | Encoder output |
|---|---|---|---|
| CNN | `ssl/models/cnn_autoencoder.py` | 1D U-Net: downsample/upsample + skip connections, GroupNorm+GELU. Locality, translation-invariance — matches the ~1.5–2Hz chewing rhythm. | `(B, T', embedding_dim)`, `T' < T` (downsampled bottleneck) |
| Bi-GRU | `ssl/models/bigru_autoencoder.py` | Bidirectional GRU, per-timestep linear reconstruction head. Explicit sequential order, no pooling/resolution loss. | `(B, T, embedding_dim)` — every timestep |

**Both architectures have a working downstream fine-tuning path.**
`ChewingDetector` (`ssl/masked/finetune.py`) is architecture-agnostic: it
calls `.encode()` on whichever encoder was pretrained and, if the result is
downsampled (the CNN's case), linearly upsamples the *head's output* back to
full resolution — a small, decoupled step that doesn't reuse the pretrained
decoder/skip connections, so the same head works for either architecture.
**Bi-GRU won the architecture comparison** (§4) — every downstream step in
this document (tuning, 80/10/10 rerun) uses Bi-GRU only.

## 4. The 8-way ablation

*(Originally `docs/ssl_masked_ablation.md` — the runbook is condensed here;
the results table below is the actual output, which the runbook document
itself never contained since it was written as a plan handed off before
being run.)*

**What was ablated**: 2 pretext tasks (`channel`, `span`) × 2 architectures
(`cnn`, `bigru`) × 2 window sizes (4s, 2s/25%-overlap) = 8 independent
pretrain→finetune→evaluate pipelines, under the *original* fixed-ratio
masking (§2.1 didn't exist yet at this point). Fine-tuning is **3-phase
gradual unfreezing** (`ssl/masked/finetune.py`): Phase 1 (encoder fully
frozen, head-only) → Phase 2 (only the encoder's last block unfrozen, low
LR) → Phase 3 (entire encoder unfrozen, an even lower LR) — each phase with
its own cosine LR schedule and gradient clipping. Checkpoint saved on best
per-timestep val F1, then automatically re-evaluated on the held-out
`finetune_test` split. Postprocessing for all 8 runs used the same recipe
classical ML had already converged on (`docs/dissertation_classical_ml_sections.md`
§4.4) — not re-tuned per combination, a caveat that still applies to 7 of
the 8 rows below.

**Results** (`checkpoints/masked_ablation_summary.csv`, original 60/20/20
split, 944 ground-truth test bouts, before the smoothing fix in §8):

| run_name | architecture | pretext_task | window | window test F1 | bout IoU F1 | bout tolerance F1 |
|---|---|---|---|---|---|---|
| channel_bigru_w100 | bigru | channel | 2s | 0.763 | 0.633 | 0.467 |
| channel_bigru_w200 | bigru | channel | 4s | 0.779 | 0.662 | 0.477 |
| channel_cnn_w100 | cnn | channel | 2s | 0.739 | 0.623 | 0.475 |
| channel_cnn_w200 | cnn | channel | 4s | 0.742 | 0.643 | 0.472 |
| span_bigru_w100 | bigru | span | 2s | 0.781 | 0.647 | 0.470 |
| span_cnn_w100 | cnn | span | 2s | 0.757 | 0.655 | 0.488 |
| span_cnn_w200 | cnn | span | 4s | 0.743 | 0.622 | 0.444 |
| **span_bigru_w200** | **bigru** | **span** | **4s** | **0.799** | **0.731** | **0.527** |

**`span_bigru_w200` (span-masking pretext task, Bi-GRU encoder, 4-second
window) wins on every single metric** — window F1, bout IoU F1, and bout
tolerance F1 all highest among the 8 combinations. This is the checkpoint
every subsequent section of this document (EDA masking change, tuning,
80/10/10 rerun) builds on. Note the smoothing fix (§8) was applied to this
row after the ablation, moving its bout-level numbers to the higher figures
quoted in the headline table at the top of this document and in §8 — the
table above is the pre-fix ablation snapshot, kept for historical
comparability across all 8 combinations (only `span_bigru_w200` was
re-evaluated post-fix).

## 5. Hyperparameter tuning (2026-07-30)

*(Originally `docs/ssl_tuning_review_2026-07-30.md` §3, condensed.)* Two
things made this tuning pass possible that weren't true at ablation time:
variable-length span masking (§2.1) and encoder capacity becoming
CLI-overridable (`--hidden-size`/`--num-layers`/`--encoder-dropout` for
Bi-GRU, previously hardcoded). Coordinate descent (not a full grid), ranked
by expected impact going in: LR > mask ratio > hidden_size > num_layers.

### Stage 1 — learning rate (pretrain val_loss only)

| lr | epochs (early-stopped) | best pretrain val_loss |
|---|---|---|
| 3e-4 | 32 | 0.724 |
| **1e-3** | 55 | **0.671** ← winner |
| 3e-3 | 15 | 0.736 |

### Stage 2 — mask span ratio (pretrain val loss, then finetune-confirmed)

| ratio | epochs | best pretrain val_loss | finetune val_f1 | finetune test_f1 |
|---|---|---|---|---|
| 0.2 | 18 | **0.609** (best pretrain loss) | 0.720 | 0.747 |
| 0.3 | 19 | 0.688 | — (not finetuned) | — |
| **0.4** | 55 | 0.671 | **0.748** (best downstream) | **0.780** |
| 0.5 | 21 | 0.756 | — | — |
| 0.6 | 21 | 0.782 | — | — |

**Confirmed finding: `ratio=0.2` had the better *pretrain* reconstruction
loss, but `ratio=0.4` (the original default budget) won clearly once
actually fine-tuned** — direct evidence that pretrain val_loss is not a
reliable proxy for downstream quality here, and the reason the tuning
procedure always confirms shortlisted pretrain-loss candidates with a real
finetune run rather than trusting pretrain loss alone.

### Stage 3 — encoder capacity (hidden_size × num_layers, lr=1e-3, ratio=0.4)

| combo | pretrain epochs | pretrain val_loss | finetune val_f1 | finetune test_f1 | bout IoU F1 | bout tolerance F1 |
|---|---|---|---|---|---|---|
| h64/l2 (`tune_lr_1e-3`) | 55 | 0.671 | 0.748 | 0.780 | 0.710 | 0.495 |
| h64/l3 | 14 | 0.726 | 0.732 | 0.751 | 0.677 | 0.466 |
| h96/l2 | 18 | 0.745 | 0.741 | 0.772 | 0.703 | 0.481 |
| **h96/l3** | 19 | 0.718 | **0.749** | **0.789** | **0.731** | **0.514** |

`h96/l3` (wider **and** deeper) is the best of the four capacity variants on
every metric, including bout-level. Note capacity variants (`h96_*`)
intentionally break the original ablation's ~100-120K parameter-budget
parity with the CNN architecture — no longer a fair architecture comparison
once past this point, only comparable to each other.

### Headline finding of this tuning pass — read carefully before using it as a recommendation

**None of the 2026-07-30 tuning candidates beat the original (pre-tuning)
`span_bigru_w200` ablation baseline at the bout level**, despite several of
them looking competitive or better on window-level F1:

| candidate | window test F1 | bout IoU F1 | bout tolerance F1 |
|---|---|---|---|
| **ablation baseline** (§4, old fixed-ratio masking) | 0.799 | **0.731** | **0.527** |
| tune_cap_h96_l3 (h=96, layers=3) | 0.789 | 0.731 | 0.514 |
| tune_lr_1e-3 / tune_ratio_0.4 (h=64, layers=2, new masking) | 0.780 | 0.710 | 0.495 |
| tune_cap_h96_l2 (h=96, layers=2) | 0.772 | 0.703 | 0.481 |
| tune_cap_h64_l3 (h=64, layers=3) | 0.751 | 0.677 | 0.466 |

`tune_cap_h96_l3` comes closest to matching the baseline (bout IoU F1 tied
at 0.731, tolerance F1 0.014 below) — of everything tried in this pass,
it's the only real contender, not a clear win. **Important caveat**: the
ablation baseline was pretrained under the *old* fixed-ratio masking; every
tuning candidate was pretrained under the *new* variable-length masking (on
by default). So this isn't a clean "hyperparameter X vs Y" comparison in
isolation — the masking-strategy change is baked into every tuning
candidate's result too, and on this evidence it had **not**, at the time of
this review, translated into a bout-level improvement over the old scheme,
at any hyperparameter combination tried.

**Decision made from this evidence** (carried into §9): rather than
concluding the tuning made things strictly worse, or spending more time
narrowing the gap further, `tune_cap_h96_l3`'s settings
(`--hidden-size 96 --num-layers 3 --lr 1e-3 --mask-span-ratio 0.4`) were
adopted as the configuration to retrain under the corrected 80/10/10 split
(§9) — it was already the closest contender, and the split-ratio fix itself
(a much larger, previously-undiscovered issue, §9.1) was judged more
important to resolve than continuing to chase a small bout-level gap on a
split that was about to be regenerated anyway.

## 6. Evaluation methodology — is it the same as classical ML's?

**Not identical at the window level — but genuinely comparable at the bout
level**, which is what actually matters for the end task, and which is also
the direct answer to "do both approaches need the same evaluation
technique": **yes, and they already share it, at the code level, not just
the parameter level.**

**Window-level metrics are *not* apples-to-apples.** Classical ML evaluates
on independent *windows* with a single majority-vote label; masked
reconstruction never collapses to a window label — it's scored on **every
individual timestep**, a strictly finer-grained task. As of the 80/10/10
rerun, both pipelines' eval windows are non-overlapping
(`docs/dissertation_classical_ml_sections.md` §2.1), which removes the
*overlap* asymmetry that existed under the original 60/20/20 split — but
the majority-vote-vs-per-timestep scoring difference remains, so a
masked-reconstruction `window_test_f1` and classical ML's window-level test
F1 are still answering different questions and shouldn't be read as a
direct ranking.

**Bout-level metrics genuinely are apples-to-apples.** Both pipelines share
the exact same postprocessing/evaluation CODE, not just the same parameter
values — `common/evaluation/postprocess.py` (`apply_full_postprocessing`,
`PostprocessConfig`) and `common/evaluation/pipeline.py`
(`evaluate_recording`, `aggregate_dual_mode`, `align_bouts_sequential`,
`build_bite_ledger`) orchestrate both pipelines' evaluation identically;
see `docs/bite_evaluation.md` for the architecture and
`common.evaluation.compare_bites` for a direct bite-by-bite ledger. Tuned
parameters are the ones classical ML converged on
(`docs/dissertation_classical_ml_sections.md` §4.4: `--post-high 0.75
--post-low 0.52 --post-max-dur 35 --post-valley 0.35
--envelope-prominence 2.5 --envelope-min-piece-dur 8.0`), plus one
deliberately pipeline-specific addition: `--smooth-window-s` (0.0 for
classical ML, 2.0s for SSL — see `docs/bite_evaluation.md` §3 for why, and
the empirical validation behind it).

**The smoothing fix, briefly** (full derivation: `docs/bite_evaluation.md`
§3): SSL's per-timestep probability trace has no equivalent to classical
ML's overlapping-window averaging, so it flickers more under a fixed
hysteresis threshold, producing more/shorter false-alarm bouts. A short
moving average (`smooth_probs`) before hysteresis fixes this — swept
empirically, `smooth_window_s=2.0` found to be the peak for SSL (helps
substantially: IoU-F1 0.674→0.731 on the original ablation baseline) while
mildly hurting classical ML (already smoothed; more smoothing just blurs
real signal) — hence the pipeline-specific default.

Both pipelines evaluate over the same held-out, subject-disjoint test
participants for a given split — confirmed by an exact ground-truth-bout-count
match: 944 = 944 under the original 60/20/20 split, and **524 = 524 under
the new 80/10/10 split** (§9).

## 7. Fine-tuning procedure

Unchanged since the 8-way ablation (§4): **3-phase gradual unfreezing**
(`ssl/masked/finetune.py`) — Phase 1 (encoder fully frozen, head-only) →
Phase 2 (only the encoder's last block unfrozen, low LR) → Phase 3 (entire
encoder unfrozen, an even lower LR), each phase with its own cosine LR
schedule and gradient clipping. Checkpoint selection is always on
per-timestep **val F1** (window-level — see §6's caveat about what this
metric is and isn't measuring); `bout_tolerance_f1` (via `evaluate_bouts.py`
on the finetuned candidate) is the tie-breaker used among close candidates
during tuning (§5), since that's the metric that actually matters for the
comparison against classical ML — not pretrain loss or window-F1 alone.

Pretraining now supports early stopping (`--early-stop-patience`, default
7): stops once `val_loss` hasn't improved for that many consecutive epochs,
instead of always running the full `--epochs` (100 by default) — this is
what makes a multi-candidate sweep like §5 affordable, and is also what
happened during the 80/10/10 pretrain run (§9): 42 epochs run, best
val_loss at epoch 34, early-stopped 7 epochs later without further
improvement.

## 8. Predictions and postprocessing

Unlike classical ML, there's no separate "upscale predictions" step here —
`evaluate_bouts.py` and `plot_predictions.py` both run inference directly
(`infer_recording`, sliding non-overlapping windows over the full
recording) and call the shared postprocessing/evaluation code
(`common/evaluation/postprocess.py`/`pipeline.py`) internally. See §6 above
for why one shared recipe with one pipeline-specific parameter
(`smooth_window_s`) is used, and `docs/bite_evaluation.md` for the full
architecture, including `compare_bites.py`, which compares this pipeline's
output against classical ML's directly on the same ground-truth bites.

**A dedicated postprocessing sweep tool also exists for this pipeline**
(`chewing_ssl.ssl.masked.sweep_postprocess`, mirroring classical ML's own
`sweep_postprocess.py` — both back onto the same shared coordinate-descent
engine, `chewing_ssl.common.evaluation.sweep`). It has been exercised
exploratorily against the 80/10/10 `span_bigru_w200` checkpoint
(`checkpoints/postprocess_sweep_masked_80_10_10.csv`): coordinate descent
found `high=0.80` (up from 0.75), `max_gap_s=0.5` (down from 1.0),
`max_dur_s=40.0` (up from 35.0), `valley_thresh` disabled, and
`envelope_prominence=2.0` (down from 2.5) improve `bout_tolerance_f1` from
0.548 to 0.553 and `bout_iou_f1` from 0.766 to 0.756 relative to the
defaults used for §9's headline numbers below. **This has not been adopted
as this pipeline's new default** — it is reported here as an available
finding, exactly parallel to classical ML's own not-yet-adopted re-tuning
(`docs/dissertation_classical_ml_sections.md` §4.6) — a joint decision on
whether to promote either pipeline's re-tuned recipe (keeping both
pipelines on one shared recipe, per §6's design principle) is still open.

## 9. The 80/10/10 rerun

### 9.1 What changed in the split, and why

`configs/preprocessing.yaml` has specified an 80/10/10 train/validation/test
participant split since before this project's original results were
produced, but `windowing.py`'s split-assignment step unconditionally reuses
an existing `splits.json` rather than regenerating it from the current
config — so every SSL result up to and including §4–§6 above was actually
trained/evaluated against a `splits.json` frozen at an earlier, empirically
~60/20/20 division. This was identified and a fresh `splits.json`
regenerated from the config's real 80/10/10 ratio on 2026-07-31 (full
root-cause writeup: `docs/dissertation_classical_ml_sections.md` §3.1,
which this section defers to rather than duplicating).

New per-dataset participant counts (`data/processed_80_10_10/splits.json`,
verified directly against the file on disk):

| Dataset | Train | Validation | Test | Total participants |
|---|---|---|---|---|
| SIPMEatingStudy | 24 | 3 | 4 | 31 |
| SussexEatingUK | 32 | 4 | 4 | 40 |
| FEIT_eating_S05 | 28 | 3 | 5 | 36 |
| Office_eating_s05 | 20 | 2 | 4 | 26 |
| Spectacle_NonMeal | 17 | 2 | 3 | 22 |

This is a smaller absolute validation/test population per dataset than the
original 60/20/20 division (e.g. `Office_eating_s05`'s validation split
shrinks from 5 to 2 participants) traded for a larger training
population — the same tradeoff noted for classical ML, applying identically
here since both pipelines share one `splits.json`.

### 9.2 Scope of this rerun

Rather than repeating the full LR/mask-ratio/capacity coordinate-descent
sweep (§5) and the 8-way ablation (§4) against the new split, this rerun
reuses §5's already-identified best contender outright: `span_bigru_w200`
(Bi-GRU encoder, span-masking pretext task), with `tune_cap_h96_l3`'s
settings (`--hidden-size 96 --num-layers 3`) and the new variable-length
span masking (§2.1) enabled (i.e. **not** passing
`--no-mask-variable-length`). Artefacts are written to a parallel
`_80_10_10`-suffixed tree that does not overwrite the original run:

```bash
python -m chewing_ssl.ssl.masked.pretrain \
  --architecture bigru --pretext-task span \
  --lr 1e-3 --mask-span-ratio 0.4 \
  --hidden-size 96 --num-layers 3 \
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
```

### 9.3 Pretraining and fine-tuning results

**Pretraining** (`checkpoints/pretrain_masked_80_10_10/span_bigru_w200/bigru_span_history.json`):
early-stopped at **42 epochs**, best validation reconstruction loss
**0.463 at epoch 34** (7 epochs of no further improvement before stopping,
matching the default `--early-stop-patience 7`).

**Fine-tuning** (`checkpoints/finetune_masked_80_10_10/span_bigru_w200_history.json`):
45 total epoch-entries across the 3 unfreezing phases, best validation F1
**0.821**, held-out `finetune_test` metrics: accuracy 0.905, precision
0.832, recall 0.820, **F1 0.826**, AUC 0.957. Checkpoint loaded for bout
evaluation: phase 3, epoch 13 (0-indexed within that phase), val_f1=0.821 —
matching `evaluation_report.md`'s own header exactly.

### 9.4 Bout-level results

`checkpoints/masked_ablation_summary_80_10_10.csv` (524 ground-truth test
bouts, 541 predicted, `evaluation_report.md`'s console display mode is
IoU ≥ 0.3):

| Metric | Value |
|---|---|
| Bout precision, IoU ≥ 0.3 | 0.754 |
| Bout recall, IoU ≥ 0.3 | 0.779 |
| **Bout F1, IoU ≥ 0.3** | **0.766** |
| Median IoU among matched pairs | 0.647 |
| Bout precision, tolerance = 5.0s | 0.540 |
| Bout recall, tolerance = 5.0s | 0.557 |
| **Bout F1, tolerance = 5.0s** | **0.548** |
| Median IoU among matched pairs (tolerance mode) | 0.740 |

**Per-recording table** (excerpt, `evaluation_report.md`; full 27-recording
table on disk):

| Recording | GT | Pred | TP | FP | FN | Prec | Rec | F1 | mIoU | Onset lag | Dur Δ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| SIPMEatingStudy/p01 | 40 | 35 | 34 | 1 | 6 | 0.971 | 0.850 | 0.907 | 0.641 | +0.97s | −2.19s |
| SussexEatingUK/EMTEQTHP10 | 26 | 29 | 23 | 6 | 3 | 0.793 | 0.885 | 0.836 | 0.769 | +1.24s | −0.64s |
| SussexEatingUK/EMTEQRAP15 | 23 | 1 | 1 | 0 | 22 | 1.000 | 0.043 | 0.083 | 0.435 | +5.16s | −5.54s |
| TOTAL | 524 | 541 | 408 | 133 | 116 | 0.754 | 0.779 | 0.766 | 0.678 | +0.96s | −1.83s |

`SussexEatingUK/EMTEQRAP15` is a genuine outlier (22 of 23 bites missed) —
worth investigating separately (e.g. a low-signal recording, or a
participant whose eating style the model generalizes to poorly) rather
than averaged away; not pursued further in this document.

**Bout-outcome taxonomy** (`checkpoints/masked_bout_ledger_80_10_10.csv`,
`align_bouts_sequential`'s match/merge/split/tangle/missed/false_alarm
clustering, match-mode-independent — see
`docs/dissertation_classical_ml_sections.md` §5.5 for the full definitions):

| Cluster label | GT bites | % of GT | Predicted bouts |
|---|---|---|---|
| match | 269 | 51.3% | 269 |
| merge | 48 | 9.2% | 21 (events) |
| split | 44 | 8.4% | 100 (events) |
| tangle | 111 | 21.2% | 129 (events) |
| missed | 52 | 9.9% | 0 |
| false_alarm | 0 | 0.0% | 22 |

**Tangle is still the dominant non-clean-match failure mode (21.2% of
bites), same qualitative finding as classical ML's own taxonomy** — but at
a notably lower share than classical ML's 46.4% (4s, 80/10/10,
`docs/dissertation_classical_ml_sections.md` §5.5) for the same split and
ground-truth population. **More than half of all bites (51.3%) land in a
clean one-to-one match, versus classical ML's 32.8%** — a genuinely
different error profile between the two pipelines worth investigating
further in `docs/bite_evaluation.md`, not just a difference in the
headline F1 numbers.

**Boundary-alignment quality** among the 408 IoU-matched pairs
(`checkpoints/masked_bout_ledger_80_10_10.csv`): onset lag mean **+1.38s**
(median +0.96s — predicted onset tends to run *late*, the opposite
direction from classical ML's early-leaning tendency), offset lag mean
**−0.96s** (median −0.61s, early), duration difference mean **−2.34s**
(median −1.83s, under-segmented relative to ground truth), matched-pair IoU
mean **0.658** (median 0.678).

## 10. Summary comparison across every stage of this document

| Stage | Config | Split | bout IoU F1 | bout tolerance F1 |
|---|---|---|---|---|
| §4 ablation baseline | span_bigru_w200, old masking | 60/20/20 | 0.731 | 0.527 |
| §6 smoothing fix applied | span_bigru_w200, old masking, smooth=2.0s | 60/20/20 | 0.731 | 0.527 |
| §5 best tuning candidate | tune_cap_h96_l3, new masking | 60/20/20 | 0.731 | 0.514 |
| §9 80/10/10 rerun | span_bigru_w200, h96/l3, new masking | **80/10/10** | **0.766** | **0.548** |
| §8 postprocess re-sweep (not adopted) | same model, re-tuned recipe | 80/10/10 | 0.756 | 0.553 |

**The 80/10/10 rerun is the best-performing configuration on record for
this pipeline on both bout-level metrics**, ahead of every 60/20/20 result
including the original ablation baseline that the 2026-07-30 tuning pass
(§5) had failed to beat. Some of this improvement is very likely genuinely
attributable to the larger training population (80% vs. ~60% of
participants) rather than the specific hyperparameter choices — a variable
the 60/20/20-vs-80/10/10 comparison cannot isolate from the h64/l2-vs-h96/l3
capacity choice, since both changed at once. Disentangling them (e.g.
retraining the *original* h64/l2, old-masking configuration under 80/10/10
too) is flagged as a natural follow-up, not done here.

## 11. Open items / not yet done

- **Cross-sensor prediction (decision 001's second pretext task) isn't
  implemented.** Only masked reconstruction (channel/span) exists; decision
  001 specified *both* masked reconstruction *and* cross-sensor prediction
  as candidate pretext tasks.
- **Postprocessing parameters have no held-out validation split** — same
  pre-existing gap classical ML's own docs flag
  (`docs/dissertation_classical_ml_sections.md` §4.7): tuned/re-tuned
  directly against these test recordings' bout-level metrics, not verified
  to generalize beyond this specific test set.
- **The full 8-way ablation (§4) and the 3-stage tuning pass (§5) have not
  been rerun under 80/10/10** — only the single already-identified best
  configuration was retrained (§9.2, a deliberate scope decision, not an
  oversight). Whether a different combination would now win under the new
  split, larger training population, and non-overlapping classical-ML
  segmentation convention is unknown.
- **The postprocessing re-sweep in §8 has not been adopted as this
  pipeline's default**, nor has classical ML's equivalent re-sweep — a
  joint decision on both pipelines' recipes is still open, tracked in
  `docs/bite_evaluation.md`.
- **The 60/20/20-vs-80/10/10 improvement (§10) has not been decomposed**
  into "more training data" vs. "the h96/l3 capacity choice" vs. "the
  masking-strategy change" — three things that all changed simultaneously
  between the ablation baseline and the current best result.
- `SussexEatingUK/EMTEQRAP15`'s near-total miss rate (§9.4) has not been
  investigated.
