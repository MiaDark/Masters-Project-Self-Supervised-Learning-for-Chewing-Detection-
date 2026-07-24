# SSL: Masked reconstruction

Decision 001's original self-supervised pretext task: reconstruct deliberately
hidden parts of a window from context. This is one of three SSL pretraining
approaches in this project (the other two, TF-C and TF-FC, are contrastive —
see `docs/ssl_tfc.md` / `docs/ssl_tffc.md`); this one predates and is
independent of them. Implemented in `ssl/masked/`.

## 1. Input

Same 10-channel, globally z-scored window as every other pipeline in this
project: `accel_x`, `accel_y`, `accel_z`, `nav_x_left_temple`,
`nav_y_left_temple`, and each of those five again as a first-derivative
(`_d1`) channel — shape `(10, T)`, `T=200` (4s @ 50Hz) by default, `T=100`
for the 2s/25%-overlap variant.

**`preprocess.py`'s raw-to-normalized pipeline fully applies here**: gap
segmentation, resampling to 50Hz, the zero-phase Butterworth low-pass filter
(8Hz cutoff, order 4), and the first-derivative channels are **not**
classical-ML-specific; they're shared upstream infrastructure that produces
`data/interim_normalized/`, which every pipeline in this project (classical
ML, masked reconstruction, TF-C, TF-FC) reads from identically. See
`docs/preprocessing.md` for the full raw→interim→normalized sequence
(`preprocess.py` → `windowing.py --splits-only` → `refit_global_norm.py`).

What differs between classical ML and masked reconstruction is one step
*later*: how the already-preprocessed signal gets turned into training
examples.

| | Classical ML | Masked reconstruction (+ TF-C/TF-FC) |
|---|---|---|
| Windowing code | Its own `classical_ml/segment.py` | Shared `common/data/windowing.py` |
| Materialization | Eagerly materializes dense `(N, 10, T)` arrays to disk | Lightweight index only (`segment_path, start, end`) — `WindowDataset` slices the real values out of the cached normalized parquet at read time |
| Window/stride | Configurable via CLI (`--window-seconds`/`--stride-seconds`); the *same* stride is applied to every split, including eval | Fixed convention: 50% overlap for train-purpose windows, **zero** overlap for eval-purpose windows (deliberately, to avoid inflating effective eval sample size) |
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
described above (see §6 for how that compares to classical ML's per-window
label).

## 2. The pretext task

Implemented in `ssl/models/masking.py`. Two masking mechanisms, combined per
window (decision 001: "hide random temporal spans *and* whole sensor
modalities"):

- **Whole-channel masking**: each of the 10 channels independently has a 15%
  chance (`channel_mask_prob`) of being masked for its entire duration —
  forces the model to reconstruct it from the *other* channels, a warm-up
  toward genuine cross-sensor prediction.
- **Temporal span masking**: 1–3 contiguous spans covering ~40%
  (`span_mask_ratio`) of timesteps, masked across all channels at once. This
  forces the model to use temporal context (what came before/after) to fill
  the gap.

Masked positions are replaced with a **learnable mask token**
(`nn.Parameter(n_channels)`), not zero — on z-scored data, zero is the
channel mean, i.e. a perfectly plausible real value that would carry no
"this is missing" signal. The loss (`masked_mse`) is computed **only over
masked positions** — otherwise the model could trivially copy the unmasked
input straight through and the loss would mostly reward that, not genuine
reconstruction.

## 3. Models

Two candidate encoder architectures, trained under identical masking policy,
comparable parameter budget (~100–120K), optimizer, batch size and number of
training windows — so the comparison measures inductive bias, not confounds
(`ssl/masked/pretrain.py`):

| Architecture | File | Inductive bias | Encoder output |
|---|---|---|---|
| CNN | `ssl/models/cnn_autoencoder.py` | 1D U-Net: downsample/upsample + skip connections, GroupNorm+GELU. Locality, translation-invariance — matches the ~1.5–2Hz chewing rhythm. | `(B, T', embedding_dim)`, full temporal resolution preserved via skips |
| Bi-GRU | `ssl/models/bigru_autoencoder.py` | Bidirectional GRU, per-timestep linear reconstruction head. Explicit sequential order, no pooling/resolution loss. | `(B, T, 128)` — 64 forward + 64 backward hidden units, every timestep |

Only **Bi-GRU** currently has a downstream fine-tuning head built
(`ChewingDetector` in `ssl/masked/finetune.py` wraps the pretrained GRU
directly) — CNN pretraining is implemented and runnable
(`--architecture cnn`), but there's no `ChewingDetector`-equivalent for it
yet, so it isn't part of the trained/evaluated pipeline below.

## 4. Training

### 4.1 Pretraining (`ssl/masked/pretrain.py`)

```
python -m chewing_ssl.ssl.masked.pretrain --architecture bigru --epochs 20 \
    --checkpoint-dir checkpoints/pretrain_masked_4s
```

AdamW, MSE loss on masked positions only, logged per-channel too (so a
channel that reconstructs worse than others is visible, not averaged away).
`shuffle_by_segment` keeps windows from the same segment consecutive within
an epoch so the small LRU segment cache doesn't thrash.

**This had never actually been run before this round** — no
`checkpoints/pretrain` existed. Doing so surfaced a real, pre-existing bug:
the window index (`data/processed/index/*.parquet`) was pointing at
`data/interim/` (raw, unnormalized, accelerometer magnitudes in the tens of
thousands) instead of `data/interim_normalized/` — every purpose, not just
pretraining. Masked-reconstruction's MSE loss made this glaringly obvious
(millions-scale loss on epoch 1); TF-C/TF-FC's NT-Xent loss is bounded
regardless of input scale, so the same bug was invisible in their loss
curves even though it affected them identically. Fixed by rebuilding the
index against the normalized tree (same `splits.json` reused, so no leakage
risk) — see `docs/ssl_tfc.md` for the full story, since it affects all three
SSL approaches equally.

**Real result** (bigru, 20 epochs, post-fix): train loss 0.62 → 0.50, val
loss → 0.72, ~145s/epoch.

### 4.2 Fine-tuning (`ssl/masked/finetune.py`)

```
python -m chewing_ssl.ssl.masked.finetune \
    --checkpoint checkpoints/pretrain_masked_4s/bigru_best.pt \
    --out-dir checkpoints/finetune_masked_4s
```

`ChewingDetector` = the pretrained GRU + a linear head projecting each
timestep's 128-dim hidden state to one chewing logit — output `(B, T)`, one
probability **per timestep**, not per window. Two-phase, same convention as
TF-C/TF-FC:

- **Phase 1** (5 epochs): encoder frozen, only the head trains.
- **Phase 2** (10 epochs): encoder unfrozen at 10x lower LR, so pretrained
  representations adapt without catastrophic forgetting.

Class-imbalanced BCE (`pos_weight` from the train split's chewing fraction).
Checkpoint saved on best val F1, then automatically re-evaluated on the
held-out `finetune_test` split at the end (this final test step didn't
previously exist for TF-C/TF-FC's equivalent script — added this round, see
`docs/ssl_tfc.md`).

**Real result:** val F1 climbed smoothly 0.185 → 0.775 across both phases,
then:

| Split | Loss | F1 | Precision | Recall | AUC | Accuracy |
|---|---|---|---|---|---|---|
| val (best) | 0.299 | 0.775 | 0.788 | 0.762 | 0.932 | 0.876 |
| **test** | 0.274 | **0.804** | 0.759 | 0.854 | 0.947 | 0.885 |

## 5. How predictions are made

At inference (`ssl/masked/evaluate_bouts.py`): slide **non-overlapping**
`window_len`-sized chunks over a full test recording (same no-overlap
convention `windowing.py` already uses for eval purposes), run each through
`ChewingDetector`, and its `(T,)` output slots directly into the
corresponding stretch of a per-timestep probability array for the whole
recording — no averaging needed, since chunks never overlap (contrast with
the classical-ML path, whose overlapping windows require averaging at
upscale time; see §6).

That continuous probability trace then goes through postprocessing
(`common/evaluation/postprocess.py`): hysteresis thresholding → gap-fill →
duration/valley-based splitting → raw-signal envelope splitting → minimum
duration filter — producing discrete predicted chewing bouts, matched
against ground-truth bouts by IoU or onset/offset tolerance
(`common/evaluation/bouts.py`).

```
python -m chewing_ssl.ssl.masked.evaluate_bouts \
    --checkpoint checkpoints/finetune_masked_4s/bigru_finetune_best.pt \
    --pred-dir checkpoints/predictions_masked_4s

python -m chewing_ssl.ssl.masked.plot_predictions --all \
    --checkpoint checkpoints/finetune_masked_4s/bigru_finetune_best.pt \
    --out-dir checkpoints/plots/masked_predictions_4s/all
```

### 5.1 Postprocessing: default vs. classical-ML-tuned

`evaluate_bouts.py` originally only exposed hysteresis + gap-fill + a flat
minimum duration (no splitting). Extended this round to add the same
duration-bounded splitting, interior-valley splitting, and raw-signal
envelope splitting the classical-ML pipeline arrived at after its own
iteration (`docs/classical_ml.md` §6a–§6c) — same shared code
(`common/evaluation/postprocess.py`), same tuned parameter values
(`--post-high 0.75 --post-low 0.52 --post-max-dur 35 --post-valley 0.35
--envelope-prominence 2.5 --envelope-min-piece-dur 8.0`), now the defaults
here too.

**Measured effect** (48 test recordings, GT=944 bouts — same total as
classical ML's, confirming both share the exact held-out test population):

| Postprocessing | Match mode | Precision | Recall | F1 | Median IoU |
|---|---|---|---|---|---|
| Default (hysteresis 0.65/0.35 only, no splitting) | IoU≥0.3 | 0.435 | 0.350 | 0.388 | 0.600 |
| **Classical-ML-tuned** | IoU≥0.3 | **0.593** | **0.754** | **0.664** | 0.687 |
| Classical-ML-tuned | tolerance=5s | 0.424 | 0.539 | 0.475 | 0.766 |

Same qualitative finding classical ML had: tuned postprocessing (especially
envelope-based splitting on the raw nav signal, which recovers bite-to-bite
pauses the smoothed probability trace washes out) is a large, genuine win —
F1 nearly doubled. On `SIPMEatingStudy/p01` specifically (classical ML's own
visual reference recording): **36/40 GT bouts detected, 2 false alarms**
(tolerance mode) — see
`checkpoints/plots/masked_predictions_4s/all/SIPMEatingStudy_p01_segment_00.png`.

## 6. Is the evaluation the same as classical ML's?

**Not identical — but converges to genuinely comparable at the level that
actually matters.**

**Window-level metrics are *not* apples-to-apples.** Classical ML evaluates
on independent *windows* with a single majority-vote label, and (per its own
documented caveat) its eval windows currently overlap 50%, inflating its
numbers somewhat. Masked-reconstruction never collapses to a window label —
it's scored on **every individual timestep**, a strictly finer-grained task,
over non-overlapping windows. So test F1=0.804 (masked-reconstruction,
per-timestep) and classical ML's test F1=0.844/0.846 (window-level,
overlapping) are answering different questions and shouldn't be read as a
direct ranking.

**Bout-level metrics now genuinely are apples-to-apples.** This took real
work this round: masked-reconstruction's bout evaluator was extended to use
the exact same shared code (`common/evaluation/bouts.py`,
`postprocess.py`) with the same tuned parameters classical ML converged on.
Both pipelines evaluate over the same held-out, subject-disjoint test
participants (confirmed by the exact GT-bout-count match, 944 = 944). The
resulting numbers — masked-reconstruction F1=0.664 (IoU-mode, tuned
postprocessing) vs. classical ML's F1=0.763 (4s) / 0.743 (2s), both under
the identical postprocessing recipe — are a fair, like-for-like comparison,
even though classical ML is still ahead at this level.

## Open items / not yet done

- **Only Bi-GRU has a downstream detector.** CNN pretraining runs and is
  logged, but there's no `ChewingDetector`-equivalent wired to it, so the
  original "compare CNN vs. Bi-GRU inductive bias" framing from
  `pretrain.py`'s own docstring isn't actually testable end-to-end yet.
- **Cross-sensor prediction (decision 001's second pretext task) isn't
  implemented.** Only masked reconstruction exists; decision 001 specified
  *both* masked reconstruction *and* cross-sensor prediction as combined
  pretext tasks.
- **Postprocessing parameters have no held-out validation split** — same
  pre-existing gap classical ML's own docs flag: `--post-high`/`--post-low`/
  envelope parameters were tuned directly against these 48 test recordings'
  bout-level metrics (by classical ML, then reused here unchanged), not
  verified to generalize beyond this specific test set.
- **The 2s/25%-overlap variant** (window index already built at
  `data/processed_w2s0.25`, matching classical ML's second configuration)
  hasn't been pretrained/fine-tuned yet — paused mid-run at the user's
  request; the 4s results above are the only completed run so far.
