# SSL Masked-Reconstruction Ablation Study — Pipeline & Ronin Runbook

An expanded version of the masked-reconstruction pipeline (`docs/ssl_masked.md`),
built to run channel-masking and span-masking as two genuinely **separate**
pretext tasks (not OR'd together), across both encoder architectures, across
both window sizes — an 8-way ablation — with deeper fine-tuning, on Ronin
cloud GPU (g6.2xlarge). This document covers every step from pretraining
through per-timestep prediction ("upscaling"), stopping short of bout-level
postprocessing tuning (that stays a separate, later step — see
`docs/ssl_masked.md` §5.1 and `docs/classical_ml.md` §6 for how that was
approached for the single 4s run already completed).

## 1. The 8 combinations

| Axis | Values |
|---|---|
| Pretext task | `channel` (whole-channel masking only), `span` (temporal span masking only) |
| Architecture | `cnn`, `bigru` |
| Window size | 4s (`data/processed`), 2s/25%-overlap (`data/processed_w2s0.25`) |

2 × 2 × 2 = 8 independent pretrain→fine-tune→evaluate pipelines. Every
output is named `{pretext_task}_{architecture}_w{window_len_samples}`, e.g.
`channel_bigru_w200` (4s) or `span_cnn_w100` (2s) — this `run_name` is stamped
into every checkpoint's metadata and used consistently for every output
directory/file below, so results from different combinations never collide
even if run in any order or re-run individually.

## 2. What changed in the code (this round)

- **`ssl/models/masking.py`**: `sample_mask_batch` gained a `mode` parameter
  (`"channel"`, `"span"`, `"both"`) — channel-masking and span-masking can
  now be sampled independently instead of always OR'd together.
- **`ssl/models/cnn_autoencoder.py`**: fixed to handle window lengths not
  divisible by `2**depth` (2s windows = 100 samples, not divisible by 8) via
  replicate-padding before the down-path and cropping after the up-path.
  Also added optional dropout in `ConvBlock` for regularization now that
  training runs much longer.
- **`ssl/models/bigru_autoencoder.py`**: added `layer_parameters(layer_idx)`,
  used for gradual unfreezing in fine-tuning.
- **`ssl/masked/pretrain.py`**: new required `--pretext-task {channel,span,both}`
  flag; checkpoints now stamp `architecture`, `pretext_task`, `window_len`,
  `model_kwargs` metadata (previously just raw weights) so every downstream
  script can auto-detect everything about a checkpoint without extra flags;
  checkpoint filenames are `{architecture}_{pretext_task}_best.pt`; default
  epochs raised 10 → 50 (GPU-appropriate).
- **`ssl/masked/finetune.py`** (rewritten): `ChewingDetector` is now
  architecture-agnostic — works with either encoder via one `.encode()` →
  head → (upsample if needed) interface, auto-detected from the pretrain
  checkpoint (no `--architecture` flag needed). Fine-tuning is now
  **3-phase gradual unfreezing** (frozen → last encoder block only → full
  encoder), each phase with its own cosine LR schedule and gradient
  clipping, epoch budgets raised (10/15/20 vs. the old 5/10). Every
  checkpoint stamps full metadata for downstream auto-detection.
- **`ssl/masked/evaluate_bouts.py`**: architecture-agnostic loading (same
  auto-detection); computes bout-level metrics under **both** IoU and
  tolerance matching every run (not just whichever `--match-mode` is
  displayed); new `--summary-csv` appends one row per combination — the
  single comparison table across all 8 runs.
- **`ssl/masked/plot_predictions.py`**: same architecture-agnostic loading.

All of the above were smoke-tested locally (tiny epoch counts, small window
caps) across all 8 architecture × pretext-task × window-size combinations
before being handed off here — see the commit history / this session's
transcript for the verification runs. No large-scale training has been run
on this data yet; that's what the runbook below is for.

## 3. Directory layout

```
checkpoints/
  pretrain_masked/{run_name}/          {architecture}_{pretext_task}_best.pt, _history.json
                                        (needs a per-combo subdir: pretrain's own filename doesn't
                                         encode window_len, so 2s/4s runs would otherwise collide)
  finetune_masked/                     {run_name}_best.pt, {run_name}_history.json
                                        (flat -- run_name itself encodes task+arch+window, so no
                                         subdirectory is needed here)
  predictions_masked/{run_name}/       per-timestep CSVs (prob, pred_raw, pred_post, label)
  plots/masked_predictions/{run_name}/ prediction-overlay PNGs
  masked_ablation_summary.csv          ONE shared file, one row per run_name (see §8)
```

## 4. Prerequisites on Ronin

1. Sync the repo (code) and the data this pipeline reads from:
   `data/interim_normalized/`, `data/processed/` (splits.json + index/*.parquet),
   `data/processed_w2s0.25/` (splits.json + index/*.parquet). These already
   exist locally — transfer them rather than regenerating (regenerating
   needs the raw datasets too, which is a separate, much larger transfer).
2. `pip install -e .` (or `pip install -e ".[dev]"` for pytest) in a fresh venv.
3. Confirm GPU is visible: `python -c "import torch; print(torch.cuda.is_available())"`
   should print `True`. Every script here auto-detects CUDA via `pick_device()`
   — no code changes needed for GPU vs. CPU/MPS.
4. `python -m pytest tests/ -q` — should show `130 passed` before spending any
   GPU time, exactly as it does locally.

## 5. Commands — pretraining (8 runs)

Epoch counts below are the new defaults; override with `--epochs` if GPU
throughput suggests going higher is cheap (see §7).

```bash
# 4s window (data/processed)
for task in channel span; do
  for arch in cnn bigru; do
    python -m chewing_ssl.ssl.masked.pretrain \
      --architecture $arch --pretext-task $task \
      --processed-root data/processed \
      --checkpoint-dir checkpoints/pretrain_masked/${task}_${arch}_w200 \
      --device cuda
  done
done

# 2s/25%-overlap window (data/processed_w2s0.25)
for task in channel span; do
  for arch in cnn bigru; do
    python -m chewing_ssl.ssl.masked.pretrain \
      --architecture $arch --pretext-task $task \
      --processed-root data/processed_w2s0.25 \
      --checkpoint-dir checkpoints/pretrain_masked/${task}_${arch}_w100 \
      --device cuda
  done
done
```

## 6. Commands — fine-tuning (8 runs)

```bash
for f in checkpoints/pretrain_masked/*/*_best.pt; do
  python -m chewing_ssl.ssl.masked.finetune \
    --checkpoint "$f" \
    --processed-root $(echo "$f" | grep -q w200 && echo data/processed || echo data/processed_w2s0.25) \
    --out-dir checkpoints/finetune_masked \
    --device cuda
done
```

(The script infers everything else — architecture, pretext task, window
size — from the checkpoint's own metadata, and writes
`checkpoints/finetune_masked/{run_name}_best.pt` automatically, so this loop
is safe to run over all 8 pretrain outputs without per-combo path juggling.
If your shell's glob/grep substitution above is awkward, it's equally fine
to just write out the 8 explicit calls, one per combination, each pointing
`--processed-root` at the matching window-size directory.)

## 7. Commands — inference, bout-level evaluation, and the comparison CSV

This is the "upscaling" step: `evaluate_bouts.py` runs the fine-tuned
detector over each full test recording, slides non-overlapping windows,
paints each window's per-timestep output into a continuous probability
trace for the whole recording, postprocesses it into discrete bouts, and
matches against ground truth. One shared `--summary-csv` accumulates a row
per combination.

```bash
for f in checkpoints/finetune_masked/*_best.pt; do
  run_name=$(basename "$f" _best.pt)
  proc_root=data/processed; [[ "$run_name" == *w100 ]] && proc_root=data/processed_w2s0.25
  python -m chewing_ssl.ssl.masked.evaluate_bouts \
    --checkpoint "$f" \
    --processed-root "$proc_root" \
    --pred-dir checkpoints/predictions_masked/$run_name \
    --summary-csv checkpoints/masked_ablation_summary.csv
done
```

Optionally, plot a representative recording per combination:

```bash
for f in checkpoints/finetune_masked/*_best.pt; do
  run_name=$(basename "$f" _best.pt)
  proc_root=data/processed; [[ "$run_name" == *w100 ]] && proc_root=data/processed_w2s0.25
  python -m chewing_ssl.ssl.masked.plot_predictions --all \
    --checkpoint "$f" --processed-root "$proc_root" \
    --out-dir checkpoints/plots/masked_predictions/$run_name
done
```

## 8. Reading `checkpoints/masked_ablation_summary.csv`

One row per combination, columns:

| Column | Meaning |
|---|---|
| `run_name`, `architecture`, `pretext_task`, `window_len` | Combination identity |
| `window_test_f1/precision/recall/auc` | Per-timestep test-set classification metrics (from fine-tuning's own held-out test evaluation) |
| `bout_iou_precision/recall/f1/median_iou` | Bout-level metrics, IoU≥0.3 matching |
| `bout_tolerance_precision/recall/f1/median_iou` | Bout-level metrics, ±5s onset/offset tolerance matching |
| `n_gt_bouts`, `n_pred_bouts` | Sanity totals — should read 944 GT bouts for the 4s runs (matches classical ML's count exactly, confirming the same held-out test population) |

Re-running `evaluate_bouts.py` for a `run_name` already in the CSV replaces
that row rather than duplicating it, so the file is always safe to
regenerate incrementally.

**Postprocessing caveat, carried over from the single completed 4s run**: the
hysteresis/duration/envelope-splitting parameters used here are the ones
classical ML converged on after its own separate tuning pass
(`docs/classical_ml.md` §6a–§6c) — not re-tuned specifically for this
8-way ablation. If a particular combination's bout-level numbers look
unexpectedly weak, that may reflect a postprocessing mismatch rather than a
genuinely worse encoder — worth a follow-up sweep once these 8 baseline
numbers exist.

## 9. What this doesn't cover

- No hyperparameter sweep over `--mask-channel-prob`/`--mask-span-ratio`,
  head width, or LR values — one fixed setting per axis being ablated
  (pretext task × architecture × window size), not a full grid.
- No re-tuning of postprocessing parameters per combination (see §8 caveat).
- Cross-sensor prediction (decision 001's second, still-unimplemented pretext
  task) and the causal/predictive (CPC-style) third pretext task discussed
  separately are both out of scope for this round.
