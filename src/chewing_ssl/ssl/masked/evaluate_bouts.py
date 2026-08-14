"""Bout-by-bout evaluation of fine-tuned chewing detection model.

For each test participant this script:
  1. Runs sliding-window inference over the full recording (the ONE
     pipeline-specific step -- everything after this is shared code)
  2. Saves per-timestep predictions to  checkpoints/predictions/<participant>.csv
  3. Postprocesses the probability trace and extracts chewing bouts from
     both predictions and ground truth
  4. Matches predicted bouts to GT bouts under BOTH IoU and onset/offset-
     tolerance matching (always both -- see --summary-csv)
  5. Prints per-participant and aggregate bout-level metrics, and (--out-csv)
     writes a per-bite ledger

Bout-level metrics reported:
  Precision  = TP / (TP + FP)   — fraction of predicted bouts that are real
  Recall     = TP / (TP + FN)   — fraction of GT bouts that were detected
  F1         = harmonic mean of P and R
  Median IoU — overlap quality of matched pairs (1.0 = perfect temporal alignment)
  Onset lag  — median (predicted centre − GT centre) in seconds; negative = early
  Duration Δ — median (predicted duration − GT duration) in seconds

Postprocessing (applying the shared recipe, matching bouts, aggregating
results, building the ledger/summary row) is shared code
(`chewing_ssl.common.evaluation.postprocess`/`pipeline`), identical to what
`chewing_ssl.classical_ml.evaluate_bouts` and
`chewing_ssl.common.evaluation.compare_bites` use, so the two pipelines'
numbers are directly comparable. The one deliberate difference:
`--smooth-window-s` defaults to 2.0s here (vs. classical ML's 0.0) --
this script's per-timestep probability has no equivalent to classical ML's
overlap-averaging smoothing, and empirically (see
`chewing_ssl.common.evaluation.compare_bites`'s module docstring) a 2.0s
moving average closes a real gap: F1 0.674->0.731 (IoU mode), 0.489->0.527
(tolerance mode) on span_bigru_w200.

Run from the project root:
  python -m chewing_ssl.ssl.masked.evaluate_bouts --checkpoint <path>
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from chewing_ssl.common.data.preprocess import SENSOR_CHANNELS
from chewing_ssl.common.data.windowing import TARGET_HZ, WINDOW_LEN
from chewing_ssl.common.evaluation.pipeline import (
    add_match_args,
    add_postprocess_args,
    aggregate_alignment,
    aggregate_dual_mode,
    append_summary_row,
    build_summary_row,
    evaluate_recording,
    ledger_rows_for_recording,
    postprocess_config_from_args,
)
from chewing_ssl.common.evaluation.postprocess import FS, compute_envelope
from chewing_ssl.common.evaluation.reporting import print_matched_pairs, print_summary_table
from chewing_ssl.common.evaluation.reporting import print_alignment_summary, print_worst_offenders
from chewing_ssl.common.evaluation.reporting import capture_console_report, save_console_report
from chewing_ssl.ssl.masked.finetune import ChewingDetector
from chewing_ssl.ssl.models import ARCHITECTURES

ENVELOPE_CHANNELS = ["nav_x_left_temple", "nav_y_left_temple"]


def load_finetuned_model(path: Path) -> tuple[ChewingDetector, dict]:
    """Load a finetune.py checkpoint, auto-detecting architecture/embedding_dim/
    head shape from the metadata finetune.py stamps into every checkpoint.
    Requires the new checkpoint format (see finetune.py's load_pretrained_encoder)."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    required = ("architecture", "embedding_dim", "model_state_dict")
    missing = [k for k in required if k not in state]
    if missing:
        raise ValueError(
            f"{path} is missing {missing} -- looks like a pre-ablation-study checkpoint. "
            "Re-run finetune.py to regenerate it."
        )
    architecture = state["architecture"]
    model_kwargs = state.get("model_kwargs", {})
    # Older checkpoints (saved before channel-subset training existed) don't
    # have a "channels" key -- fall back to the full SENSOR_CHANNELS, which
    # is what they were always trained on.
    channels = state.get("channels", SENSOR_CHANNELS)
    encoder = ARCHITECTURES[architecture](n_channels=len(channels), **model_kwargs)
    model = ChewingDetector(
        encoder,
        architecture,
        state["embedding_dim"],
        state.get("head_hidden_dim", 64),
        state.get("head_dropout", 0.3),
    )
    model.load_state_dict(state["model_state_dict"])
    meta = {
        "architecture": architecture,
        "pretext_task": state.get("pretext_task", "unknown"),
        "window_len": state.get("window_len", WINDOW_LEN),
        "val_f1": state.get("val_f1", float("nan")),
        "phase": state.get("phase"),
        "epoch": state.get("epoch"),
        "channels": channels,
    }
    return model, meta


# ── Inference ──────────────────────────────────────────────────────────────────


@torch.no_grad()
def infer_recording(
    model: ChewingDetector, x_full: np.ndarray, window_len: int = WINDOW_LEN
) -> np.ndarray:
    """Slide non-overlapping window_len windows over (C, N) array.

    window_len must match whatever window length the checkpoint was
    fine-tuned on (--window-seconds), not necessarily configs/preprocessing.yaml's
    default -- e.g. a model fine-tuned on 2s windows needs window_len=100, not
    WINDOW_LEN's 4s-derived 200.

    Returns per-timestep probability array of length N (NaN for uncovered tail).
    """
    C, N = x_full.shape
    probs = np.full(N, np.nan, dtype=np.float32)
    for start in range(0, N - window_len + 1, window_len):
        xw = torch.from_numpy(x_full[:, start : start + window_len]).unsqueeze(0)
        logits = model(xw).squeeze(0)
        probs[start : start + window_len] = torch.sigmoid(logits).numpy()
    return probs


def load_test_recordings(
    model: ChewingDetector,
    interim_root: Path,
    processed_root: Path,
    window_len: int,
    envelope_window_s: float = 1.5,
    dataset: str | None = None,
    channels: list[str] = SENSOR_CHANNELS,
) -> list[dict]:
    """Run inference once per `finetune_test` recording, returning
    `{dataset, tag, segment, participant, gt, probs, envelope}` dicts -- the
    expensive, postprocessing-INDEPENDENT step (model inference), factored
    out of `evaluate_participant` so a postprocessing sweep
    (`chewing_ssl.common.evaluation.sweep`) can reuse these same cached
    arrays across many `PostprocessConfig` combos without re-running
    inference per combo. Deliberately skips per-timestep CSV writing and any
    particular postprocessing config -- callers that also want those (e.g.
    `evaluate_participant`) still do that themselves."""
    idx = pd.read_parquet(processed_root / "index" / "finetune_test.parquet")
    if dataset:
        idx = idx[idx["dataset"] == dataset]
    recs = sorted(idx[["dataset", "tag"]].drop_duplicates().itertuples(index=False))

    recordings = []
    for ds, tag in recs:
        seg_dir = interim_root / ds / tag
        for seg_path in sorted(seg_dir.glob("segment_*.parquet")):
            df = pd.read_parquet(seg_path)
            if "label" not in df.columns:
                continue
            x = df[channels].to_numpy(dtype=np.float32).T
            y = df["label"].to_numpy(dtype=np.float32)
            probs = infer_recording(model, x, window_len=window_len)
            signal_mag = np.sqrt((df[ENVELOPE_CHANNELS].to_numpy(dtype=np.float64) ** 2).sum(axis=1))
            envelope = compute_envelope(signal_mag, fs=FS, window_s=envelope_window_s)
            recordings.append(
                {
                    "dataset": ds,
                    "tag": tag,
                    "segment": seg_path.stem,
                    "participant": f"{ds}/{tag}/{seg_path.stem}",
                    "gt": y.astype(bool),
                    "probs": probs,
                    "envelope": envelope,
                }
            )
    return recordings


# ── Per-participant evaluation ─────────────────────────────────────────────────


def evaluate_participant(
    model: ChewingDetector,
    seg_path: Path,
    participant: str,
    pred_dir: Path,
    cfg,
    min_bout_dur: float,
    match_mode: str,
    iou_thresh: float,
    tolerance_s: float,
    window_len: int = WINDOW_LEN,
    threshold: float = 0.5,
    channels: list[str] = SENSOR_CHANNELS,
) -> dict | None:
    if not seg_path.exists():
        print(f"  [skip] segment not found: {seg_path}")
        return None

    df = pd.read_parquet(seg_path)
    if "label" not in df.columns:
        print(f"  [skip] no label column in {seg_path}")
        return None

    x = df[channels].to_numpy(dtype=np.float32).T  # (C, N)
    y = df["label"].to_numpy(dtype=np.float32)  # (N,)
    probs = infer_recording(model, x, window_len=window_len)

    envelope = None
    if cfg.envelope_split:
        signal_mag = np.sqrt((df[ENVELOPE_CHANNELS].to_numpy(dtype=np.float64) ** 2).sum(axis=1))
        envelope = compute_envelope(signal_mag, fs=FS, window_s=cfg.envelope_window_s)

    rec = {"participant": participant, "gt": y.astype(bool), "probs": probs, "envelope": envelope}
    result = evaluate_recording(rec, cfg, min_bout_dur, match_mode, iou_thresh, tolerance_s)

    # ── Save per-timestep predictions ──
    # participant may be "dataset/tag[/segment_NN]" (main()'s recording label),
    # which doubles as a safe relative subpath -- write_text-style flatten
    # would collide across datasets that reuse participant ids otherwise.
    preds_raw = (np.nan_to_num(probs, nan=0.0) >= threshold).astype(np.float32)
    preds_post = result["binary"].astype(np.float32)
    out_csv = pred_dir / f"{participant}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time_s", "prob", "pred_raw", "pred_post", "label"])
        for i, (pr, raw, post, gt) in enumerate(zip(probs, preds_raw, preds_post, y)):
            writer.writerow(
                [f"{i / FS:.4f}", "" if np.isnan(pr) else f"{pr:.6f}", int(raw), int(post), int(gt)]
            )

    result["_iou_thresh"] = iou_thresh
    result["_tolerance_s"] = tolerance_s
    result["_min_dur"] = min_bout_dur
    return result


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Bout-by-bout evaluation of fine-tuned model.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/finetune/bigru_finetune_best.pt"),
    )
    parser.add_argument(
        "--interim-root",
        type=Path,
        default=Path("data/interim_normalized"),
        help="Must be the NORMALIZED tree (not data/interim) -- the model was trained on "
        "globally z-scored input; feeding raw-scale data here silently produces garbage "
        "predictions rather than an error.",
    )
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--dataset", default=None, help="Restrict to one dataset. Default: all datasets."
    )
    parser.add_argument("--pred-dir", type=Path, default=Path("checkpoints/predictions"))
    parser.add_argument("--threshold", type=float, default=0.5, help="Raw probability threshold saved to CSV.")
    add_postprocess_args(parser)
    # Override the shared --smooth-window-s default (0.0) for this pipeline --
    # see module docstring for why 2.0s is the validated default here specifically.
    for action in parser._actions:
        if action.dest == "smooth_window_s":
            action.default = 2.0
            action.help = (
                "Moving-average window (s) applied to probs before hysteresis. "
                "Default 2.0s for THIS pipeline (validated -- see module docstring); "
                "pass 0.0 to disable. Classical ML's own default is 0.0 (already "
                "smoothed by overlapping-window averaging)."
            )
    add_match_args(parser, default_match_mode="iou")
    parser.add_argument(
        "--show-pairs", action="store_true", help="Also print individual matched pairs per participant"
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Write the full per-bite ledger (one row per GT/predicted bout, tagged "
        "detected/missed/false_alarm) here, e.g. checkpoints/masked_bout_ledger.csv. "
        "Default: skip (kept opt-in since this script already saves full per-timestep "
        "predictions to --pred-dir).",
    )
    parser.add_argument("--top-n", type=int, default=15, help="Rows in the worst-offenders console table (--out-csv only).")
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=None,
        help="Override the window length auto-detected from the checkpoint (rarely needed).",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("checkpoints/masked_ablation_summary.csv"),
        help="Append one row (combo metadata + window-level test metrics from the finetune "
        "history JSON, if found + bout-level metrics under BOTH match modes) here every run -- "
        "the single file comparing every architecture/pretext-task/window-size combination. "
        "Pass an empty string to skip.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=None,
        help="Save the exact console report (per-recording table, glossary, alignment "
        "breakdown, worst offenders) to this markdown file. Default: "
        "<--pred-dir>/evaluation_report.md (kept alongside that run's own predictions, so "
        "it can't collide with a different checkpoint's report). Pass an empty string to skip.",
    )
    args = parser.parse_args()

    if not args.checkpoint.exists():
        print(f"Checkpoint not found: {args.checkpoint}")
        return

    report_path = args.report_path
    if report_path is None:
        report_path = args.pred_dir / "evaluation_report.md"

    cfg = postprocess_config_from_args(args)

    # ── Load model (architecture/pretext_task/window_len auto-detected) ──
    model, meta = load_finetuned_model(args.checkpoint)
    model.eval()
    run_name = f"{meta['pretext_task']}_{meta['architecture']}_w{meta['window_len']}"
    window_len = int(args.window_seconds * TARGET_HZ) if args.window_seconds else meta["window_len"]

    report_ctx = capture_console_report() if report_path else contextlib.nullcontext(None)
    with report_ctx as report_buffer:
        _run_evaluation(args, cfg, model, meta, run_name, window_len)

    if report_path and report_buffer is not None:
        save_console_report(report_buffer, report_path, title=f"Bout-by-Bout Evaluation -- {run_name}")
        print(f"  Full console report  →  {report_path}")


def _run_evaluation(args, cfg, model, meta, run_name, window_len) -> None:
    print("\nBout-by-Bout Evaluation")
    print("=" * 60)
    print(f"  Checkpoint   : {args.checkpoint}  [{run_name}]")
    print(
        f"  Architecture : {meta['architecture']}  |  Pretext task: {meta['pretext_task']}  |  "
        f"Window: {window_len} samples"
    )
    print(
        f"  Match mode   : {args.match_mode} "
        f"({'tolerance=' + str(args.tolerance_s) + 's' if args.match_mode == 'tolerance' else 'IoU>=' + str(args.iou_thresh)})"
    )
    print(
        f"  Post-process : hysteresis ({cfg.high}/{cfg.low})  gap-fill {cfg.max_gap_s}s  "
        f"dur=[{cfg.min_dur_s}, {cfg.max_dur_s}]s  valley={cfg.valley_thresh}  "
        f"envelope-split={cfg.envelope_split}  smooth={cfg.smooth_window_s}s"
    )
    print(f"  Predictions  → {args.pred_dir}/")
    print(f"  Loaded       : phase={meta['phase']}  epoch={(meta['epoch'] or 0) + 1}  val_f1={meta['val_f1']:.3f}\n")

    # ── Resolve test recordings (dataset, tag) from the window index itself --
    # more robust than regex-extracting a participant id from segment_path,
    # and correctly covers every dataset, not just one hardcoded name. ──
    idx = pd.read_parquet(args.processed_root / "index" / "finetune_test.parquet")
    if args.dataset:
        idx = idx[idx["dataset"] == args.dataset]
    recordings = sorted(idx[["dataset", "tag"]].drop_duplicates().itertuples(index=False))
    print(f"  Test recordings: {len(recordings)}  (datasets: {sorted(idx['dataset'].unique())})\n")

    results = []
    for dataset, tag in recordings:
        # A recording can be split into more than one gap-bounded segment
        # (chewing_ssl.common.data.preprocess.segment_at_gaps) -- evaluate each
        # segment independently rather than hardcoding segment_00, since
        # concatenating segments across a real device-dropout gap would let a
        # predicted bout spuriously bridge it.
        seg_dir = args.interim_root / dataset / tag
        seg_paths = sorted(seg_dir.glob("segment_*.parquet"))
        for seg_path in seg_paths:
            label = f"{dataset}/{tag}" + (f"/{seg_path.stem}" if len(seg_paths) > 1 else "")
            print(f"  Processing {label} ...", end=" ", flush=True)
            r = evaluate_participant(
                model, seg_path, label, args.pred_dir, cfg,
                min_bout_dur=args.min_bout_dur, match_mode=args.match_mode,
                iou_thresh=args.iou_thresh, tolerance_s=args.tolerance_s,
                window_len=window_len, threshold=args.threshold, channels=meta["channels"],
            )
            if r is not None:
                r["dataset"], r["tag"], r["segment"] = dataset, tag, seg_path.stem
                results.append(r)
                print(
                    f"GT={r['n_gt_bouts']} bouts  Pred={r['n_pred_bouts']} bouts  "
                    f"TP={r['tp']} FP={r['fp']} FN={r['fn']}  F1={r['f1']:.3f}"
                )

    if results:
        print_summary_table(results, match_mode=args.match_mode)
        if args.show_pairs:
            print_matched_pairs(results)
        print_alignment_summary(aggregate_alignment(results))
        print(f"\n  Per-timestep predictions saved to: {args.pred_dir}/")

        if args.out_csv:
            ledger_rows = []
            for r in results:
                ident = {k: r[k] for k in ("dataset", "tag", "segment")}
                ident["participant"] = r["tag"]  # same convention as classical_ml's ledger
                ledger_rows += ledger_rows_for_recording(ident, r, args.match_mode)
            ledger = pd.DataFrame(ledger_rows)
            print_worst_offenders(ledger, top_n=args.top_n)
            args.out_csv.parent.mkdir(parents=True, exist_ok=True)
            ledger.to_csv(args.out_csv, index=False)
            print(f"\n  Full per-bite ledger ({len(ledger)} rows) -> {args.out_csv}")

        if args.summary_csv:
            dual = aggregate_dual_mode(results)
            window_metrics = None
            hist_path = args.checkpoint.parent / f"{run_name}_history.json"
            if hist_path.exists():
                hist = json.loads(hist_path.read_text())
                window_metrics = hist.get("test_metrics", {})
            identity = {
                "run_name": run_name,
                "architecture": meta["architecture"],
                "pretext_task": meta["pretext_task"],
                "window_len": meta["window_len"],
                "checkpoint": str(args.checkpoint),
            }
            row = build_summary_row(identity, dual, window_metrics)
            append_summary_row(args.summary_csv, row)
            print(f"  Results summary row  →  {args.summary_csv}  ({row['run_name']})")

    print()


if __name__ == "__main__":
    main()
