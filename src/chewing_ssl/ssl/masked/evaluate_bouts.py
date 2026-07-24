"""Bout-by-bout evaluation of fine-tuned chewing detection model.

For each test participant this script:
  1. Runs sliding-window inference over the full recording
  2. Saves per-timestep predictions to  checkpoints/predictions/<participant>.csv
  3. Extracts chewing bouts from both predictions and ground truth
  4. Matches predicted bouts to GT bouts using Intersection-over-Union (IoU)
  5. Prints per-participant and aggregate bout-level metrics

Bout-level metrics reported:
  Precision  = TP / (TP + FP)   — fraction of predicted bouts that are real
  Recall     = TP / (TP + FN)   — fraction of GT bouts that were detected
  F1         = harmonic mean of P and R
  Median IoU — overlap quality of matched pairs (1.0 = perfect temporal alignment)
  Onset lag  — median (predicted centre − GT centre) in seconds; negative = early
  Duration Δ — median (predicted duration − GT duration) in seconds

A bout is a contiguous block of timesteps labelled/predicted as "chewing".
Short fragments < min_bout_duration are discarded (default 0.5 s).
A GT bout and a predicted bout are considered a match when IoU ≥ iou_thresh (default 0.3).

Run from the project root:
  python -m chewing_ssl.ssl.masked.evaluate_bouts
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from chewing_ssl.common.data.preprocess import SENSOR_CHANNELS
from chewing_ssl.common.data.windowing import TARGET_HZ, WINDOW_LEN
from chewing_ssl.common.evaluation.bouts import extract_bouts, match_bouts, match_bouts_tolerant
from chewing_ssl.common.evaluation.postprocess import (
    compute_envelope,
    postprocess,
    remove_short_bouts,
    split_bouts_by_envelope,
)
from chewing_ssl.common.evaluation.reporting import fmt as _fmt
from chewing_ssl.common.evaluation.reporting import print_matched_pairs, print_summary_table
from chewing_ssl.ssl.masked.finetune import ChewingDetector
from chewing_ssl.ssl.models import ARCHITECTURES

FS = 50.0  # Hz
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
    encoder = ARCHITECTURES[architecture](n_channels=len(SENSOR_CHANNELS), **model_kwargs)
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


# ── Per-participant evaluation ─────────────────────────────────────────────────


def evaluate_participant(
    model: ChewingDetector,
    seg_path: Path,
    participant: str,
    pred_dir: Path,
    iou_thresh: float,
    min_bout_dur: float,
    threshold: float,
    post_high: float,
    post_low: float,
    post_max_gap_s: float,
    post_min_dur_s: float,
    window_len: int = WINDOW_LEN,
    match_mode: str = "iou",
    tolerance_s: float = 5.0,
    post_max_dur_s: float | None = None,
    post_valley: float | None = None,
    envelope_split: bool = False,
    envelope_window_s: float = 1.5,
    envelope_min_piece_dur_s: float = 8.0,
    envelope_prominence_frac: float = 2.5,
) -> dict | None:
    if not seg_path.exists():
        print(f"  [skip] segment not found: {seg_path}")
        return None

    df = pd.read_parquet(seg_path)
    if "label" not in df.columns:
        print(f"  [skip] no label column in {seg_path}")
        return None

    x = df[SENSOR_CHANNELS].to_numpy(dtype=np.float32).T  # (C, N)
    y = df["label"].to_numpy(dtype=np.float32)  # (N,)

    probs = infer_recording(model, x, window_len=window_len)

    # Post-process: hysteresis → gap fill → [duration/valley split] → min duration,
    # same pipeline and (by default) same tuned parameters as
    # chewing_ssl.classical_ml.evaluate_bouts, for a direct comparison.
    preds_post = postprocess(
        probs,
        high=post_high,
        low=post_low,
        max_gap_s=post_max_gap_s,
        min_dur_s=post_min_dur_s,
        max_dur_s=post_max_dur_s,
        valley_thresh=post_valley,
    )
    if envelope_split:
        signal_mag = np.sqrt(
            (df[ENVELOPE_CHANNELS].to_numpy(dtype=np.float64) ** 2).sum(axis=1)
        )
        envelope = compute_envelope(signal_mag, fs=FS, window_s=envelope_window_s)
        preds_post = split_bouts_by_envelope(
            preds_post,
            envelope,
            fs=FS,
            min_piece_dur_s=envelope_min_piece_dur_s,
            prominence_frac=envelope_prominence_frac,
        )
        preds_post = remove_short_bouts(preds_post, fs=FS, min_dur_s=post_min_dur_s)
    preds_post = preds_post.astype(np.float32)

    # Raw threshold prediction (for CSV)
    preds_raw = (probs >= threshold).astype(np.float32)
    preds_raw[np.isnan(probs)] = 0.0

    # ── Save per-timestep predictions ──
    # participant may be "dataset/tag[/segment_NN]" (main()'s recording label),
    # which doubles as a safe relative subpath -- write_text-style flatten
    # would collide across datasets that reuse participant ids otherwise.
    out_csv = pred_dir / f"{participant}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time_s", "prob", "pred_raw", "pred_post", "label"])
        for i, (pr, raw, post, gt) in enumerate(zip(probs, preds_raw, preds_post, y)):
            writer.writerow(
                [f"{i / FS:.4f}", "" if np.isnan(pr) else f"{pr:.6f}", int(raw), int(post), int(gt)]
            )

    # ── Bout extraction (on post-processed predictions) ──
    gt_bouts = extract_bouts(y.astype(bool), min_dur=min_bout_dur)
    pred_bouts = extract_bouts(preds_post.astype(bool), min_dur=min_bout_dur)

    # Compute BOTH match modes always (cheap -- bout lists are short), so the
    # results summary CSV always has both regardless of which mode drives
    # this run's console table.
    iou_result = match_bouts(gt_bouts, pred_bouts, iou_thresh=iou_thresh)
    tol_result = match_bouts_tolerant(gt_bouts, pred_bouts, tolerance_s=tolerance_s)
    result = tol_result if match_mode == "tolerance" else iou_result

    def _stats(r: dict) -> dict:
        tp, fp, fn = r["tp"], r["fp"], r["fn"]
        prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        rec = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else float("nan")
        ious = [m["iou"] for m in r["matches"]]
        onset_lag = [(m["pred"]["start_s"] - m["gt"]["start_s"]) for m in r["matches"]]
        dur_delta = [(m["pred"]["duration_s"] - m["gt"]["duration_s"]) for m in r["matches"]]
        return {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "median_iou": float(np.median(ious)) if ious else float("nan"),
            "median_onset_lag_s": float(np.median(onset_lag)) if onset_lag else float("nan"),
            "median_dur_delta_s": float(np.median(dur_delta)) if dur_delta else float("nan"),
        }

    displayed = _stats(result)
    summary = {
        "participant": participant,
        "n_gt_bouts": len(gt_bouts),
        "n_pred_bouts": len(pred_bouts),
        **displayed,
        "matches": result["matches"],
        "iou_stats": _stats(iou_result),
        "tolerance_stats": _stats(tol_result),
    }
    return summary


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
    parser.add_argument(
        "--match-mode",
        choices=["iou", "tolerance"],
        default="iou",
        help="How a predicted bout is credited against a ground-truth bout (default: iou, "
        "chewing_ssl.classical_ml.evaluate_bouts's 'best-measured combination' when paired "
        "with duration-bounded postprocessing; that script itself defaults to tolerance).",
    )
    parser.add_argument(
        "--tolerance-s",
        type=float,
        default=5.0,
        help="--match-mode tolerance: max onset/offset deviation in seconds (default 5.0).",
    )
    parser.add_argument(
        "--iou-thresh",
        type=float,
        default=0.3,
        help="--match-mode iou: minimum IoU to count a predicted bout as a TP (default 0.3)",
    )
    parser.add_argument(
        "--min-bout-dur",
        type=float,
        default=0.5,
        help="Minimum bout duration in seconds for bout matching (default 0.5)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Raw probability threshold saved to CSV (default 0.5)",
    )
    # Post-processing args -- defaults match chewing_ssl.classical_ml.evaluate_bouts's
    # tuned values (docs/classical_ml.md §6c) for a direct comparison.
    parser.add_argument(
        "--post-high",
        type=float,
        default=0.75,
        help="Hysteresis upper threshold — starts a bout (default 0.75)",
    )
    parser.add_argument(
        "--post-low",
        type=float,
        default=0.52,
        help="Hysteresis lower threshold — ends a bout (default 0.52)",
    )
    parser.add_argument(
        "--post-max-gap",
        type=float,
        default=1.0,
        help="Maximum gap in seconds to bridge between nearby bouts (default 1.0)",
    )
    parser.add_argument(
        "--post-max-dur",
        type=float,
        default=35.0,
        help="Split postprocessed bouts longer than this at their lowest-probability point "
        "(default 35.0s; real bite bouts run ~3-17s). Pass a negative value to disable.",
    )
    parser.add_argument(
        "--post-valley",
        type=float,
        default=0.35,
        help="Split postprocessed bouts at any interior probability dip below this, "
        "regardless of duration (default 0.35). Pass a negative value to disable.",
    )
    parser.add_argument(
        "--envelope-split",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Additionally split bouts using a raw-signal (nav_x/nav_y) activity envelope "
        "-- see docs/classical_ml.md §6b/§6c. Default on.",
    )
    parser.add_argument("--envelope-window-s", type=float, default=1.5)
    parser.add_argument("--envelope-min-piece-dur", type=float, default=8.0)
    parser.add_argument("--envelope-prominence", type=float, default=2.5)
    parser.add_argument(
        "--post-min-dur",
        type=float,
        default=2.0,
        help="Minimum bout duration in seconds after post-processing (default 2.0)",
    )
    parser.add_argument(
        "--show-pairs",
        action="store_true",
        help="Also print individual matched pairs per participant",
    )
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
    args = parser.parse_args()

    if not args.checkpoint.exists():
        print(f"Checkpoint not found: {args.checkpoint}")
        return

    post_max_dur = args.post_max_dur if args.post_max_dur >= 0 else None
    post_valley = args.post_valley if args.post_valley >= 0 else None

    # ── Load model (architecture/pretext_task/window_len auto-detected) ──
    model, meta = load_finetuned_model(args.checkpoint)
    model.eval()
    run_name = f"{meta['pretext_task']}_{meta['architecture']}_w{meta['window_len']}"
    window_len = int(args.window_seconds * TARGET_HZ) if args.window_seconds else meta["window_len"]

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
        f"  Post-process : hysteresis ({args.post_high}/{args.post_low})  "
        f"gap-fill {args.post_max_gap}s  dur=[{args.post_min_dur}, {post_max_dur}]s  "
        f"valley={post_valley}  envelope-split={args.envelope_split}"
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
                model,
                seg_path,
                label,
                args.pred_dir,
                iou_thresh=args.iou_thresh,
                min_bout_dur=args.min_bout_dur,
                threshold=args.threshold,
                post_high=args.post_high,
                post_low=args.post_low,
                post_max_gap_s=args.post_max_gap,
                post_min_dur_s=args.post_min_dur,
                window_len=window_len,
                match_mode=args.match_mode,
                tolerance_s=args.tolerance_s,
                post_max_dur_s=post_max_dur,
                post_valley=post_valley,
                envelope_split=args.envelope_split,
                envelope_window_s=args.envelope_window_s,
                envelope_min_piece_dur_s=args.envelope_min_piece_dur,
                envelope_prominence_frac=args.envelope_prominence,
            )
            if r is not None:
                r["_iou_thresh"] = args.iou_thresh
                r["_tolerance_s"] = args.tolerance_s
                r["_min_dur"] = args.min_bout_dur
                results.append(r)
                print(
                    f"GT={r['n_gt_bouts']} bouts  Pred={r['n_pred_bouts']} bouts  "
                    f"TP={r['tp']} FP={r['fp']} FN={r['fn']}  F1={_fmt(r['f1'])}"
                )

    if results:
        print_summary_table(results, match_mode=args.match_mode)
        if args.show_pairs:
            print_matched_pairs(results)
        print(f"\n  Per-timestep predictions saved to: {args.pred_dir}/")

        if args.summary_csv:
            row = build_summary_row(run_name, meta, args.checkpoint, results)
            append_summary_row(args.summary_csv, row)
            print(f"  Results summary row  →  {args.summary_csv}  ({row['run_name']})")

    print()


# ── Results summary CSV (one row per architecture/pretext-task/window combo) ──


def _aggregate(results: list[dict], key: str) -> dict:
    """Sum tp/fp/fn across every recording's results[key] (either 'iou_stats'
    or 'tolerance_stats') and recompute overall precision/recall/f1 -- the
    same aggregation print_summary_table does, just for a specific mode."""
    tp = sum(r[key]["tp"] for r in results)
    fp = sum(r[key]["fp"] for r in results)
    fn = sum(r[key]["fn"] for r in results)
    prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    rec = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else float("nan")
    ious = [r[key]["median_iou"] for r in results if not np.isnan(r[key]["median_iou"])]
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "median_iou": float(np.median(ious)) if ious else float("nan"),
    }


def build_summary_row(run_name: str, meta: dict, checkpoint: Path, results: list[dict]) -> dict:
    iou_agg = _aggregate(results, "iou_stats")
    tol_agg = _aggregate(results, "tolerance_stats")

    # Window-level test metrics come from finetune.py's history JSON, which
    # sits alongside the checkpoint under the same run_name -- best-effort,
    # since it's an optional cross-reference, not required for the bout-level
    # numbers this script itself computed.
    window_metrics = {}
    hist_path = checkpoint.parent / f"{run_name}_history.json"
    if hist_path.exists():
        hist = json.loads(hist_path.read_text())
        window_metrics = hist.get("test_metrics", {})

    return {
        "run_name": run_name,
        "architecture": meta["architecture"],
        "pretext_task": meta["pretext_task"],
        "window_len": meta["window_len"],
        "checkpoint": str(checkpoint),
        "n_gt_bouts": sum(r["n_gt_bouts"] for r in results),
        "n_pred_bouts": sum(r["n_pred_bouts"] for r in results),
        "window_test_f1": window_metrics.get("f1"),
        "window_test_precision": window_metrics.get("precision"),
        "window_test_recall": window_metrics.get("recall"),
        "window_test_auc": window_metrics.get("auc"),
        "bout_iou_precision": iou_agg["precision"],
        "bout_iou_recall": iou_agg["recall"],
        "bout_iou_f1": iou_agg["f1"],
        "bout_iou_median_iou": iou_agg["median_iou"],
        "bout_tolerance_precision": tol_agg["precision"],
        "bout_tolerance_recall": tol_agg["recall"],
        "bout_tolerance_f1": tol_agg["f1"],
        "bout_tolerance_median_iou": tol_agg["median_iou"],
    }


def append_summary_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df_row = pd.DataFrame([row])
    if path.exists():
        existing = pd.read_csv(path)
        existing = existing[existing["run_name"] != row["run_name"]]  # replace stale re-runs
        df_row = pd.concat([existing, df_row], ignore_index=True)
    df_row.to_csv(path, index=False)


if __name__ == "__main__":
    main()
