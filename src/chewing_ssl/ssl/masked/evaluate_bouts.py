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
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from chewing_ssl.common.data.preprocess import SENSOR_CHANNELS
from chewing_ssl.common.data.windowing import WINDOW_LEN
from chewing_ssl.common.evaluation.bouts import extract_bouts, match_bouts
from chewing_ssl.common.evaluation.postprocess import postprocess
from chewing_ssl.ssl.masked.finetune import ChewingDetector

FS = 50.0  # Hz


# ── Inference ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def infer_recording(model: ChewingDetector, x_full: np.ndarray) -> np.ndarray:
    """Slide non-overlapping WINDOW_LEN windows over (C, N) array.

    Returns per-timestep probability array of length N (NaN for uncovered tail).
    """
    C, N = x_full.shape
    probs = np.full(N, np.nan, dtype=np.float32)
    for start in range(0, N - WINDOW_LEN + 1, WINDOW_LEN):
        xw = torch.from_numpy(x_full[:, start:start + WINDOW_LEN]).unsqueeze(0)
        logits = model(xw).squeeze(0)
        probs[start:start + WINDOW_LEN] = torch.sigmoid(logits).numpy()
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
) -> dict | None:
    if not seg_path.exists():
        print(f"  [skip] segment not found: {seg_path}")
        return None

    df = pd.read_parquet(seg_path)
    if "label" not in df.columns:
        print(f"  [skip] no label column in {seg_path}")
        return None

    x = df[SENSOR_CHANNELS].to_numpy(dtype=np.float32).T  # (C, N)
    y = df["label"].to_numpy(dtype=np.float32)             # (N,)

    probs = infer_recording(model, x)

    # Post-process: hysteresis → gap fill → min duration
    preds_post = postprocess(
        probs,
        high=post_high, low=post_low,
        max_gap_s=post_max_gap_s, min_dur_s=post_min_dur_s,
    ).astype(np.float32)

    # Raw threshold prediction (for CSV)
    preds_raw = (probs >= threshold).astype(np.float32)
    preds_raw[np.isnan(probs)] = 0.0

    # ── Save per-timestep predictions ──
    pred_dir.mkdir(parents=True, exist_ok=True)
    out_csv = pred_dir / f"{participant}.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time_s", "prob", "pred_raw", "pred_post", "label"])
        for i, (pr, raw, post, gt) in enumerate(zip(probs, preds_raw, preds_post, y)):
            writer.writerow([f"{i / FS:.4f}",
                             "" if np.isnan(pr) else f"{pr:.6f}",
                             int(raw), int(post), int(gt)])

    # ── Bout extraction (on post-processed predictions) ──
    gt_bouts   = extract_bouts(y.astype(bool),          min_dur=min_bout_dur)
    pred_bouts = extract_bouts(preds_post.astype(bool),  min_dur=min_bout_dur)

    result = match_bouts(gt_bouts, pred_bouts, iou_thresh=iou_thresh)
    tp, fp, fn = result["tp"], result["fp"], result["fn"]

    prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    rec  = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else float("nan")

    matched = result["matches"]
    ious      = [m["iou"]                                            for m in matched]
    onset_lag = [(m["pred"]["start_s"] - m["gt"]["start_s"])        for m in matched]
    dur_delta = [(m["pred"]["duration_s"] - m["gt"]["duration_s"])  for m in matched]

    summary = {
        "participant":   participant,
        "n_gt_bouts":    len(gt_bouts),
        "n_pred_bouts":  len(pred_bouts),
        "tp": tp, "fp": fp, "fn": fn,
        "precision":     prec,
        "recall":        rec,
        "f1":            f1,
        "median_iou":    float(np.median(ious))      if ious      else float("nan"),
        "median_onset_lag_s":  float(np.median(onset_lag)) if onset_lag else float("nan"),
        "median_dur_delta_s":  float(np.median(dur_delta))  if dur_delta  else float("nan"),
        "gt_bouts":      gt_bouts,
        "pred_bouts":    pred_bouts,
        "matches":       matched,
    }
    return summary


# ── Printing ───────────────────────────────────────────────────────────────────

def _fmt(v, fmt=".3f"):
    return "nan" if (isinstance(v, float) and np.isnan(v)) else format(v, fmt)


def print_summary_table(results: list[dict]) -> None:
    hdr = (f"{'Part':<6}  {'GT':>4}  {'Pred':>4}  {'TP':>3}  {'FP':>3}  {'FN':>3}  "
           f"{'Prec':>6}  {'Rec':>6}  {'F1':>6}  {'mIoU':>6}  "
           f"{'Onset lag':>10}  {'Dur Δ':>8}")
    print("\n" + "=" * len(hdr))
    print("  Bout-by-bout evaluation  (IoU threshold = {:.2f},  min bout = {:.1f}s)".format(
        results[0].get("_iou_thresh", 0.3), results[0].get("_min_dur", 0.5)
    ))
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))

    agg = {"tp": 0, "fp": 0, "fn": 0, "ious": [], "onset": [], "dur": []}
    for r in results:
        agg["tp"] += r["tp"]; agg["fp"] += r["fp"]; agg["fn"] += r["fn"]
        agg["ious"]  += [m["iou"]                                          for m in r["matches"]]
        agg["onset"] += [(m["pred"]["start_s"] - m["gt"]["start_s"])      for m in r["matches"]]
        agg["dur"]   += [(m["pred"]["duration_s"] - m["gt"]["duration_s"]) for m in r["matches"]]

        print(f"  {r['participant']:<6}  {r['n_gt_bouts']:>4}  {r['n_pred_bouts']:>4}  "
              f"{r['tp']:>3}  {r['fp']:>3}  {r['fn']:>3}  "
              f"{_fmt(r['precision']):>6}  {_fmt(r['recall']):>6}  {_fmt(r['f1']):>6}  "
              f"{_fmt(r['median_iou']):>6}  "
              f"{_fmt(r['median_onset_lag_s'], '+.2f'):>10}s  "
              f"{_fmt(r['median_dur_delta_s'], '+.2f'):>7}s")

    print("-" * len(hdr))
    tp, fp, fn = agg["tp"], agg["fp"], agg["fn"]
    prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    rec  = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else float("nan")
    m_iou    = float(np.median(agg["ious"]))   if agg["ious"]  else float("nan")
    m_onset  = float(np.median(agg["onset"]))  if agg["onset"] else float("nan")
    m_dur    = float(np.median(agg["dur"]))    if agg["dur"]   else float("nan")

    total_gt   = sum(r["n_gt_bouts"]   for r in results)
    total_pred = sum(r["n_pred_bouts"] for r in results)

    print(f"  {'TOTAL':<6}  {total_gt:>4}  {total_pred:>4}  "
          f"{tp:>3}  {fp:>3}  {fn:>3}  "
          f"{_fmt(prec):>6}  {_fmt(rec):>6}  {_fmt(f1):>6}  "
          f"{_fmt(m_iou):>6}  "
          f"{_fmt(m_onset, '+.2f'):>10}s  "
          f"{_fmt(m_dur, '+.2f'):>7}s")
    print("=" * len(hdr))

    print("\n  Glossary")
    print("  --------")
    print("  GT        number of ground-truth chewing bouts")
    print("  Pred      number of predicted chewing bouts (after min-duration filter)")
    print("  TP        matched pairs (IoU ≥ threshold)")
    print("  FP        predicted bouts with no matching GT bout  (false alarms)")
    print("  FN        GT bouts with no matching prediction       (missed bouts)")
    print("  Prec      bout-level precision = TP / (TP + FP)")
    print("  Rec       bout-level recall    = TP / (TP + FN)")
    print("  F1        harmonic mean of Prec and Rec")
    print("  mIoU      median Intersection-over-Union of matched pairs (1.0 = perfect overlap)")
    print("  Onset lag median (pred_start − gt_start):  + = predicted late,  − = predicted early")
    print("  Dur Δ     median (pred_duration − gt_duration): + = over-segmented, − = under-segmented")


def print_matched_pairs(results: list[dict], max_per_participant: int = 5) -> None:
    print("\n\n  Matched pairs (up to {} per participant)".format(max_per_participant))
    print("  " + "-" * 72)
    for r in results:
        print(f"\n  {r['participant']}  —  {r['tp']} matched / {r['n_gt_bouts']} GT bouts")
        hdr2 = f"    {'GT start':>9}  {'GT end':>7}  {'GT dur':>7}  {'Pred start':>10}  {'Pred end':>8}  {'Pred dur':>8}  {'IoU':>5}"
        print(hdr2)
        print("    " + "-" * (len(hdr2) - 4))
        for m in r["matches"][:max_per_participant]:
            g, p = m["gt"], m["pred"]
            print(f"    {g['start_s']:>9.2f}s  {g['end_s']:>6.2f}s  {g['duration_s']:>6.2f}s  "
                  f"{p['start_s']:>10.2f}s  {p['end_s']:>7.2f}s  {p['duration_s']:>7.2f}s  "
                  f"{m['iou']:>5.3f}")
        if len(r["matches"]) > max_per_participant:
            print(f"    ... ({len(r['matches']) - max_per_participant} more matched pairs)")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Bout-by-bout evaluation of fine-tuned model.")
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("checkpoints/finetune/arrow_of_time_bigru_personal_finetune_best.pt"))
    parser.add_argument("--interim-root", type=Path,
                        default=Path("src/chewing_ssl/data/data/interim"))
    parser.add_argument("--processed-root", type=Path,
                        default=Path("src/chewing_ssl/data/data/processed"))
    parser.add_argument("--dataset", default="SIPMEatingStudy")
    parser.add_argument("--variant", default="personal")
    parser.add_argument("--pred-dir", type=Path, default=Path("checkpoints/predictions"))
    parser.add_argument("--iou-thresh", type=float, default=0.3,
                        help="Minimum IoU to count a predicted bout as a TP (default 0.3)")
    parser.add_argument("--min-bout-dur", type=float, default=0.5,
                        help="Minimum bout duration in seconds for IoU matching (default 0.5)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Raw probability threshold saved to CSV (default 0.5)")
    # Post-processing args
    parser.add_argument("--post-high", type=float, default=0.65,
                        help="Hysteresis upper threshold — starts a bout (default 0.65)")
    parser.add_argument("--post-low", type=float, default=0.35,
                        help="Hysteresis lower threshold — ends a bout (default 0.35)")
    parser.add_argument("--post-max-gap", type=float, default=2.0,
                        help="Maximum gap in seconds to bridge between nearby bouts (default 2.0)")
    parser.add_argument("--post-min-dur", type=float, default=1.0,
                        help="Minimum bout duration in seconds after post-processing (default 1.0)")
    parser.add_argument("--show-pairs", action="store_true",
                        help="Also print individual matched pairs per participant")
    args = parser.parse_args()

    if not args.checkpoint.exists():
        print(f"Checkpoint not found: {args.checkpoint}")
        return

    print("\nBout-by-Bout Evaluation")
    print("=" * 60)
    print(f"  Checkpoint   : {args.checkpoint}")
    print(f"  IoU threshold: {args.iou_thresh}")
    print(f"  Post-process : hysteresis ({args.post_high}/{args.post_low})  "
          f"gap-fill {args.post_max_gap}s  min-dur {args.post_min_dur}s")
    print(f"  Predictions  → {args.pred_dir}/\n")

    # ── Load model ──
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = ChewingDetector()
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    print(f"  Loaded: phase={state.get('phase')}  "
          f"epoch={state.get('epoch', 0)+1}  val_f1={state.get('val_f1', float('nan')):.3f}\n")

    # ── Resolve test participants ──
    idx = pd.read_parquet(
        args.processed_root / "index" / args.variant / "finetune_test.parquet"
    )
    participants = sorted(
        idx[idx["dataset"] == args.dataset]["segment_path"]
        .str.extract(f"/{args.dataset}/([^/]+)/")[0]
        .dropna().unique()
    )
    print(f"  Test participants: {', '.join(participants)}\n")

    results = []
    for participant in participants:
        seg_path = (args.interim_root / args.variant / args.dataset
                    / participant / "segment_00.parquet")
        print(f"  Processing {participant} ...", end=" ", flush=True)
        r = evaluate_participant(
            model, seg_path, participant, args.pred_dir,
            iou_thresh=args.iou_thresh,
            min_bout_dur=args.min_bout_dur,
            threshold=args.threshold,
            post_high=args.post_high,
            post_low=args.post_low,
            post_max_gap_s=args.post_max_gap,
            post_min_dur_s=args.post_min_dur,
        )
        if r is not None:
            r["_iou_thresh"] = args.iou_thresh
            r["_min_dur"]    = args.min_bout_dur
            results.append(r)
            print(f"GT={r['n_gt_bouts']} bouts  Pred={r['n_pred_bouts']} bouts  "
                  f"TP={r['tp']} FP={r['fp']} FN={r['fn']}  F1={_fmt(r['f1'])}")

    if results:
        print_summary_table(results)
        if args.show_pairs:
            print_matched_pairs(results)
        print(f"\n  Per-timestep predictions saved to: {args.pred_dir}/")

    print()


if __name__ == "__main__":
    main()
