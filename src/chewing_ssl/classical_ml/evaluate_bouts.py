"""Bout-by-bout evaluation of the classical-ML pipeline's upscaled predictions
Also produces a per-bout ledger (--out-csv): every ground-truth bout tagged
detected/missed, every predicted bout tagged true-positive/false-alarm.

Matching (--match-mode, default "tolerance"):
  tolerance  A predicted bout matches a ground-truth bout when its onset AND
             offset are both within --tolerance-s seconds of the ground
             truth's -- "detected the same event at roughly the right time",
             not "detected pixel-perfect boundaries". Default --tolerance-s
             is 5.0, NOT the 2.0s a literal reading of "~100 rows" would
             suggest -- see docs/classical_ml.md §6a: because IoU's
             tolerance for a shift scales with bout length (median GT bout
             here is 10.9s) but this tolerance is a fixed absolute cap, 2.0s
             is actually stricter than IoU>=0.3 for bouts this long, not
             more lenient.
  iou        Strict-overlap matching: IoU(pred, gt) >= --iou-thresh. Combined
             with the duration-bounded postprocessing below, this is the
             best-measured combination so far (F1 0.458 -> 0.763, see §6a/§6c).

Postprocessing enforces a plausible bout-duration range (--post-min-dur /
--post-max-dur, default 2s-35s, per the EDA bout-length finding in
docs/architecture.md) and, by default, splits at any interior probability
dip below --post-valley regardless of duration (--post-valley -1 disables
this trigger, keeping duration-only splitting).

Both are still just splitting the *classifier's probability* trace, which
comes from window-averaged predictions and can plateau smoothly across an
entire multi-bite eating episode -- no interior dip to find at all -- even
where the raw sensor signal clearly shows the pauses between bites (see
docs/classical_ml.md §6b). --envelope-split (on by default) recovers that
structure directly from the raw signal instead, using a rolling-RMS
envelope and prominence-based valley detection.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from chewing_ssl.common.evaluation.bouts import extract_bouts, match_bouts, match_bouts_tolerant
from chewing_ssl.common.evaluation.postprocess import (
    compute_envelope,
    postprocess,
    remove_short_bouts,
    split_bouts_by_envelope,
)

FS = 50.0
ENVELOPE_CHANNELS = ["nav_x_left_temple", "nav_y_left_temple"]


def _agg_metrics(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    rec = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else float("nan")
    return prec, rec, f1


def make_match_fn(match_mode: str, iou_thresh: float, tolerance_s: float):
    """Build a (gt_bouts, pred_bouts) -> dict callable per --match-mode."""
    if match_mode == "tolerance":
        return lambda gt, pred: match_bouts_tolerant(gt, pred, tolerance_s=tolerance_s)
    if match_mode == "iou":
        return lambda gt, pred: match_bouts(gt, pred, iou_thresh=iou_thresh)
    raise ValueError(f"Unknown --match-mode {match_mode!r}")


def load_recordings(
    interim_root: Path, upscaled_dir: Path, envelope_window_s: float = 1.5
) -> list[dict]:
    """manifest.csv rows -> [{dataset, participant, tag, segment, gt, probs, envelope}, ...]
    per recording.

    gt: bool array, one entry per row of the original segment.
    probs: float array, same length, NaN where no window covers that row.
    envelope: float array, same length -- rolling-RMS magnitude of
    ENVELOPE_CHANNELS (see compute_envelope), used by split_bouts_by_envelope.
    """
    manifest = pd.read_csv(upscaled_dir / "manifest.csv")
    recordings = []
    for _, row in manifest.iterrows():
        seg_path = interim_root / row["dataset"] / row["tag"] / f"{row['segment']}.parquet"
        if not seg_path.exists():
            continue
        seg = pd.read_parquet(seg_path, columns=["label", *ENVELOPE_CHANNELS])
        gt = seg["label"].to_numpy(dtype=bool)
        signal_mag = np.sqrt((seg[ENVELOPE_CHANNELS].to_numpy(dtype=np.float64) ** 2).sum(axis=1))
        envelope = compute_envelope(signal_mag, fs=FS, window_s=envelope_window_s)

        pred_df = pd.read_parquet(row["out_path"])
        probs = np.full(len(gt), np.nan, dtype=np.float32)
        probs[pred_df["row_idx"].to_numpy()] = pred_df["pred_prob"].to_numpy()
        recordings.append(
            {
                "dataset": row["dataset"],
                "participant": row["participant"],
                "tag": row["tag"],
                "segment": row["segment"],
                "gt": gt,
                "probs": probs,
                "envelope": envelope,
            }
        )
    return recordings


def evaluate(recordings: list[dict], binary_fn, min_bout_dur: float, match_fn) -> dict:
    tp = fp = fn = 0
    n_gt_bouts = n_pred_bouts = 0
    ious: list[float] = []
    for rec in recordings:
        binary = binary_fn(rec)
        gt_bouts = extract_bouts(rec["gt"], fs=FS, min_dur=min_bout_dur)
        pred_bouts = extract_bouts(binary, fs=FS, min_dur=min_bout_dur)
        r = match_fn(gt_bouts, pred_bouts)
        tp += r["tp"]
        fp += r["fp"]
        fn += r["fn"]
        ious += [m["iou"] for m in r["matches"]]
        n_gt_bouts += len(gt_bouts)
        n_pred_bouts += len(pred_bouts)

    prec, rec_, f1 = _agg_metrics(tp, fp, fn)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": prec,
        "recall": rec_,
        "f1": f1,
        "median_iou": float(np.median(ious)) if ious else float("nan"),
        "n_gt_bouts": n_gt_bouts,
        "n_pred_bouts": n_pred_bouts,
    }


def print_result(name: str, r: dict) -> None:
    print(
        f"  {name:<32s} prec={r['precision']:.3f}  rec={r['recall']:.3f}  f1={r['f1']:.3f}  "
        f"median_iou={r['median_iou']:.3f}  "
        f"GT={r['n_gt_bouts']} pred={r['n_pred_bouts']} (tp={r['tp']} fp={r['fp']} fn={r['fn']})"
    )


def bout_ledger(recordings: list[dict], binary_fn, min_bout_dur: float, match_fn) -> pd.DataFrame:
    """One row per bout (GT or predicted), tagged with its match outcome --
    the queryable alternative to eyeballing plot_predictions.py's shading.

    status:
      detected     GT bout matched to a predicted bout (per match_fn)
      missed       GT bout with no matching predicted bout               (false negative)
      false_alarm  predicted bout with no matching GT bout               (false positive)
    Matched rows (detected/its counterpart) also carry the matched bout's
    own [start_s, end_s] and the IoU, so onset/offset drift is inspectable
    per bout, not just in aggregate.
    """
    rows = []
    for rec in recordings:
        binary = binary_fn(rec)
        gt_bouts = extract_bouts(rec["gt"], fs=FS, min_dur=min_bout_dur)
        pred_bouts = extract_bouts(binary, fs=FS, min_dur=min_bout_dur)
        result = match_fn(gt_bouts, pred_bouts)
        matched_gt_ids = {id(m["gt"]) for m in result["matches"]}
        matched_pred_ids = {id(m["pred"]) for m in result["matches"]}
        ident = {k: rec[k] for k in ("dataset", "participant", "tag", "segment")}

        for m in result["matches"]:
            rows.append(
                {
                    **ident,
                    "bout_type": "gt",
                    "status": "detected",
                    "start_s": m["gt"]["start_s"],
                    "end_s": m["gt"]["end_s"],
                    "duration_s": m["gt"]["duration_s"],
                    "iou": m["iou"],
                    "matched_start_s": m["pred"]["start_s"],
                    "matched_end_s": m["pred"]["end_s"],
                }
            )
        for g in gt_bouts:
            if id(g) not in matched_gt_ids:
                rows.append(
                    {
                        **ident,
                        "bout_type": "gt",
                        "status": "missed",
                        "start_s": g["start_s"],
                        "end_s": g["end_s"],
                        "duration_s": g["duration_s"],
                        "iou": 0.0,
                        "matched_start_s": None,
                        "matched_end_s": None,
                    }
                )
        for p in pred_bouts:
            if id(p) not in matched_pred_ids:
                rows.append(
                    {
                        **ident,
                        "bout_type": "pred",
                        "status": "false_alarm",
                        "start_s": p["start_s"],
                        "end_s": p["end_s"],
                        "duration_s": p["duration_s"],
                        "iou": 0.0,
                        "matched_start_s": None,
                        "matched_end_s": None,
                    }
                )
    return pd.DataFrame(rows)


def print_per_recording_summary(ledger: pd.DataFrame, top_n: int = 15) -> None:
    """Worst-offenders-first table -- which recordings actually need a manual
    look, instead of scanning all of them."""
    gt_rows = ledger[ledger["bout_type"] == "gt"]
    pred_rows = ledger[ledger["bout_type"] == "pred"]

    per_rec = (
        gt_rows.groupby(["dataset", "participant", "tag", "segment"])
        .apply(
            lambda g: pd.Series(
                {
                    "n_gt": len(g),
                    "n_detected": (g["status"] == "detected").sum(),
                    "n_missed": (g["status"] == "missed").sum(),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    fa_counts = (
        pred_rows.groupby(["dataset", "participant", "tag", "segment"])
        .size()
        .rename("n_false_alarm")
    )
    per_rec = per_rec.merge(fa_counts, on=["dataset", "participant", "tag", "segment"], how="left")
    per_rec["n_false_alarm"] = per_rec["n_false_alarm"].fillna(0).astype(int)
    per_rec["n_problems"] = per_rec["n_missed"] + per_rec["n_false_alarm"]
    per_rec = per_rec.sort_values("n_problems", ascending=False)

    print(
        f"\n  Per-recording summary, worst {min(top_n, len(per_rec))} of {len(per_rec)} "
        f"(most missed+false-alarm bouts first):"
    )
    hdr = f"    {'Recording':<55}  {'GT':>3}  {'Det':>3}  {'Miss':>4}  {'FalseAlarm':>10}"
    print(hdr)
    print("    " + "-" * (len(hdr) - 4))
    for _, r in per_rec.head(top_n).iterrows():
        rec_id = f"{r['dataset']}/{r['participant']}/{r['tag']}/{r['segment']}"
        print(
            f"    {rec_id:<55}  {r['n_gt']:>3}  {r['n_detected']:>3}  {r['n_missed']:>4}  {r['n_false_alarm']:>10}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interim-root", type=Path, default=Path("data/interim_normalized"))
    parser.add_argument(
        "--upscaled-dir", type=Path, default=Path("data/processed/ml_predictions_upscaled")
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Raw (no postprocessing) binarization threshold.",
    )
    parser.add_argument(
        "--match-mode",
        choices=["tolerance", "iou"],
        default="tolerance",
        help="How a predicted bout is credited against a ground-truth bout (default: tolerance).",
    )
    parser.add_argument(
        "--tolerance-s",
        type=float,
        default=5.0,
        help="--match-mode tolerance: max onset/offset deviation in seconds. Default 5.0 -- "
        "roughly half the median GT bout duration (10.9s); see docs/classical_ml.md §6a for "
        "why 2.0s (a literal '~100 rows' reading) is actually stricter than IoU>=0.3 here.",
    )
    parser.add_argument(
        "--iou-thresh", type=float, default=0.3, help="--match-mode iou: minimum IoU to match."
    )
    parser.add_argument("--min-bout-dur", type=float, default=0.5)
    parser.add_argument(
        "--post-high",
        type=float,
        default=0.75,
        help="Hysteresis bout-start threshold (default 0.75, not 0.65 -- re-tuned in "
        "docs/classical_ml.md §6c, a clean win on both 4s- and 2s-window predictions).",
    )
    parser.add_argument(
        "--post-low",
        type=float,
        default=0.52,
        help="Hysteresis bout-end threshold (default 0.52, not 0.35 -- see --post-high).",
    )
    parser.add_argument(
        "--post-max-gap",
        type=float,
        default=1.0,
        help="Note: measured to have zero effect at this dataset/model's probability "
        "resolution (docs/classical_ml.md §6b) -- kept configurable, not because tuning it "
        "here helped.",
    )
    parser.add_argument(
        "--post-min-dur",
        type=float,
        default=2.0,
        help="Discard postprocessed bouts shorter than this (real bite bouts run ~3-17s).",
    )
    parser.add_argument(
        "--post-max-dur",
        type=float,
        default=35.0,
        help="Split postprocessed bouts longer than this at their lowest-probability point.",
    )
    parser.add_argument(
        "--post-valley",
        type=float,
        default=0.35,
        help="Split postprocessed bouts at any interior probability dip below this, "
        "regardless of duration (generalizes --post-max-dur to catch shorter merged "
        "blobs too). Pass a negative value to disable.",
    )
    parser.add_argument(
        "--envelope-split",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Additionally split bouts using a raw-signal activity envelope "
        "(nav_x/nav_y magnitude), which can reveal bite-to-bite pauses the "
        "classifier's own (window-averaged, smoother) probability misses "
        "entirely -- see docs/classical_ml.md §6b. Default on.",
    )
    parser.add_argument("--envelope-window-s", type=float, default=1.5)
    parser.add_argument(
        "--envelope-min-piece-dur",
        type=float,
        default=8.0,
        help="Minimum spacing between envelope valleys (default 8.0s, not 6.0 -- re-tuned "
        "in docs/classical_ml.md §6c; smaller values over-split into implausibly many "
        "fragments since real chews ripple the envelope every ~0.5-0.7s).",
    )
    parser.add_argument(
        "--envelope-prominence",
        type=float,
        default=2.5,
        help="Required envelope-valley prominence, in local std-devs (default 2.5, not 3.0 "
        "-- re-tuned in docs/classical_ml.md §6c; lower values fire on individual "
        "chew-cycle ripple, not just bite-to-bite pauses).",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default="bout_ledger.csv",
        help="Write the full per-bout ledger (one row per GT/predicted bout, tagged "
        "detected/missed/false_alarm) here, e.g. checkpoints/ml_bout_ledger.csv. "
        "Uses the RAW (no postprocessing) predictions.",
    )
    parser.add_argument(
        "--top-n", type=int, default=15, help="Rows in the worst-offenders console table."
    )
    args = parser.parse_args()

    match_fn = make_match_fn(args.match_mode, args.iou_thresh, args.tolerance_s)
    match_desc = (
        f"tolerance={args.tolerance_s}s"
        if args.match_mode == "tolerance"
        else f"IoU>={args.iou_thresh}"
    )
    valley_thresh = args.post_valley if args.post_valley >= 0 else None

    print("\nBout-level evaluation: raw vs. postprocessed (classical-ML pipeline)")
    print("=" * 78)
    recordings = load_recordings(
        args.interim_root, args.upscaled_dir, envelope_window_s=args.envelope_window_s
    )
    print(
        f"  Recordings: {len(recordings)}   Match: {args.match_mode} ({match_desc})   "
        f"min bout: {args.min_bout_dur}s   envelope-split: {args.envelope_split}\n"
    )

    def raw_binary_fn(rec):
        return np.nan_to_num(rec["probs"], nan=0.0) >= args.threshold

    def post_binary_fn(rec):
        binary = postprocess(
            rec["probs"],
            high=args.post_high,
            low=args.post_low,
            max_gap_s=args.post_max_gap,
            min_dur_s=args.post_min_dur,
            max_dur_s=args.post_max_dur,
            valley_thresh=valley_thresh,
        )
        if args.envelope_split:
            binary = split_bouts_by_envelope(
                binary,
                rec["envelope"],
                fs=FS,
                min_piece_dur_s=args.envelope_min_piece_dur,
                prominence_frac=args.envelope_prominence,
            )
            binary = remove_short_bouts(binary, fs=FS, min_dur_s=args.post_min_dur)
        return binary

    raw_result = evaluate(recordings, raw_binary_fn, args.min_bout_dur, match_fn)
    post_result = evaluate(recordings, post_binary_fn, args.min_bout_dur, match_fn)

    print_result("raw (threshold only)", raw_result)
    print_result("postprocessed (hysteresis+gapfill+splits)", post_result)

    delta_f1 = post_result["f1"] - raw_result["f1"]
    verdict = "HELPS" if delta_f1 > 0.01 else ("HURTS" if delta_f1 < -0.01 else "NO EFFECT")
    print(
        f"\n  Postprocessing (high={args.post_high}/low={args.post_low}, "
        f"gap={args.post_max_gap}s, dur=[{args.post_min_dur}, {args.post_max_dur}]s, "
        f"valley={valley_thresh}, envelope-split={args.envelope_split}) vs. raw: "
        f"F1 delta = {delta_f1:+.3f}  -> {verdict}"
    )

    ledger = bout_ledger(recordings, post_binary_fn, args.min_bout_dur, match_fn)
    print_per_recording_summary(ledger, top_n=args.top_n)

    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        ledger.to_csv(args.out_csv, index=False)
        print(f"\n  Full per-bout ledger ({len(ledger)} rows) -> {args.out_csv}")

    print("\n" + "=" * 78 + "\n")


if __name__ == "__main__":
    main()
