"""Bout-by-bout evaluation of the classical-ML pipeline's upscaled predictions
Also produces a per-bout ledger (--out-csv): every ground-truth bout tagged
detected/missed, every predicted bout tagged true-positive/false-alarm.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from chewing_ssl.common.evaluation.bouts import extract_bouts, match_bouts
from chewing_ssl.common.evaluation.postprocess import postprocess

FS = 50.0


def _agg_metrics(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    rec = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else float("nan")
    return prec, rec, f1


def load_recordings(interim_root: Path, upscaled_dir: Path) -> list[dict]:
    """manifest.csv rows -> [{dataset, participant, tag, segment, gt, probs}, ...] per recording.

    gt: bool array, one entry per row of the original segment.
    probs: float array, same length, NaN where no window covers that row.
    """
    manifest = pd.read_csv(upscaled_dir / "manifest.csv")
    recordings = []
    for _, row in manifest.iterrows():
        seg_path = interim_root / row["dataset"] / row["tag"] / f"{row['segment']}.parquet"
        if not seg_path.exists():
            continue
        gt = pd.read_parquet(seg_path, columns=["label"])["label"].to_numpy(dtype=bool)
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
            }
        )
    return recordings


def evaluate(recordings: list[dict], binary_fn, min_bout_dur: float, iou_thresh: float) -> dict:
    tp = fp = fn = 0
    n_gt_bouts = n_pred_bouts = 0
    ious: list[float] = []
    for rec in recordings:
        binary = binary_fn(rec["probs"])
        gt_bouts = extract_bouts(rec["gt"], fs=FS, min_dur=min_bout_dur)
        pred_bouts = extract_bouts(binary, fs=FS, min_dur=min_bout_dur)
        r = match_bouts(gt_bouts, pred_bouts, iou_thresh=iou_thresh)
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


def bout_ledger(
    recordings: list[dict], binary_fn, min_bout_dur: float, iou_thresh: float
) -> pd.DataFrame:
    """One row per bout (GT or predicted), tagged with its match outcome --
    the queryable alternative to eyeballing plot_predictions.py's shading.

    status:
      detected     GT bout matched to a predicted bout (IoU >= iou_thresh)
      missed       GT bout with no matching predicted bout               (false negative)
      false_alarm  predicted bout with no matching GT bout               (false positive)
    Matched rows (detected/its counterpart) also carry the matched bout's
    own [start_s, end_s] and the IoU, so onset/offset drift is inspectable
    per bout, not just in aggregate.
    """
    rows = []
    for rec in recordings:
        binary = binary_fn(rec["probs"])
        gt_bouts = extract_bouts(rec["gt"], fs=FS, min_dur=min_bout_dur)
        pred_bouts = extract_bouts(binary, fs=FS, min_dur=min_bout_dur)
        result = match_bouts(gt_bouts, pred_bouts, iou_thresh=iou_thresh)
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
    parser.add_argument("--iou-thresh", type=float, default=0.3)
    parser.add_argument("--min-bout-dur", type=float, default=0.5)
    parser.add_argument("--post-high", type=float, default=0.65)
    parser.add_argument("--post-low", type=float, default=0.35)
    parser.add_argument("--post-max-gap", type=float, default=1.0)
    parser.add_argument("--post-min-dur", type=float, default=2.0)
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Write the full per-bout ledger (one row per GT/predicted bout, tagged "
        "detected/missed/false_alarm) here, e.g. checkpoints/ml_bout_ledger.csv. "
        "Uses the RAW (no postprocessing) predictions -- see docs/classical_ml.md "
        "for why postprocessing is not applied by default in this pipeline.",
    )
    parser.add_argument(
        "--top-n", type=int, default=15, help="Rows in the worst-offenders console table."
    )
    args = parser.parse_args()

    print("\nBout-level evaluation: raw vs. postprocessed (classical-ML pipeline)")
    print("=" * 78)
    recordings = load_recordings(args.interim_root, args.upscaled_dir)
    print(
        f"  Recordings: {len(recordings)}   IoU threshold: {args.iou_thresh}   min bout: {args.min_bout_dur}s\n"
    )

    def raw_binary_fn(probs):
        return np.nan_to_num(probs, nan=0.0) >= args.threshold

    def post_binary_fn(probs):
        return postprocess(
            probs,
            high=args.post_high,
            low=args.post_low,
            max_gap_s=args.post_max_gap,
            min_dur_s=args.post_min_dur,
        )

    raw_result = evaluate(recordings, raw_binary_fn, args.min_bout_dur, args.iou_thresh)
    post_result = evaluate(recordings, post_binary_fn, args.min_bout_dur, args.iou_thresh)

    print_result("raw (threshold only)", raw_result)
    print_result("postprocessed (hysteresis+gapfill+mindur)", post_result)

    delta_f1 = post_result["f1"] - raw_result["f1"]
    verdict = "HELPS" if delta_f1 > 0.01 else ("HURTS" if delta_f1 < -0.01 else "NO EFFECT")
    print(
        f"\n  Postprocessing (high={args.post_high}/low={args.post_low}, "
        f"gap={args.post_max_gap}s, min_dur={args.post_min_dur}s) vs. raw: "
        f"F1 delta = {delta_f1:+.3f}  -> {verdict}"
    )

    ledger = bout_ledger(recordings, raw_binary_fn, args.min_bout_dur, args.iou_thresh)
    print_per_recording_summary(ledger, top_n=args.top_n)

    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        ledger.to_csv(args.out_csv, index=False)
        print(f"\n  Full per-bout ledger ({len(ledger)} rows) -> {args.out_csv}")

    print("\n" + "=" * 78 + "\n")


if __name__ == "__main__":
    main()
