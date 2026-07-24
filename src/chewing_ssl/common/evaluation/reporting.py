"""Shared bout-level result printing, used by every evaluate_bouts.py
(chewing_ssl.ssl.masked.evaluate_bouts, chewing_ssl.ssl.tfc.evaluate_bouts,
chewing_ssl.classical_ml.evaluate_bouts) so the three pipelines' terminal
output -- and therefore what gets compared across them -- has one shared
format and definition, not three subtly-different re-implementations.

Consumes the per-recording result dict shape produced by
chewing_ssl.common.evaluation.bouts.match_bouts /
chewing_ssl.common.evaluation.bouts.match_bouts_tolerant: at minimum
`participant, n_gt_bouts, n_pred_bouts, tp, fp, fn, precision, recall, f1,
median_iou, median_onset_lag_s, median_dur_delta_s, matches`.
"""

from __future__ import annotations

import numpy as np


def fmt(v, spec: str = ".3f") -> str:
    return "nan" if (isinstance(v, float) and np.isnan(v)) else format(v, spec)


def print_summary_table(results: list[dict], match_mode: str = "iou") -> None:
    hdr = (
        f"{'Recording':<20}  {'GT':>4}  {'Pred':>4}  {'TP':>3}  {'FP':>3}  {'FN':>3}  "
        f"{'Prec':>6}  {'Rec':>6}  {'F1':>6}  {'mIoU':>6}  "
        f"{'Onset lag':>10}  {'Dur Δ':>8}"
    )
    print("\n" + "=" * len(hdr))
    if match_mode == "tolerance":
        crit = "tolerance = {:.1f}s".format(results[0].get("_tolerance_s", 2.0))
    else:
        crit = "IoU threshold = {:.2f}".format(results[0].get("_iou_thresh", 0.3))
    print(f"  Bout-by-bout evaluation  ({crit},  min bout = {results[0].get('_min_dur', 0.5):.1f}s)")
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))

    agg = {"tp": 0, "fp": 0, "fn": 0, "ious": [], "onset": [], "dur": []}
    for r in results:
        agg["tp"] += r["tp"]
        agg["fp"] += r["fp"]
        agg["fn"] += r["fn"]
        agg["ious"] += [m["iou"] for m in r["matches"]]
        agg["onset"] += [(m["pred"]["start_s"] - m["gt"]["start_s"]) for m in r["matches"]]
        agg["dur"] += [(m["pred"]["duration_s"] - m["gt"]["duration_s"]) for m in r["matches"]]

        print(
            f"  {r['participant']:<20}  {r['n_gt_bouts']:>4}  {r['n_pred_bouts']:>4}  "
            f"{r['tp']:>3}  {r['fp']:>3}  {r['fn']:>3}  "
            f"{fmt(r['precision']):>6}  {fmt(r['recall']):>6}  {fmt(r['f1']):>6}  "
            f"{fmt(r['median_iou']):>6}  "
            f"{fmt(r['median_onset_lag_s'], '+.2f'):>10}s  "
            f"{fmt(r['median_dur_delta_s'], '+.2f'):>7}s"
        )

    print("-" * len(hdr))
    tp, fp, fn = agg["tp"], agg["fp"], agg["fn"]
    prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    rec = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else float("nan")
    m_iou = float(np.median(agg["ious"])) if agg["ious"] else float("nan")
    m_onset = float(np.median(agg["onset"])) if agg["onset"] else float("nan")
    m_dur = float(np.median(agg["dur"])) if agg["dur"] else float("nan")

    total_gt = sum(r["n_gt_bouts"] for r in results)
    total_pred = sum(r["n_pred_bouts"] for r in results)

    print(
        f"  {'TOTAL':<20}  {total_gt:>4}  {total_pred:>4}  "
        f"{tp:>3}  {fp:>3}  {fn:>3}  "
        f"{fmt(prec):>6}  {fmt(rec):>6}  {fmt(f1):>6}  "
        f"{fmt(m_iou):>6}  "
        f"{fmt(m_onset, '+.2f'):>10}s  "
        f"{fmt(m_dur, '+.2f'):>7}s"
    )
    print("=" * len(hdr))

    print("\n  Glossary")
    print("  --------")
    print("  GT        number of ground-truth chewing bouts")
    print("  Pred      number of predicted chewing bouts (after post-processing)")
    print("  TP        matched pairs")
    print("  FP        predicted bouts with no matching GT bout  (false alarms)")
    print("  FN        GT bouts with no matching prediction       (missed bouts)")
    print("  Prec      bout-level precision = TP / (TP + FP)")
    print("  Rec       bout-level recall    = TP / (TP + FN)")
    print("  F1        harmonic mean of Prec and Rec")
    print("  mIoU      median Intersection-over-Union of matched pairs (1.0 = perfect overlap)")
    print("  Onset lag median (pred_start − gt_start):  + = predicted late,  − = predicted early")
    print(
        "  Dur Δ     median (pred_duration − gt_duration): + = over-segmented, − = under-segmented"
    )


def print_matched_pairs(results: list[dict], max_per_participant: int = 5) -> None:
    print("\n\n  Matched pairs (up to {} per recording)".format(max_per_participant))
    print("  " + "-" * 72)
    for r in results:
        print(f"\n  {r['participant']}  —  {r['tp']} matched / {r['n_gt_bouts']} GT bouts")
        hdr2 = f"    {'GT start':>9}  {'GT end':>7}  {'GT dur':>7}  {'Pred start':>10}  {'Pred end':>8}  {'Pred dur':>8}  {'IoU':>5}"
        print(hdr2)
        print("    " + "-" * (len(hdr2) - 4))
        for m in r["matches"][:max_per_participant]:
            g, p = m["gt"], m["pred"]
            print(
                f"    {g['start_s']:>9.2f}s  {g['end_s']:>6.2f}s  {g['duration_s']:>6.2f}s  "
                f"{p['start_s']:>10.2f}s  {p['end_s']:>7.2f}s  {p['duration_s']:>7.2f}s  "
                f"{m['iou']:>5.3f}"
            )
        if len(r["matches"]) > max_per_participant:
            print(f"    ... ({len(r['matches']) - max_per_participant} more matched pairs)")
