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

import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd


class _Tee(io.TextIOBase):
    """Writes to multiple streams at once. Used to capture printed console
    output into a string buffer (for saving to disk) while it still appears
    live on the terminal as usual -- without changing any of the print_*
    functions below, which stay plain `print()` calls."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, s: str) -> int:
        for stream in self._streams:
            stream.write(s)
        return len(s)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


@contextlib.contextmanager
def capture_console_report():
    """Context manager: everything printed inside also accumulates in the
    yielded StringIO, in addition to appearing on the real stdout as usual.

    Wrap an evaluate_bouts.py run's reporting section in this, then pass the
    buffer to `save_console_report` -- saves the EXACT console output
    (per-recording table, glossary, alignment breakdown, worst offenders)
    to a file, with no duplicated formatting logic to maintain.
    """
    buffer = io.StringIO()
    real_stdout = sys.stdout
    tee = _Tee(real_stdout, buffer)
    with contextlib.redirect_stdout(tee):
        yield buffer


def save_console_report(buffer: io.StringIO, path: Path, title: str) -> None:
    """Write a captured console report (see capture_console_report) to a
    markdown file: a title, then the exact console text in a fenced code
    block so its fixed-width tables render correctly."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# {title}\n\n```\n{buffer.getvalue()}\n```\n")


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
    print(
        f"  Bout-by-bout evaluation  ({crit},  min bout = {results[0].get('_min_dur', 0.5):.1f}s)"
    )
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


def print_worst_offenders(ledger, top_n: int = 15) -> None:
    """Worst-offenders-first table from a per-bite ledger (chewing_ssl.
    common.evaluation.pipeline.build_bite_ledger's output, or any DataFrame
    with the same dataset/participant/tag/segment/bout_type/status columns)
    -- which recordings actually need a manual look, instead of scanning all
    of them. Pipeline-agnostic: works the same for classical ML's or SSL's
    ledger (or a combined one with both).
    """
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


def print_alignment_summary(agg: dict) -> None:
    """Print aggregate_alignment()'s output: for each cluster label (match/
    merge/split/tangle/missed/false_alarm), how many clusters occurred and
    what fraction of all GT bites they involved. Complements print_summary_
    table's TP/FP/FN -- this answers "what did detection failures look
    like", not just "how many.\""""
    total_gt = agg["n_gt_bouts"]
    hdr = f"  {'Label':<12}  {'Events':>7}  {'GT bites':>9}  {'% of GT':>8}  {'Pred bouts':>10}"
    print("\n  Bite alignment breakdown (sequential clustering, match-mode independent)")
    print("  " + "-" * (len(hdr) - 2))
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for label in ["match", "merge", "split", "tangle", "missed", "false_alarm"]:
        row = agg["by_label"][label]
        pct = f"{row['gt_frac']:.1%}" if total_gt > 0 else "n/a"
        print(
            f"  {label:<12}  {row['events']:>7}  {row['gt_bites']:>9}  {pct:>8}  {row['pred_bouts']:>10}"
        )
    print("  " + "-" * (len(hdr) - 2))
    print(f"  Total GT bites: {total_gt}   Total predicted bouts: {agg['n_pred_bouts']}")
    print("\n  Glossary")
    print("  --------")
    print("  match        clean 1:1 overlap between one GT bite and one predicted bout")
    print("  merge        one predicted bout spans >=2 real bites (they get bridged into one)")
    print("  split        one real bite is covered by >=2 separate predicted bouts")
    print("  tangle       a chain of >=2 GT bites and >=2 predicted bouts all overlapping --")
    print("               usually a dense run of closely-spaced bites where boundary noise")
    print("               has decoupled bite count from predicted-bout count on both sides")
    print("  missed       GT bite with no overlapping prediction at all")
    print("  false_alarm  predicted bout with no overlapping GT bite at all")


def print_matched_pairs(results: list[dict], max_per_participant: int = 5) -> None:
    print(f"\n\n  Matched pairs (up to {max_per_participant} per recording)")
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
