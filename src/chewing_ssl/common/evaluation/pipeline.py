"""Shared orchestration layer for bout-level evaluation: turns one
recording's (probs, gt[, envelope]) into a final per-recording result, and
aggregates many recordings into a results-summary row or a per-bite ledger.

This is the ONE place the "postprocess -> extract bouts -> match under BOTH
modes -> aggregate" sequence is implemented. Before this module existed, it
was implemented five times: in both `evaluate_bouts.py` scripts, in both
`plot_predictions.py` scripts, and in `compare_bites.py`. All five now call
functions here instead.

A "recording" is a plain dict with (at minimum) keys `gt` (bool array),
`probs` (float array, same length, NaN where uncovered), and `envelope`
(float array, same length -- only required if `PostprocessConfig.
envelope_split` is True). Identity keys (`dataset`, `participant`, `tag`,
`segment`) are carried through into ledger/summary rows but not required by
the evaluation functions themselves.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from chewing_ssl.common.evaluation.bouts import (
    align_bouts_sequential,
    extract_bouts,
    match_bouts,
    match_bouts_tolerant,
)
from chewing_ssl.common.evaluation.postprocess import PostprocessConfig, apply_full_postprocessing

# ── CLI wiring ───────────────────────────────────────────────────────────────


def add_postprocess_args(parser: argparse.ArgumentParser) -> None:
    """The shared postprocessing CLI flags, same names/defaults/help text
    everywhere they're used. Defaults are the tuned values docs/classical_ml.md
    §6a-§6c converged on."""
    parser.add_argument(
        "--post-high", type=float, default=0.75, help="Hysteresis bout-start threshold."
    )
    parser.add_argument(
        "--post-low", type=float, default=0.52, help="Hysteresis bout-end threshold."
    )
    parser.add_argument(
        "--post-max-gap", type=float, default=1.0, help="Max gap (s) bridged between nearby bouts."
    )
    parser.add_argument(
        "--post-min-dur", type=float, default=2.0, help="Discard bouts shorter than this (s)."
    )
    parser.add_argument(
        "--post-max-dur",
        type=float,
        default=35.0,
        help="Split bouts longer than this (s) at their lowest-probability point. Negative disables.",
    )
    parser.add_argument(
        "--post-valley",
        type=float,
        default=0.35,
        help="Split bouts at any interior probability dip below this. Negative disables.",
    )
    parser.add_argument(
        "--envelope-split",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Additionally split bouts using a raw-signal activity envelope. Default on.",
    )
    parser.add_argument("--envelope-window-s", type=float, default=1.5)
    parser.add_argument("--envelope-min-piece-dur", type=float, default=8.0)
    parser.add_argument("--envelope-prominence", type=float, default=2.5)
    parser.add_argument(
        "--smooth-window-s",
        type=float,
        default=0.0,
        help="Moving-average window (s) applied to probs before hysteresis. Default 0.0 (off). "
        "Chiefly useful for a per-timestep trace that isn't already smoothed by overlapping-"
        "window averaging (masked-reconstruction SSL) -- see postprocess.smooth_probs.",
    )


def postprocess_config_from_args(args: argparse.Namespace) -> PostprocessConfig:
    return PostprocessConfig(
        high=args.post_high,
        low=args.post_low,
        max_gap_s=args.post_max_gap,
        min_dur_s=args.post_min_dur,
        max_dur_s=args.post_max_dur if args.post_max_dur >= 0 else None,
        valley_thresh=args.post_valley if args.post_valley >= 0 else None,
        envelope_split=args.envelope_split,
        envelope_window_s=args.envelope_window_s,
        envelope_min_piece_dur=args.envelope_min_piece_dur,
        envelope_prominence=args.envelope_prominence,
        smooth_window_s=args.smooth_window_s,
    )


def add_match_args(parser: argparse.ArgumentParser, default_match_mode: str = "tolerance") -> None:
    """The shared bout-matching CLI flags. `default_match_mode` differs by
    script on purpose (classical ML defaults to tolerance, SSL masked-
    reconstruction defaults to iou -- classical ML's own best-measured
    combination) -- both modes are always computed regardless (see
    `evaluate_recording`), this only controls console/ledger display."""
    parser.add_argument(
        "--match-mode",
        choices=["tolerance", "iou"],
        default=default_match_mode,
        help=f"How a predicted bout is credited against ground truth (default: {default_match_mode}).",
    )
    parser.add_argument(
        "--tolerance-s",
        type=float,
        default=5.0,
        help="--match-mode tolerance: max onset/offset deviation (s).",
    )
    parser.add_argument(
        "--iou-thresh", type=float, default=0.3, help="--match-mode iou: minimum IoU to match."
    )
    parser.add_argument(
        "--min-bout-dur", type=float, default=0.5, help="Minimum bout duration (s)."
    )


# ── Per-recording evaluation ─────────────────────────────────────────────────


def _stats(r: dict) -> dict:
    tp, fp, fn = r["tp"], r["fp"], r["fn"]
    prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    rec = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else float("nan")
    ious = [m["iou"] for m in r["matches"]]
    onset = [(m["pred"]["start_s"] - m["gt"]["start_s"]) for m in r["matches"]]
    dur = [(m["pred"]["duration_s"] - m["gt"]["duration_s"]) for m in r["matches"]]
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "median_iou": float(np.median(ious)) if ious else float("nan"),
        "median_onset_lag_s": float(np.median(onset)) if onset else float("nan"),
        "median_dur_delta_s": float(np.median(dur)) if dur else float("nan"),
    }


def _alignment_stats(events: list[dict]) -> dict:
    """One recording's align_bouts_sequential() output -> counts per label,
    both in "number of clusters" and "number of GT/predicted bites involved"
    terms (a single merge cluster can absorb several GT bites, so cluster
    count alone understates its impact)."""
    event_counts: dict[str, int] = {}
    gt_by_label: dict[str, int] = {}
    pred_by_label: dict[str, int] = {}
    for ev in events:
        label = ev["label"]
        event_counts[label] = event_counts.get(label, 0) + 1
        gt_by_label[label] = gt_by_label.get(label, 0) + ev["n_gt"]
        pred_by_label[label] = pred_by_label.get(label, 0) + ev["n_pred"]
    return {
        "event_counts": event_counts,
        "gt_bites_by_label": gt_by_label,
        "pred_bouts_by_label": pred_by_label,
    }


def evaluate_recording(
    rec: dict,
    cfg: PostprocessConfig,
    min_bout_dur: float,
    match_mode: str,
    iou_thresh: float,
    tolerance_s: float,
) -> dict:
    """Postprocess one recording (cfg.envelope_split requires rec["envelope"])
    and return one result dict:
      - matches print_summary_table's expected shape (participant/n_gt_bouts/
        n_pred_bouts/tp/fp/fn/precision/recall/f1/median_iou/
        median_onset_lag_s/median_dur_delta_s/matches) for whichever
        `match_mode` is selected
      - PLUS `iou_stats`/`tolerance_stats` (both modes, always -- cheap, bout
        lists are short) and `iou_result`/`tolerance_result` (the full match
        dicts, for per-bite ledger building) so a summary-csv row or ledger
        never depends on which mode drove the console display.
      - PLUS `alignment_events`/`alignment_stats` (match_mode-independent --
        see align_bouts_sequential's docstring for why this is a different
        question from TP/FP/FN).
    """
    binary = apply_full_postprocessing(rec["probs"], rec.get("envelope"), cfg)
    gt_bouts = extract_bouts(rec["gt"], fs=cfg.fs, min_dur=min_bout_dur)
    pred_bouts = extract_bouts(binary, fs=cfg.fs, min_dur=min_bout_dur)

    iou_result = match_bouts(gt_bouts, pred_bouts, iou_thresh=iou_thresh)
    tol_result = match_bouts_tolerant(gt_bouts, pred_bouts, tolerance_s=tolerance_s)
    result = tol_result if match_mode == "tolerance" else iou_result
    alignment_events = align_bouts_sequential(gt_bouts, pred_bouts)

    participant = rec.get("participant") or f"{rec.get('dataset', '?')}/{rec.get('tag', '?')}"
    return {
        "participant": participant,
        "n_gt_bouts": len(gt_bouts),
        "n_pred_bouts": len(pred_bouts),
        **_stats(result),
        "matches": result["matches"],
        "iou_stats": _stats(iou_result),
        "tolerance_stats": _stats(tol_result),
        "iou_result": iou_result,
        "tolerance_result": tol_result,
        "alignment_events": alignment_events,
        "alignment_stats": _alignment_stats(alignment_events),
        "gt_bouts": gt_bouts,
        "pred_bouts": pred_bouts,
        "binary": binary,
    }


# ── Aggregation across many recordings ──────────────────────────────────────


def aggregate_dual_mode(results: list[dict]) -> dict:
    """Sum tp/fp/fn across every recording's iou_stats/tolerance_stats and
    recompute overall precision/recall/f1 per mode -- the one aggregation
    used by every results-summary-CSV writer."""

    def _agg(key: str) -> dict:
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

    return {
        "n_gt_bouts": sum(r["n_gt_bouts"] for r in results),
        "n_pred_bouts": sum(r["n_pred_bouts"] for r in results),
        "iou": _agg("iou_stats"),
        "tolerance": _agg("tolerance_stats"),
    }


ALIGNMENT_LABELS = ["match", "merge", "split", "tangle", "missed", "false_alarm"]


def aggregate_alignment(results: list[dict]) -> dict:
    """Sum every recording's alignment_stats (match/merge/split/tangle/
    missed/false_alarm) into one report: events (cluster count) and gt_bites
    (bites actually involved -- what matters for "what fraction of real
    bites suffered X", since one merge cluster can absorb several)."""
    total_gt = sum(r["n_gt_bouts"] for r in results)
    total_pred = sum(r["n_pred_bouts"] for r in results)
    by_label = {label: {"events": 0, "gt_bites": 0, "pred_bouts": 0} for label in ALIGNMENT_LABELS}
    for r in results:
        stats = r["alignment_stats"]
        for label in ALIGNMENT_LABELS:
            by_label[label]["events"] += stats["event_counts"].get(label, 0)
            by_label[label]["gt_bites"] += stats["gt_bites_by_label"].get(label, 0)
            by_label[label]["pred_bouts"] += stats["pred_bouts_by_label"].get(label, 0)
    for label in ALIGNMENT_LABELS:
        by_label[label]["gt_frac"] = (
            by_label[label]["gt_bites"] / total_gt if total_gt > 0 else float("nan")
        )
    return {"n_gt_bouts": total_gt, "n_pred_bouts": total_pred, "by_label": by_label}


# ── Per-bite ledger ──────────────────────────────────────────────────────────


def ledger_rows_for_recording(ident: dict, result: dict, match_mode: str) -> list[dict]:
    """One recording's evaluate_recording() output -> ledger rows (one per
    GT bite and per predicted bite): detected/missed/false_alarm, with
    onset_lag_s/offset_lag_s/duration_delta_s for matched pairs. Also tags
    every row with `cluster_label`/`cluster_n_gt`/`cluster_n_pred` from
    align_bouts_sequential -- match_mode-independent, so e.g. a bite that
    IS a tolerance-mode "detected" can still be flagged cluster_label="tangle"
    if it happened to win the greedy pairing inside a messy cluster."""
    match_result = result["tolerance_result"] if match_mode == "tolerance" else result["iou_result"]
    matched_gt_ids = {id(m["gt"]) for m in match_result["matches"]}
    matched_pred_ids = {id(m["pred"]) for m in match_result["matches"]}

    cluster_by_gt_id: dict[int, dict] = {}
    cluster_by_pred_id: dict[int, dict] = {}
    for ev in result["alignment_events"]:
        for tag, _idx, b in ev["cluster"]:
            (cluster_by_gt_id if tag == "gt" else cluster_by_pred_id)[id(b)] = ev

    def _cluster_cols(bout: dict, by_id: dict[int, dict]) -> dict:
        ev = by_id.get(id(bout))
        if ev is None:
            return {"cluster_label": None, "cluster_n_gt": None, "cluster_n_pred": None}
        return {
            "cluster_label": ev["label"],
            "cluster_n_gt": ev["n_gt"],
            "cluster_n_pred": ev["n_pred"],
        }

    rows = []
    for m in match_result["matches"]:
        g, p = m["gt"], m["pred"]
        rows.append(
            {
                **ident,
                "bout_type": "gt",
                "status": "detected",
                "start_s": g["start_s"],
                "end_s": g["end_s"],
                "duration_s": g["duration_s"],
                "iou": m["iou"],
                "matched_start_s": p["start_s"],
                "matched_end_s": p["end_s"],
                "onset_lag_s": p["start_s"] - g["start_s"],
                "offset_lag_s": p["end_s"] - g["end_s"],
                "duration_delta_s": p["duration_s"] - g["duration_s"],
                **_cluster_cols(g, cluster_by_gt_id),
            }
        )
    for g in result["gt_bouts"]:
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
                    "onset_lag_s": None,
                    "offset_lag_s": None,
                    "duration_delta_s": None,
                    **_cluster_cols(g, cluster_by_gt_id),
                }
            )
    for p in result["pred_bouts"]:
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
                    "onset_lag_s": None,
                    "offset_lag_s": None,
                    "duration_delta_s": None,
                    **_cluster_cols(p, cluster_by_pred_id),
                }
            )
    return rows


def build_bite_ledger(
    recordings: list[dict],
    cfg: PostprocessConfig,
    min_bout_dur: float,
    match_mode: str,
    iou_thresh: float,
    tolerance_s: float,
    extra_ident: dict | None = None,
) -> pd.DataFrame:
    """recordings -> one DataFrame, one row per GT/predicted bite across all
    of them. `extra_ident` (e.g. {"pipeline": "classical_ml", "run_name":
    "rf_4s"}) is merged into every row's identity columns."""
    extra_ident = extra_ident or {}
    rows: list[dict] = []
    for rec in recordings:
        result = evaluate_recording(rec, cfg, min_bout_dur, match_mode, iou_thresh, tolerance_s)
        ident = {
            **extra_ident,
            **{k: rec[k] for k in ("dataset", "participant", "tag", "segment") if k in rec},
        }
        rows += ledger_rows_for_recording(ident, result, match_mode)
    return pd.DataFrame(rows)


# ── Results-summary CSV (one row per model/combination) ────────────────────


def build_summary_row(identity: dict, dual: dict, window_metrics: dict | None = None) -> dict:
    """identity must include `run_name` (the replace-by-run_name key used by
    append_summary_row) plus whatever else the caller wants recorded
    (architecture/pretext_task/window_len/checkpoint for SSL; model_type for
    classical ML; pipeline+run_name for compare_bites). `dual` is
    aggregate_dual_mode's output. `window_metrics` (optional) is a dict with
    f1/precision/recall/auc, merged into window_test_* columns."""
    wm = window_metrics or {}
    return {
        **identity,
        "n_gt_bouts": dual["n_gt_bouts"],
        "n_pred_bouts": dual["n_pred_bouts"],
        "window_test_f1": wm.get("f1"),
        "window_test_precision": wm.get("precision"),
        "window_test_recall": wm.get("recall"),
        "window_test_auc": wm.get("auc"),
        "bout_iou_precision": dual["iou"]["precision"],
        "bout_iou_recall": dual["iou"]["recall"],
        "bout_iou_f1": dual["iou"]["f1"],
        "bout_iou_median_iou": dual["iou"]["median_iou"],
        "bout_tolerance_precision": dual["tolerance"]["precision"],
        "bout_tolerance_recall": dual["tolerance"]["recall"],
        "bout_tolerance_f1": dual["tolerance"]["f1"],
        "bout_tolerance_median_iou": dual["tolerance"]["median_iou"],
    }


def append_summary_row(path: Path, row: dict) -> None:
    """Append, replacing any existing row with the same run_name -- safe to
    re-run for a given run_name without duplicating rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df_row = pd.DataFrame([row])
    if path.exists():
        existing = pd.read_csv(path)
        existing = existing[existing["run_name"] != row["run_name"]]
        df_row = pd.concat([existing, df_row], ignore_index=True)
    df_row.to_csv(path, index=False)
