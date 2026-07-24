"""Bout extraction and bout matching, shared by the classical-ML and SSL
evaluation scripts (chewing_ssl.classical_ml.evaluate_bouts and
chewing_ssl.ssl.masked.evaluate_bouts).

A bout is a contiguous block of timesteps labelled/predicted as "chewing".

Two matching strategies are provided:
  - `match_bouts`: IoU overlap >= a threshold. Penalizes a bout that is
    onset/offset-shifted from ground truth even if both endpoints are only
    off by a fraction of a second, since a short bout's IoU is very
    sensitive to small absolute shifts.
  - `match_bouts_tolerant`: onset and offset within `tolerance_s` seconds of
    ground truth, independent of bout length. Matches "detect the same
    event, roughly at the same time" rather than "detect the same event,
    at pixel-perfect boundaries" -- see docs/classical_ml.md.
"""

from __future__ import annotations

import numpy as np


def extract_bouts(binary: np.ndarray, fs: float = 50.0, min_dur: float = 0.5) -> list[dict]:
    """Convert a binary per-timestep array into a list of bout dicts.

    Each bout: {start_s, end_s, duration_s, start_idx, end_idx}
    Bouts shorter than min_dur seconds are discarded.
    """
    bouts, in_bout, start = [], False, 0
    for i, v in enumerate(binary):
        if v and not in_bout:
            start, in_bout = i, True
        elif not v and in_bout:
            dur = (i - start) / fs
            if dur >= min_dur:
                bouts.append(
                    dict(
                        start_s=start / fs, end_s=i / fs, duration_s=dur, start_idx=start, end_idx=i
                    )
                )
            in_bout = False
    if in_bout:
        dur = (len(binary) - start) / fs
        if dur >= min_dur:
            bouts.append(
                dict(
                    start_s=start / fs,
                    end_s=len(binary) / fs,
                    duration_s=dur,
                    start_idx=start,
                    end_idx=len(binary),
                )
            )
    return bouts


def _iou(a: dict, b: dict) -> float:
    inter = max(0.0, min(a["end_s"], b["end_s"]) - max(a["start_s"], b["start_s"]))
    union = a["duration_s"] + b["duration_s"] - inter
    return inter / union if union > 0 else 0.0


def match_bouts(gt: list[dict], pred: list[dict], iou_thresh: float = 0.3) -> dict:
    """Greedy IoU matching (highest-IoU pairs first).

    Returns dict with keys: tp, fp, fn, matches (list of matched pairs with iou).
    """
    pairs = sorted(
        [
            (gi, pi, _iou(g, p))
            for gi, g in enumerate(gt)
            for pi, p in enumerate(pred)
            if _iou(g, p) >= iou_thresh
        ],
        key=lambda x: -x[2],
    )
    used_gt, used_pred, matches = set(), set(), []
    for gi, pi, score in pairs:
        if gi not in used_gt and pi not in used_pred:
            used_gt.add(gi)
            used_pred.add(pi)
            matches.append({"gt": gt[gi], "pred": pred[pi], "iou": score})

    return {
        "tp": len(matches),
        "fp": len(pred) - len(matches),
        "fn": len(gt) - len(matches),
        "matches": matches,
    }


def _boundary_deviation(gt: dict, pred: dict) -> float:
    """Sum of |onset error| + |offset error|, seconds -- lower is a tighter match."""
    return abs(pred["start_s"] - gt["start_s"]) + abs(pred["end_s"] - gt["end_s"])


def match_bouts_tolerant(gt: list[dict], pred: list[dict], tolerance_s: float = 2.0) -> dict:
    """Greedy onset/offset-tolerance matching (tightest total deviation first).

    A predicted bout matches a ground-truth bout when BOTH its onset and its
    offset fall within `tolerance_s` seconds of the ground-truth bout's onset
    and offset, regardless of bout length or IoU -- e.g. GT [560, 890] with
    tolerance_s=2.0 (100 rows @ 50Hz) accepts any prediction with onset in
    [558, 562] and offset in [888, 892].

    Returns the same shape as match_bouts: dict with tp, fp, fn, matches
    (each match also carries "iou" for diagnostics, even though IoU isn't
    the matching criterion here).
    """
    candidates = sorted(
        (
            (gi, pi, _boundary_deviation(g, p))
            for gi, g in enumerate(gt)
            for pi, p in enumerate(pred)
            if abs(p["start_s"] - g["start_s"]) <= tolerance_s
            and abs(p["end_s"] - g["end_s"]) <= tolerance_s
        ),
        key=lambda x: x[2],
    )
    used_gt, used_pred, matches = set(), set(), []
    for gi, pi, _deviation in candidates:
        if gi not in used_gt and pi not in used_pred:
            used_gt.add(gi)
            used_pred.add(pi)
            matches.append({"gt": gt[gi], "pred": pred[pi], "iou": _iou(gt[gi], pred[pi])})

    return {
        "tp": len(matches),
        "fp": len(pred) - len(matches),
        "fn": len(gt) - len(matches),
        "matches": matches,
    }
