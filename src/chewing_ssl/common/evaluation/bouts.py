"""Bout extraction and IoU-based bout matching, shared by the classical-ML and
SSL evaluation scripts (chewing_ssl.classical_ml.evaluate_bouts and
chewing_ssl.ssl.masked.evaluate_bouts).

A bout is a contiguous block of timesteps labelled/predicted as "chewing".
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
