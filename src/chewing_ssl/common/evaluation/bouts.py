"""Bout extraction and bout matching, shared by the classical-ML and SSL
evaluation scripts (chewing_ssl.classical_ml.evaluate_bouts and
chewing_ssl.ssl.masked.evaluate_bouts).

A bout is a contiguous block of timesteps labelled/predicted as "chewing".

Three matching/alignment strategies are provided:
  - `match_bouts`: IoU overlap >= a threshold. Penalizes a bout that is
    onset/offset-shifted from ground truth even if both endpoints are only
    off by a fraction of a second, since a short bout's IoU is very
    sensitive to small absolute shifts. Greedy tightest-IoU-first, one-to-one.
  - `match_bouts_tolerant`: onset and offset within `tolerance_s` seconds of
    ground truth, independent of bout length. Matches "detect the same
    event, roughly at the same time" rather than "detect the same event,
    at pixel-perfect boundaries" -- see docs/classical_ml.md. Greedy
    tightest-total-deviation-first, one-to-one.
  - `align_bouts_sequential`: NOT a one-to-one matcher -- groups GT and
    predicted bouts into maximal temporally-overlapping clusters (a single
    left-to-right sweep, no scoring/sorting) and labels each cluster
    match/merge/split/tangle/missed/false_alarm. Both `match_bouts*`
    functions answer "how many bouts were correctly detected"; this answers
    "when detection fails, what did it fail *like*" -- a bite lost inside a
    merge is a structurally different failure from one simply never
    detected, even though both count as an FN under match_bouts*.
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
                    {
                        "start_s": start / fs,
                        "end_s": i / fs,
                        "duration_s": dur,
                        "start_idx": start,
                        "end_idx": i,
                    }
                )
            in_bout = False
    if in_bout:
        dur = (len(binary) - start) / fs
        if dur >= min_dur:
            bouts.append(
                {
                    "start_s": start / fs,
                    "end_s": len(binary) / fs,
                    "duration_s": dur,
                    "start_idx": start,
                    "end_idx": len(binary),
                }
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


def align_bouts_sequential(gt: list[dict], pred: list[dict]) -> list[dict]:
    """Group GT and predicted bouts into maximal temporally-overlapping
    clusters via a single left-to-right sweep (both lists are already
    time-sorted, as `extract_bouts` produces them) -- classic "merge
    overlapping intervals," tagging each interval by its source list.

    Every GT bout and every predicted bout ends up in exactly one cluster.
    Each cluster is labelled by how many GT (n_gt) and predicted (n_pred)
    bouts it contains:
      - n_gt=0            -> "false_alarm"  (isolated predicted bout)
      - n_pred=0           -> "missed"       (isolated GT bout)
      - n_gt=1, n_pred=1  -> "match"         (clean one-to-one overlap)
      - n_gt>=2, n_pred=1 -> "merge"         (one predicted bout spans several real bites)
      - n_gt=1, n_pred>=2 -> "split"         (one real bite covered by several predicted bouts)
      - n_gt>=2, n_pred>=2 -> "tangle"       (a chain of several GT and several predicted
                                               bouts all overlapping each other -- typically
                                               a dense run of closely-spaced real bites where
                                               boundary noise has decoupled bite count from
                                               predicted-bout count on both sides at once)

    Returns a list of dicts: {"label", "n_gt", "n_pred", "cluster"} where
    `cluster` is [(("gt"|"pred"), original_index, bout_dict), ...] in time
    order -- kept around so callers (e.g. the per-bite ledger) can look up
    which cluster a specific GT/predicted bout id landed in.
    """
    tagged = sorted(
        [("gt", i, b) for i, b in enumerate(gt)] + [("pred", i, b) for i, b in enumerate(pred)],
        key=lambda t: t[2]["start_s"],
    )
    raw_clusters: list[list[tuple[str, int, dict]]] = []
    cluster: list[tuple[str, int, dict]] = []
    cluster_end = -1.0
    for tag, idx, b in tagged:
        if cluster and b["start_s"] > cluster_end:
            raw_clusters.append(cluster)
            cluster = []
            cluster_end = -1.0
        cluster.append((tag, idx, b))
        cluster_end = max(cluster_end, b["end_s"])
    if cluster:
        raw_clusters.append(cluster)

    events = []
    for cluster in raw_clusters:
        n_gt = sum(1 for t, _, _ in cluster if t == "gt")
        n_pred = sum(1 for t, _, _ in cluster if t == "pred")
        if n_gt == 0:
            label = "false_alarm"
        elif n_pred == 0:
            label = "missed"
        elif n_gt == 1 and n_pred == 1:
            label = "match"
        elif n_gt >= 2 and n_pred == 1:
            label = "merge"
        elif n_gt == 1 and n_pred >= 2:
            label = "split"
        else:
            label = "tangle"
        events.append({"label": label, "n_gt": n_gt, "n_pred": n_pred, "cluster": cluster})
    return events


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
