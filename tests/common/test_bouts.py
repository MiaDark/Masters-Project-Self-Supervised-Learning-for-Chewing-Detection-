from __future__ import annotations

import numpy as np

from chewing_ssl.common.evaluation.bouts import extract_bouts, match_bouts, match_bouts_tolerant

FS = 50.0


def _bout(start_row: int, end_row: int, fs: float = FS) -> dict:
    return dict(
        start_s=start_row / fs,
        end_s=end_row / fs,
        duration_s=(end_row - start_row) / fs,
        start_idx=start_row,
        end_idx=end_row,
    )


def test_extract_bouts_finds_two_separate_bouts():
    binary = np.zeros(1300, dtype=bool)
    binary[560:890] = True
    binary[970:1200] = True
    bouts = extract_bouts(binary, fs=FS, min_dur=0.5)
    assert len(bouts) == 2
    assert bouts[0]["start_idx"] == 560 and bouts[0]["end_idx"] == 890
    assert bouts[1]["start_idx"] == 970 and bouts[1]["end_idx"] == 1200


def test_tolerant_match_accepts_small_boundary_shift_within_tolerance():
    # GT 560-890 (rows), prediction shifted by 10 rows (0.2s) on each side --
    # well within a 2s/100-row tolerance, and should count as a match even
    # though it's a poor IoU case for a short bout.
    gt = [_bout(560, 890)]
    pred = [_bout(550, 900)]
    result = match_bouts_tolerant(gt, pred, tolerance_s=2.0)
    assert result["tp"] == 1
    assert result["fp"] == 0
    assert result["fn"] == 0


def test_tolerant_match_rejects_shift_beyond_tolerance():
    gt = [_bout(560, 890)]
    # 150-row (3s) onset shift, outside a 2s tolerance
    pred = [_bout(410, 890)]
    result = match_bouts_tolerant(gt, pred, tolerance_s=2.0)
    assert result["tp"] == 0
    assert result["fp"] == 1
    assert result["fn"] == 1


def test_tolerant_match_two_gt_bouts_both_detected():
    gt = [_bout(560, 890), _bout(970, 1200)]
    pred = [_bout(555, 895), _bout(965, 1205)]
    result = match_bouts_tolerant(gt, pred, tolerance_s=2.0)
    assert result["tp"] == 2
    assert result["fp"] == 0
    assert result["fn"] == 0


def test_tolerant_match_greedy_prefers_tightest_deviation():
    # One predicted bout falls within tolerance of BOTH gt bouts (a genuine
    # assignment conflict) but is a tighter fit for gt[0] (deviation 0.8s)
    # than gt[1] (1.6s). Greedy-by-smallest-deviation must claim gt[0] first,
    # leaving gt[1] unmatched since no other prediction is available for it.
    gt = [_bout(0, 50), _bout(60, 110)]  # 0.0-1.0s, 1.2-2.2s
    pred = [_bout(20, 70)]  # 0.4-1.4s
    result = match_bouts_tolerant(gt, pred, tolerance_s=2.0)
    assert result["tp"] == 1
    assert result["fn"] == 1
    assert result["fp"] == 0
    assert id(result["matches"][0]["gt"]) == id(gt[0])


def test_tolerant_match_no_double_counting_one_pred_one_gt():
    gt = [_bout(560, 890)]
    pred = [_bout(560, 890), _bout(561, 891)]  # two near-duplicate predictions
    result = match_bouts_tolerant(gt, pred, tolerance_s=2.0)
    assert result["tp"] == 1
    assert result["fp"] == 1  # the extra duplicate prediction is a false alarm
    assert result["fn"] == 0


def test_tolerant_and_iou_match_can_disagree_on_short_shifted_bout():
    # A short bout (1.0s) shifted by 0.6s (well within a 2s tolerance) has
    # IoU = 0.4/1.6 = 0.25, below a 0.5 IoU threshold -- but it's a clean
    # tolerant match. This is the whole point of the new matching mode: a
    # short bout's IoU is very sensitive to small absolute shifts.
    gt = [_bout(0, 50)]  # 1.0s
    pred = [_bout(30, 80)]  # shifted by 0.6s (30 rows), same length
    iou_result = match_bouts(gt, pred, iou_thresh=0.5)
    tolerant_result = match_bouts_tolerant(gt, pred, tolerance_s=2.0)
    assert iou_result["tp"] == 0
    assert tolerant_result["tp"] == 1
