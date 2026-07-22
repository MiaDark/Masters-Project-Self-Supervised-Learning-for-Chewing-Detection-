"""Post-processing pipeline for per-timestep chewing predictions.

Applied in order:
  1. Hysteresis thresholding  — start a bout at `high`, end it at `low`
  2. Gap filling              — bridge gaps shorter than `max_gap_s` between bouts
  3. Minimum duration         — discard bouts shorter than `min_dur_s`
"""

from __future__ import annotations

import numpy as np

FS = 50.0  # Hz


def apply_hysteresis(probs: np.ndarray, high: float = 0.65, low: float = 0.35) -> np.ndarray:
    """Binary prediction via two-threshold hysteresis.

    A bout starts when prob rises above `high` and ends when it falls below `low`.
    NaN values (uncovered tail) keep the current bout state rather than ending it.

    Returns boolean array of same length as probs.
    """
    binary = np.zeros(len(probs), dtype=bool)
    in_bout = False
    for i, p in enumerate(probs):
        if np.isnan(p):
            binary[i] = in_bout
            continue
        if not in_bout and p >= high:
            in_bout = True
        elif in_bout and p < low:
            in_bout = False
        binary[i] = in_bout
    return binary


def fill_gaps(binary: np.ndarray, fs: float = FS, max_gap_s: float = 2.0) -> np.ndarray:
    """Fill non-chewing gaps shorter than max_gap_s seconds between two chewing bouts.

    Gaps at the very start or end of the recording are never filled.
    """
    max_gap = int(max_gap_s * fs)
    result = binary.copy()
    seen_chewing = False
    in_gap = False
    gap_start = 0

    for i, v in enumerate(binary):
        if v:
            if in_gap and seen_chewing:
                if (i - gap_start) <= max_gap:
                    result[gap_start:i] = True
            seen_chewing = True
            in_gap = False
        else:
            if seen_chewing and not in_gap:
                gap_start = i
                in_gap = True
    return result


def remove_short_bouts(binary: np.ndarray, fs: float = FS, min_dur_s: float = 1.0) -> np.ndarray:
    """Remove chewing bouts shorter than min_dur_s seconds."""
    min_samples = int(min_dur_s * fs)
    result = binary.copy()
    in_bout = False
    bout_start = 0

    for i, v in enumerate(binary):
        if v and not in_bout:
            bout_start = i
            in_bout = True
        elif not v and in_bout:
            if (i - bout_start) < min_samples:
                result[bout_start:i] = False
            in_bout = False
    if in_bout and (len(binary) - bout_start) < min_samples:
        result[bout_start:] = False
    return result


def postprocess(
    probs: np.ndarray,
    high: float = 0.65,
    low: float = 0.35,
    max_gap_s: float = 1.0,
    min_dur_s: float = 2.0,
    fs: float = FS,
) -> np.ndarray:
    """Full pipeline: hysteresis → gap fill → min duration. Returns bool array."""
    binary = apply_hysteresis(probs, high=high, low=low)
    binary = fill_gaps(binary, fs=fs, max_gap_s=max_gap_s)
    binary = remove_short_bouts(binary, fs=fs, min_dur_s=min_dur_s)
    return binary
