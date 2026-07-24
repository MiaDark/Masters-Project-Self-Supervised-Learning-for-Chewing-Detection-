"""Post-processing pipeline for per-timestep chewing predictions.

Applied in order:
  1. Hysteresis thresholding  — start a bout at `high`, end it at `low`
  2. Gap filling              — bridge gaps shorter than `max_gap_s` between bouts
  3. Bout splitting           — cut bouts that are too long, or that dip through
                                 `valley_thresh`, at their lowest-probability point(s)
                                 (opt-in, off unless `max_dur_s`/`valley_thresh` is given)
  4. Minimum duration         — discard bouts shorter than `min_dur_s`

A separate, raw-signal-driven splitter (`compute_envelope` +
`split_bouts_by_envelope`) is NOT part of `postprocess()` -- it needs the
original sensor signal, not just `probs` -- but is meant to run as an
additional step after it. See its docstring for why: classifier probability
from window-averaged predictions can plateau smoothly across an entire
multi-bite eating episode (window overlap washes out the pauses between
bites) even where the raw sensor signal still shows them clearly.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

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


def _bout_bounds(binary: np.ndarray) -> list[tuple[int, int]]:
    """[(start_idx, end_idx), ...] for every contiguous True run (end exclusive)."""
    bounds = []
    in_bout, start = False, 0
    for i, v in enumerate(binary):
        if v and not in_bout:
            start, in_bout = i, True
        elif not v and in_bout:
            bounds.append((start, i))
            in_bout = False
    if in_bout:
        bounds.append((start, len(binary)))
    return bounds


def _split_bout(
    binary: np.ndarray,
    probs: np.ndarray,
    start: int,
    end: int,
    max_samples: int | None,
    valley_thresh: float | None,
) -> None:
    """Recursively cut binary[start:end] at its lowest-probability point while
    it's too long OR has an interior point below valley_thresh. Mutates
    `binary` in place."""
    too_long = max_samples is not None and (end - start) > max_samples
    segment = np.nan_to_num(probs[start:end], nan=-1.0)
    local_min = float(segment.min()) if len(segment) else 1.0
    has_valley = valley_thresh is not None and local_min < valley_thresh

    if not too_long and not has_valley:
        return

    # NaN (uncovered) samples are treated as the *least* evidence of chewing --
    # a genuine data gap is a more plausible cut point than a low-but-real
    # probability -- so they win the argmin over any real value.
    cut = start + int(segment.argmin())

    # A cut at the very edge would leave one side empty and never shrink on
    # the next recursion (infinite loop) -- fall back to the midpoint.
    if cut <= start or cut >= end - 1:
        cut = start + (end - start) // 2

    binary[cut] = False
    _split_bout(binary, probs, start, cut, max_samples, valley_thresh)
    _split_bout(binary, probs, cut + 1, end, max_samples, valley_thresh)


def split_bouts(
    binary: np.ndarray,
    probs: np.ndarray,
    fs: float = FS,
    max_dur_s: float | None = 35.0,
    valley_thresh: float | None = None,
) -> np.ndarray:
    """Split bouts at their lowest-probability point(s), recursively, while a
    bout is either longer than max_dur_s or has an interior point below
    valley_thresh -- whichever fires. Either trigger can be disabled with
    None (both disabled is a no-op).

    Real chewing bouts run ~3-17s (see docs/architecture.md's EDA finding) so
    a single 35s+ predicted bout is far more likely to be two-or-more real
    bouts merged than one genuine long chew, and any interior dip toward
    valley_thresh inside an otherwise-short bout is itself evidence of a
    similar merge that just didn't happen to cross max_dur_s. Cutting at the
    point of least predicted probability recovers the most plausible split
    rather than an arbitrary midpoint. This directly targets the "merged
    blob eats every ground-truth bout it overlaps" failure mode described in
    docs/classical_ml.md §6 -- though see §6b: probability from
    window-averaged predictions can be too smooth to have any usable valley
    at all, in which case split_bouts_by_envelope is the tool that helps.
    """
    max_samples = int(max_dur_s * fs) if max_dur_s is not None else None
    result = binary.copy()
    for start, end in _bout_bounds(result):
        _split_bout(result, probs, start, end, max_samples, valley_thresh)
    return result


def compute_envelope(signal: np.ndarray, fs: float = FS, window_s: float = 1.5) -> np.ndarray:
    """Rolling-RMS activity envelope of a 1D signal (e.g. combined sensor-channel
    magnitude), same length as `signal`.

    window_s is deliberately coarser than a single chew cycle (~0.5-0.7s at
    the 1.5-2Hz chewing rhythm, docs/architecture.md) so the envelope tracks
    bite-to-bite pauses, not individual chews -- see split_bouts_by_envelope.
    """
    window = max(1, int(window_s * fs))
    kernel = np.ones(window) / window
    return np.sqrt(np.convolve(signal.astype(np.float64) ** 2, kernel, mode="same"))


def split_bouts_by_envelope(
    binary: np.ndarray,
    envelope: np.ndarray,
    fs: float = FS,
    min_piece_dur_s: float = 2.0,
    prominence_frac: float = 0.5,
) -> np.ndarray:
    """Split bouts at prominent local minima of a raw-signal activity envelope.

    Classifier probability from window-averaged predictions can stay
    smoothly elevated across an entire multi-bite eating episode -- window
    overlap washes out the pauses between bites -- even though the raw
    sensor signal still shows them clearly (see docs/classical_ml.md §6b).
    This recovers that structure directly from the signal: scipy.signal.
    find_peaks on the negated envelope, with prominence relative to that
    bout's own envelope variability (so it adapts to different recordings'
    signal amplitude) and a minimum spacing of min_piece_dur_s between
    valleys (so it targets bite-to-bite pauses, not the much faster
    single-chew ripple within one bite).

    Any post-split piece can end up shorter than min_piece_dur_s at a bout's
    very edges (the spacing constraint only bounds valley-to-valley
    distance) -- run remove_short_bouts again afterward to clean those up.
    """
    result = binary.copy()
    min_samples = max(1, int(min_piece_dur_s * fs))
    for start, end in _bout_bounds(result):
        if end - start < 2 * min_samples:
            continue
        seg = envelope[start:end]
        prominence = prominence_frac * float(np.std(seg))
        if prominence <= 0:
            continue
        valleys, _ = find_peaks(-seg, prominence=prominence, distance=min_samples)
        for v in valleys:
            result[start + int(v)] = False
    return result


def remove_short_bouts(binary: np.ndarray, fs: float = FS, min_dur_s: float = 2.0) -> np.ndarray:
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
    max_dur_s: float | None = None,
    valley_thresh: float | None = None,
    fs: float = FS,
) -> np.ndarray:
    """Full pipeline: hysteresis -> gap fill -> [bout split] -> min duration.

    max_dur_s and valley_thresh are both opt-in (None = that trigger never
    fires) so existing callers keep their current behavior; pass e.g.
    max_dur_s=35.0 and/or valley_thresh=0.35 to enable them.
    Returns bool array.
    """
    binary = apply_hysteresis(probs, high=high, low=low)
    binary = fill_gaps(binary, fs=fs, max_gap_s=max_gap_s)
    if max_dur_s is not None or valley_thresh is not None:
        binary = split_bouts(binary, probs, fs=fs, max_dur_s=max_dur_s, valley_thresh=valley_thresh)
    binary = remove_short_bouts(binary, fs=fs, min_dur_s=min_dur_s)
    return binary
