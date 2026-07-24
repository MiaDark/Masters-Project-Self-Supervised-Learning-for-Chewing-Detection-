from __future__ import annotations

import numpy as np

from chewing_ssl.common.evaluation.postprocess import (
    apply_hysteresis,
    compute_envelope,
    fill_gaps,
    postprocess,
    remove_short_bouts,
    split_bouts,
    split_bouts_by_envelope,
)


def _bout_lengths(binary: np.ndarray) -> list[int]:
    lengths, in_bout, start = [], False, 0
    for i, v in enumerate(binary):
        if v and not in_bout:
            start, in_bout = i, True
        elif not v and in_bout:
            lengths.append(i - start)
            in_bout = False
    if in_bout:
        lengths.append(len(binary) - start)
    return lengths


def test_apply_hysteresis_basic_start_stop():
    probs = np.array([0.1, 0.7, 0.7, 0.4, 0.2, 0.1])
    binary = apply_hysteresis(probs, high=0.65, low=0.35)
    assert binary.tolist() == [False, True, True, True, False, False]


def test_fill_gaps_bridges_short_gap_not_long_gap():
    fs = 10.0
    # bout(5) gap(5=0.5s) bout(5) ... gap(30=3s) ... bout(5)
    binary = np.array([True] * 5 + [False] * 5 + [True] * 5 + [False] * 30 + [True] * 5, dtype=bool)
    filled = fill_gaps(binary, fs=fs, max_gap_s=1.0)
    lengths = _bout_lengths(filled)
    assert lengths == [15, 5]  # short gap bridged into one 15-sample bout, long gap untouched


def test_fill_gaps_never_fills_leading_or_trailing_gap():
    binary = np.array([False, False, True, True, False, False], dtype=bool)
    filled = fill_gaps(binary, fs=10.0, max_gap_s=10.0)
    assert filled.tolist() == binary.tolist()


def test_remove_short_bouts_drops_under_threshold():
    fs = 10.0
    binary = np.array([True] * 5 + [False] * 5 + [True] * 20, dtype=bool)  # 0.5s then 2.0s
    result = remove_short_bouts(binary, fs=fs, min_dur_s=1.0)
    assert _bout_lengths(result) == [20]


def test_split_bouts_cuts_at_lowest_probability_point():
    fs = 10.0
    binary = np.ones(40, dtype=bool)  # one 4.0s bout
    probs = np.full(40, 0.9)
    probs[20] = 0.1  # the one weak point in the middle
    result = split_bouts(binary, probs, fs=fs, max_dur_s=2.0)  # cap at 2.0s = 20 samples
    assert not result[20]  # cut sample itself removed
    lengths = _bout_lengths(result)
    assert lengths == [20, 19]
    assert max(lengths) <= 20


def test_split_bouts_recurses_until_within_limit():
    fs = 10.0
    binary = np.ones(90, dtype=bool)  # 9.0s bout, needs 2+ cuts to fit under 2.0s pieces
    probs = np.full(90, 0.9)
    probs[30] = 0.1
    probs[60] = 0.05
    result = split_bouts(binary, probs, fs=fs, max_dur_s=2.0)
    lengths = _bout_lengths(result)
    # every resulting piece must respect the cap ...
    assert all(length <= 20 for length in lengths)
    # ... and the two deliberately-weak points are always cut first, whatever
    # further (flat-probability, edge-guarded) splits the recursion needs to
    # bring the remaining pieces under the cap.
    assert not result[30]
    assert not result[60]


def test_split_bouts_prefers_nan_gap_as_cut_point():
    fs = 10.0
    binary = np.ones(40, dtype=bool)
    probs = np.full(40, 0.9)
    probs[15] = np.nan  # data gap -- least evidence of chewing, should be preferred over 0.3
    probs[25] = 0.3
    result = split_bouts(binary, probs, fs=fs, max_dur_s=2.0)
    assert not result[15]


def test_split_bouts_noop_when_under_limit_and_no_valley():
    fs = 10.0
    binary = np.ones(15, dtype=bool)
    probs = np.full(15, 0.9)
    result = split_bouts(binary, probs, fs=fs, max_dur_s=2.0)
    assert result.tolist() == binary.tolist()


def test_split_bouts_max_dur_none_disables_duration_trigger():
    fs = 10.0
    binary = np.ones(90, dtype=bool)  # 9.0s, would be split under a duration cap
    probs = np.full(90, 0.9)
    result = split_bouts(binary, probs, fs=fs, max_dur_s=None, valley_thresh=None)
    assert result.tolist() == binary.tolist()  # both triggers off -> no-op


def test_split_bouts_valley_thresh_splits_short_bout_with_a_dip():
    # A 1.5s bout, well under any duration cap, but with a real interior dip
    # below valley_thresh -- this is the case max_dur_s alone can't catch:
    # 2-3 short real bouts merged into one blob still under 35s.
    fs = 10.0
    binary = np.ones(15, dtype=bool)
    probs = np.full(15, 0.9)
    probs[7] = 0.1
    result = split_bouts(binary, probs, fs=fs, max_dur_s=100.0, valley_thresh=0.35)
    assert not result[7]
    lengths = _bout_lengths(result)
    assert lengths == [7, 7]


def test_split_bouts_valley_thresh_no_split_when_no_dip_crosses_it():
    fs = 10.0
    binary = np.ones(15, dtype=bool)
    probs = np.full(15, 0.9)
    probs[7] = 0.5  # a dip, but not below valley_thresh
    result = split_bouts(binary, probs, fs=fs, max_dur_s=100.0, valley_thresh=0.35)
    assert result.tolist() == binary.tolist()


def test_split_bouts_duration_and_valley_triggers_combine():
    # Duration trigger fires on the whole bout; valley trigger then also
    # fires inside a still-too-long piece -- both mechanisms cooperate.
    fs = 10.0
    binary = np.ones(60, dtype=bool)  # 6.0s
    probs = np.full(60, 0.9)
    probs[20] = 0.9  # no real valley here
    probs[45] = 0.1  # real valley here
    result = split_bouts(binary, probs, fs=fs, max_dur_s=3.0, valley_thresh=0.35)
    lengths = _bout_lengths(result)
    assert all(length <= 30 for length in lengths)  # duration cap respected
    assert not result[45]  # valley still cut


def test_postprocess_max_dur_is_opt_in():
    fs = 10.0
    probs = np.full(90, 0.9)
    probs[45] = 0.4  # a dip, but still above the default low=0.35 -- hysteresis
    # alone must NOT break the bout here, so any split below is attributable
    # only to the max_dur_s step.
    without_cap = postprocess(probs, max_gap_s=0.0, min_dur_s=0.1, fs=fs)
    assert _bout_lengths(without_cap) == [90]
    with_cap = postprocess(probs, max_gap_s=0.0, min_dur_s=0.1, max_dur_s=3.0, fs=fs)
    assert all(length <= 30 for length in _bout_lengths(with_cap))
    assert sum(_bout_lengths(with_cap)) < 90  # confirms a split actually happened


def test_postprocess_valley_thresh_is_opt_in():
    fs = 10.0
    probs = np.full(30, 0.9)
    probs[15] = 0.5  # above the default low=0.35, so hysteresis alone keeps
    # the bout continuous -- any split below is attributable only to valley_thresh.
    without_valley = postprocess(probs, max_gap_s=0.0, min_dur_s=0.1, fs=fs)
    assert _bout_lengths(without_valley) == [30]
    with_valley = postprocess(probs, max_gap_s=0.0, min_dur_s=0.1, valley_thresh=0.6, fs=fs)
    assert _bout_lengths(with_valley) == [15, 14]


def test_compute_envelope_is_high_during_activity_low_during_silence():
    fs = 50.0
    rng = np.random.default_rng(0)
    silence = rng.normal(0, 0.05, size=int(1.0 * fs))
    burst = rng.normal(0, 3.0, size=int(1.0 * fs))
    signal = np.concatenate([silence, burst, silence])
    envelope = compute_envelope(signal, fs=fs, window_s=0.3)
    assert envelope.shape == signal.shape
    mid = len(silence) + len(burst) // 2
    assert envelope[mid] > 5 * envelope[len(silence) // 2]


def test_split_bouts_by_envelope_splits_at_prominent_valley():
    fs = 50.0
    rng = np.random.default_rng(1)
    burst = lambda n: rng.normal(0, 3.0, size=n)  # noqa: E731
    pause = lambda n: rng.normal(0, 0.05, size=n)  # noqa: E731
    signal = np.concatenate([burst(150), pause(150), burst(150)])  # 3s / 3s / 3s
    envelope = compute_envelope(signal, fs=fs, window_s=0.3)
    binary = np.ones(len(signal), dtype=bool)

    result = split_bouts_by_envelope(
        binary, envelope, fs=fs, min_piece_dur_s=1.0, prominence_frac=0.5
    )
    assert not result.all()  # a cut happened
    lengths = _bout_lengths(result)
    assert len(lengths) >= 2


def test_split_bouts_by_envelope_respects_min_piece_duration():
    # Two very short (0.3s) high-activity blips close together should NOT be
    # split apart, since the resulting pieces would be far under
    # min_piece_dur_s -- the `distance` constraint on find_peaks should
    # suppress that valley.
    fs = 50.0
    rng = np.random.default_rng(2)
    blip = lambda n: rng.normal(0, 3.0, size=n)  # noqa: E731
    dip = lambda n: rng.normal(0, 0.05, size=n)  # noqa: E731
    signal = np.concatenate([blip(15), dip(10), blip(15)])  # 0.3s/0.2s/0.3s
    envelope = compute_envelope(signal, fs=fs, window_s=0.1)
    binary = np.ones(len(signal), dtype=bool)

    result = split_bouts_by_envelope(
        binary, envelope, fs=fs, min_piece_dur_s=2.0, prominence_frac=0.5
    )
    assert result.tolist() == binary.tolist()  # too close together to split


def test_split_bouts_by_envelope_noop_on_short_bout():
    fs = 50.0
    envelope = np.ones(10)
    binary = np.ones(10, dtype=bool)
    result = split_bouts_by_envelope(binary, envelope, fs=fs, min_piece_dur_s=2.0)
    assert result.tolist() == binary.tolist()
