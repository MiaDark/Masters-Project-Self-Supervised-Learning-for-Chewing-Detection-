from __future__ import annotations

import numpy as np


def _prf1(tp: float, fp: float, fn: float) -> dict:
    prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    rec = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else float("nan")
    return {"precision": prec, "recall": rec, "f1": f1}


def _tp_fp_fn(results: list[dict], stats_key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tp = np.array([r[stats_key]["tp"] for r in results], dtype=float)
    fp = np.array([r[stats_key]["fp"] for r in results], dtype=float)
    fn = np.array([r[stats_key]["fn"] for r in results], dtype=float)
    return tp, fp, fn


def bootstrap_metric(
    results: list[dict],
    stats_key: str = "tolerance_stats",
    metric: str = "f1",
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict:
    """Percentile bootstrap CI for one pooled metric (precision/recall/f1),
    resampling recordings with replacement.

    `results` is a list of per-recording dicts as returned by
    `pipeline.evaluate_recording` (each with `iou_stats`/`tolerance_stats`
    sub-dicts carrying `tp`/`fp`/`fn`). `stats_key` selects which match mode;
    `metric` selects precision/recall/f1 within it.

    Returns `{point, ci_lo, ci_hi, ci, n_boot, n_recordings}` -- `point` is
    the real (non-bootstrapped) value on the full recording set, exactly
    matching what `aggregate_dual_mode` would report; `ci_lo`/`ci_hi` bound
    the bootstrap distribution's central `ci` fraction (e.g. 0.95 -> 2.5th
    and 97.5th percentiles).
    """
    tp, fp, fn = _tp_fp_fn(results, stats_key)
    n = len(results)
    point = _prf1(tp.sum(), fp.sum(), fn.sum())[metric]

    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        draws[i] = _prf1(tp[idx].sum(), fp[idx].sum(), fn[idx].sum())[metric]

    alpha = (1 - ci) / 2
    lo, hi = np.nanpercentile(draws, [100 * alpha, 100 * (1 - alpha)])
    return {
        "point": point,
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "ci": ci,
        "n_boot": n_boot,
        "n_recordings": n,
    }


def align_by_id(
    results_a: list[dict], results_b: list[dict], id_key: str = "_id"
) -> tuple[list[dict], list[dict]]:
    """Inner-join two result lists on `id_key` (e.g. "dataset/tag/segment"),
    returning both lists re-ordered to match -- needed before a paired
    comparison across two pipelines whose recording lists aren't
    guaranteed to be built in the same order (or, rarely, don't cover the
    exact same recordings, e.g. one dropped a zero-bout Spectacle_NonMeal
    snippet the other kept)."""
    by_id_a = {r[id_key]: r for r in results_a}
    by_id_b = {r[id_key]: r for r in results_b}
    common = sorted(set(by_id_a) & set(by_id_b))
    dropped = (set(by_id_a) | set(by_id_b)) - set(common)
    if dropped:
        print(f"  [align_by_id] {len(dropped)} recording(s) not common to both sides, dropped: "
              f"{sorted(dropped)[:5]}{'...' if len(dropped) > 5 else ''}")
    return [by_id_a[i] for i in common], [by_id_b[i] for i in common]


def bootstrap_compare(
    results_a: list[dict],
    results_b: list[dict],
    stats_key: str = "tolerance_stats",
    metric: str = "f1",
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict:
    """Paired bootstrap for `metric_a - metric_b`. `results_a`/`results_b`
    MUST already be the same recordings in the same order (use
    `align_by_id` first if they come from different pipelines/loaders) --
    each bootstrap draw resamples ONE shared set of recording indices for
    both sides, so the comparison controls for which recordings happened to
    be easy/hard that draw, rather than treating A and B as independent
    samples (which would overstate the variance of the difference).

    Returns `point_a`, `point_b`, `point_diff` (= point_a - point_b, real
    values, not bootstrapped), a percentile CI on the difference, a
    two-sided bootstrap p-value (2x the smaller tail beyond zero, capped at
    1.0), and `significant` (True iff the CI excludes zero at this `ci`
    level).
    """
    if len(results_a) != len(results_b):
        raise ValueError(
            f"results_a/results_b must be the same length (aligned recordings) -- "
            f"got {len(results_a)} vs {len(results_b)}. Use align_by_id first."
        )
    tp_a, fp_a, fn_a = _tp_fp_fn(results_a, stats_key)
    tp_b, fp_b, fn_b = _tp_fp_fn(results_b, stats_key)
    n = len(results_a)

    point_a = _prf1(tp_a.sum(), fp_a.sum(), fn_a.sum())[metric]
    point_b = _prf1(tp_b.sum(), fp_b.sum(), fn_b.sum())[metric]

    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        va = _prf1(tp_a[idx].sum(), fp_a[idx].sum(), fn_a[idx].sum())[metric]
        vb = _prf1(tp_b[idx].sum(), fp_b[idx].sum(), fn_b[idx].sum())[metric]
        diffs[i] = va - vb

    alpha = (1 - ci) / 2
    lo, hi = np.nanpercentile(diffs, [100 * alpha, 100 * (1 - alpha)])
    p_value = min(2 * min((diffs <= 0).mean(), (diffs >= 0).mean()), 1.0)

    return {
        "point_a": point_a,
        "point_b": point_b,
        "point_diff": point_a - point_b,
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "ci": ci,
        "p_value": float(p_value),
        "significant": not (lo <= 0 <= hi),
        "n_boot": n_boot,
        "n_recordings": n,
    }
