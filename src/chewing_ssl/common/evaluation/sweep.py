"""Hyperparameter sweep over the shared postprocessing recipe
(`chewing_ssl.common.evaluation.postprocess.PostprocessConfig`).

Reuses the exact same `evaluate_recording`/`aggregate_dual_mode` machinery
every evaluation script already calls (`chewing_ssl.common.evaluation.
pipeline`) -- a sweep is just "apply the shared postprocess-and-match
sequence with many different configs to the same, already-computed
(probs, gt, envelope) per recording". The expensive step (model inference /
window-probability upscaling) is each pipeline's own responsibility and runs
exactly ONCE, before any of this module is called; everything in here is
cheap (pure numpy/scipy over cached arrays) and safe to repeat hundreds of
times.

Two entry points:
  - `sweep_postprocessing` -- a plain full grid (itertools.product over
    whatever fields you give it). Fine for a handful of parameters or a
    small number of values each; a full grid over all 8 tunable fields at
    ~5 values apiece is 5**8 (~390,000) combos and NOT what you want.
  - `coordinate_descent_sweep` -- tunes one field at a time, holding the
    others at the current running-best, exactly the approach already used
    and documented for this project's SSL hyperparameter tuning
    (docs/ssl_masked.md §6: "coordinate descent, not a full grid"). Costs
    sum(len(choices)) combos instead of a product -- a few dozen total for
    the default grid below -- while still covering every parameter.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import fields, replace
from typing import Any

import pandas as pd

from chewing_ssl.common.evaluation.pipeline import aggregate_dual_mode, evaluate_recording
from chewing_ssl.common.evaluation.postprocess import PostprocessConfig

_CFG_FIELDS = {f.name for f in fields(PostprocessConfig)}

# envelope_window_s is deliberately excluded from both grids below: unlike
# every other field, it changes what `compute_envelope` produces, not just
# how the already-rolled envelope is used -- sweeping it would mean
# recomputing `rec["envelope"]` per value, which the recording-loading step
# (compute_upscaled_recordings / load_test_recordings) has already fixed by
# the time these functions run. Nothing in this project's own postprocessing
# tuning history (docs/dissertation_classical_ml_sections.md §4.3, Finding 6)
# ever varied it either -- only envelope_prominence/envelope_min_piece_dur
# were found to matter.
COORDINATE_DESCENT_GRID: dict[str, list[Any]] = {
    "high": [0.60, 0.65, 0.70, 0.75, 0.80, 0.85],
    "low": [0.30, 0.35, 0.45, 0.52, 0.60],
    "max_gap_s": [0.5, 1.0, 1.5, 2.0],
    "max_dur_s": [20.0, 25.0, 30.0, 35.0, 40.0],
    "valley_thresh": [None, 0.25, 0.35, 0.45],
    "envelope_prominence": [1.5, 2.0, 2.5, 3.0, 3.5],
    "envelope_min_piece_dur": [4.0, 6.0, 8.0, 10.0],
    "smooth_window_s": [0.0, 1.0, 2.0, 3.0],
}


def _validate_grid(grid: dict[str, list]) -> None:
    unknown = set(grid) - _CFG_FIELDS
    if unknown:
        raise ValueError(f"Unknown PostprocessConfig field(s) in grid: {sorted(unknown)}")
    if "envelope_window_s" in grid:
        raise ValueError(
            "envelope_window_s can't be swept here -- it changes what "
            "compute_envelope produces, not just how the already-rolled "
            "envelope is used, and every `rec['envelope']` passed in is "
            "already fixed at whatever window_s the recording-loading step "
            "used. Re-run that step per envelope_window_s value instead."
        )


def sweep_postprocessing(
    recordings: list[dict],
    grid: dict[str, list[Any]],
    base_cfg: PostprocessConfig | None = None,
    min_bout_dur: float = 0.5,
    iou_thresh: float = 0.3,
    tolerance_s: float = 5.0,
    verbose: bool = True,
) -> pd.DataFrame:
    """Try every combination in `grid` (itertools.product over its values)
    against `recordings`, under BOTH match modes, and return one row per
    combo: the swept parameter values plus bout_iou_{precision,recall,f1,
    median_iou} and bout_tolerance_{precision,recall,f1,median_iou}.

    `grid` keys must be `PostprocessConfig` field names, e.g.
    `{"high": [0.65, 0.75], "low": [0.45, 0.52]}`. Any combo where the
    resulting config would have `low >= high`, or `max_dur_s < min_dur_s`,
    is silently skipped (not a valid hysteresis band / duration bound) --
    the returned row count may be less than the full product size; the
    console summary reports how many were skipped.

    `recordings` must already carry `gt`/`probs`(/`envelope` if any combo
    has `envelope_split=True`) -- the same shape `evaluate_recording`
    expects (see `chewing_ssl.common.evaluation.pipeline`'s module
    docstring), typically built once via `compute_upscaled_recordings`
    (classical ML) or `load_test_recordings` (masked SSL).
    """
    _validate_grid(grid)
    base_cfg = base_cfg or PostprocessConfig()

    keys = list(grid)
    combos = list(itertools.product(*(grid[k] for k in keys)))

    rows = []
    for combo in combos:
        overrides = dict(zip(keys, combo))
        cfg = replace(base_cfg, **overrides)
        if cfg.low >= cfg.high:
            continue
        if cfg.max_dur_s is not None and cfg.max_dur_s < cfg.min_dur_s:
            continue

        results = [
            evaluate_recording(rec, cfg, min_bout_dur, "iou", iou_thresh, tolerance_s)
            for rec in recordings
        ]
        dual = aggregate_dual_mode(results)

        row = {
            **overrides,
            "n_gt_bouts": dual["n_gt_bouts"],
            "n_pred_bouts": dual["n_pred_bouts"],
            "bout_iou_precision": dual["iou"]["precision"],
            "bout_iou_recall": dual["iou"]["recall"],
            "bout_iou_f1": dual["iou"]["f1"],
            "bout_iou_median_iou": dual["iou"]["median_iou"],
            "bout_tolerance_precision": dual["tolerance"]["precision"],
            "bout_tolerance_recall": dual["tolerance"]["recall"],
            "bout_tolerance_f1": dual["tolerance"]["f1"],
            "bout_tolerance_median_iou": dual["tolerance"]["median_iou"],
        }
        rows.append(row)
        if verbose:
            print(
                f"    {overrides}  ->  iou_f1={dual['iou']['f1']:.3f}  "
                f"tolerance_f1={dual['tolerance']['f1']:.3f}"
            )

    df = pd.DataFrame(rows)
    skipped = len(combos) - len(rows)
    if verbose:
        print(
            f"    ({len(rows)}/{len(combos)} combos valid, {skipped} skipped: low>=high or max_dur<min_dur)"
        )
    return df


def coordinate_descent_sweep(
    recordings: list[dict],
    param_grid: dict[str, list[Any]],
    base_cfg: PostprocessConfig | None = None,
    *,
    min_bout_dur: float = 0.5,
    iou_thresh: float = 0.3,
    tolerance_s: float = 5.0,
    metric: str = "bout_tolerance_f1",
    param_order: list[str] | None = None,
    verbose: bool = True,
) -> tuple[PostprocessConfig, pd.DataFrame]:
    """Tune one `PostprocessConfig` field at a time -- holding every other
    field at the current running-best -- rather than a full cross-product
    grid. Same approach already used for this project's SSL hyperparameter
    tuning (docs/ssl_masked.md §6): a full grid over `COORDINATE_DESCENT_GRID`
    (8 fields, ~4-6 values each) would be tens of thousands of combos; this
    costs `sum(len(choices) for choices in param_grid.values())` instead --
    a few dozen for the default grid -- while still covering every field.

    `metric` picks which of the two headline numbers (`bout_iou_f1` or
    `bout_tolerance_f1`) each step optimizes for. The two criteria are
    sensitive to different failure modes (IoU cares about overlap fraction,
    tolerance about absolute boundary error -- see docs/
    dissertation_classical_ml_sections.md §4.3) and can disagree on a
    winner; picking one to drive the search doesn't stop the returned
    `history` from reporting BOTH for every combo tried, so you can always
    check what the road not taken would have scored.

    Not guaranteed to find the joint optimum -- parameters can interact
    (§4.3 Finding 6: a window-size change required re-tuning hysteresis and
    envelope-splitting together) -- but this mirrors the project's own
    validated tuning methodology rather than inventing a new one, and is
    cheap enough to re-run per model/split.

    Returns `(best_cfg, history)`: `best_cfg` is the final tuned config;
    `history` is every combo tried across every parameter, in order, tagged
    by which parameter was being tuned at the time (`_tuned_param` column).
    """
    cfg = replace(base_cfg or PostprocessConfig())
    order = param_order or list(param_grid)
    history_rows = []

    for param in order:
        choices = param_grid[param]
        if verbose:
            print(f"\n  Tuning {param} (holding everything else at current best) ...")
        df = sweep_postprocessing(
            recordings,
            {param: choices},
            cfg,
            min_bout_dur=min_bout_dur,
            iou_thresh=iou_thresh,
            tolerance_s=tolerance_s,
            verbose=verbose,
        )
        if df.empty:
            if verbose:
                print(f"    (no valid combo for {param} -- skipping, keeping current value)")
            continue
        df.insert(0, "_tuned_param", param)
        history_rows.append(df)

        best_row = df.loc[df[metric].idxmax()]
        best_value = best_row[param]
        # pandas silently upcasts a column mixing `None` with floats to NaN
        # (e.g. valley_thresh's [None, 0.25, 0.35, 0.45] choices) -- restore
        # the semantic `None` ("this trigger is disabled") rather than
        # feeding a NaN sentinel into PostprocessConfig, which would *look*
        # equivalent (any `x < NaN` comparison is False, same effective
        # behavior as None) but is the wrong value and breaks JSON output.
        if isinstance(best_value, float) and math.isnan(best_value) and None in choices:
            best_value = None
        cfg = replace(cfg, **{param: best_value})
        if verbose:
            print(f"    -> {param} = {best_value}  ({metric} = {best_row[metric]:.4f})")

    history = pd.concat(history_rows, ignore_index=True) if history_rows else pd.DataFrame()
    return cfg, history


def summarize_best(df: pd.DataFrame, top_n: int = 5) -> None:
    """Print the top-N combos by each of the two headline metrics, and the
    single best row for each -- flags whether the two criteria agree on a
    winner (they often don't, see `coordinate_descent_sweep`'s docstring)."""
    if df.empty:
        print("No valid combos to summarize.")
        return
    top_n = min(top_n, len(df))
    print(f"\nTop {top_n} by bout_iou_f1:")
    print(df.sort_values("bout_iou_f1", ascending=False).head(top_n).to_string(index=False))
    print(f"\nTop {top_n} by bout_tolerance_f1:")
    print(df.sort_values("bout_tolerance_f1", ascending=False).head(top_n).to_string(index=False))

    best_iou = df.loc[df["bout_iou_f1"].idxmax()]
    best_tol = df.loc[df["bout_tolerance_f1"].idxmax()]
    same_winner = best_iou.equals(best_tol)
    print(f"\nBest by IoU-mode F1:       {best_iou.to_dict()}")
    print(f"Best by tolerance-mode F1: {best_tol.to_dict()}")
    print(
        f"\n{'Same combo wins both criteria.' if same_winner else 'The two criteria pick DIFFERENT combos -- see which tradeoff you care about more.'}"
    )
