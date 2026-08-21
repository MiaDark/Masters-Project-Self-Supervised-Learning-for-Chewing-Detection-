"""Bite-by-bite (bout-by-bout) comparison between the classical-ML and SSL
masked-reconstruction pipelines, under ONE shared postprocessing recipe.

Both `chewing_ssl.classical_ml.evaluate_bouts` and
`chewing_ssl.ssl.masked.evaluate_bouts` apply the same shared postprocessing
code (`chewing_ssl.common.evaluation.postprocess`/`pipeline`) with the same
tuned parameter values (docs/classical_ml.md §6a-§6c) -- but each is a
separate CLI invocation with its own ledger (or, previously, no ledger at
all for the SSL side). This script produces ONE combined per-bite ledger
with both pipelines' predictions next to the same ground-truth bite.

Unification point: by the time predictions exist, BOTH pipelines already
reduce to the same shape of artifact -- one continuous per-timestep
probability array per recording, plus the dense ground-truth label array.
  - Classical ML: `chewing_ssl.classical_ml.upscale_predictions.
    compute_upscaled_recordings` computes the (overlapping-window-averaged)
    probability trace FRESH from `--classical-ml-model-path` every run --
    no cached prediction artifact, so this always reflects whatever model
    that path currently points at.
  - SSL masked-reconstruction: `load_ssl_recordings` below reads the
    per-timestep CSVs `ssl/masked/evaluate_bouts.py` already saved
    (`checkpoints/predictions_masked/{run_name}/**/*.csv`: time_s, prob,
    pred_raw, pred_post, label) -- no model/torch needed here, since
    inference already happened once.

Everything from postprocessing onward
(`chewing_ssl.common.evaluation.postprocess`/`pipeline`) is IDENTICAL code
applied to both pipelines' traces -- so any remaining difference in the
resulting ledger is attributable to the model, not to postprocessing or
evaluation-code drift between the two pipelines' own scripts.

One parameter is deliberately allowed to differ between the two pipelines:
`--smooth-window-s` (classical ML's own default 0.0, SSL's own default 2.0s)
-- see `chewing_ssl.common.evaluation.postprocess.smooth_probs`'s docstring
for why: classical ML's probability trace is already smoothed by
`upscale_predictions.py` averaging overlapping windows, SSL's isn't, and
empirically (see docs/classical_ml.md's bite-by-bite comparison section)
giving SSL an equivalent amount of smoothing closes a real gap that has
nothing to do with model quality -- it's compensating for a structural
difference in how each pipeline arrives at its probability trace, not
tilting postprocessing in either pipeline's favor. Every OTHER
postprocessing parameter is identical for both by default.

Output:
  --out-ledger   one row per GT bite AND per predicted bite, across BOTH
                 pipelines, tagged with which pipeline/run produced it,
                 detected/missed/false_alarm status, and (for matched bites)
                 onset_lag_s / offset_lag_s / duration_delta_s.
  --out-summary  one row per pipeline, both match modes always computed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from chewing_ssl.classical_ml.upscale_predictions import compute_upscaled_recordings
from chewing_ssl.common.data.windowing import TARGET_HZ, WINDOW_LEN
from chewing_ssl.common.evaluation.pipeline import (
    add_match_args,
    add_postprocess_args,
    aggregate_alignment,
    aggregate_dual_mode,
    evaluate_recording,
    ledger_rows_for_recording,
    postprocess_config_from_args,
)
from chewing_ssl.common.evaluation.plotting import gt_panel, plot_bout_panels, pred_panel
from chewing_ssl.common.evaluation.postprocess import ENVELOPE_CHANNELS, FS, compute_envelope
from chewing_ssl.common.evaluation.reporting import print_alignment_summary

# ── Loading ──────────────────────────────────────────────────────────────────


def load_ssl_recordings(
    pred_dir: Path, interim_root: Path, envelope_window_s: float = 1.5
) -> list[dict]:
    """checkpoints/predictions_masked/{run_name}/**/*.csv -> the same shape
    compute_upscaled_recordings produces: [{dataset, participant, tag,
    segment, gt, probs, envelope}, ...].

    Path layout mirrors ssl/masked/evaluate_bouts.py's own out_csv naming:
    `{dataset}/{tag}.csv` for a single-segment recording, or
    `{dataset}/{tag}/{segment}.csv` when a recording has more than one
    gap-bounded segment.
    """
    recordings = []
    for csv_path in sorted(pred_dir.rglob("*.csv")):
        parts = csv_path.relative_to(pred_dir).with_suffix("").parts
        if len(parts) == 2:
            dataset, tag = parts
            segment = "segment_00"
        elif len(parts) == 3:
            dataset, tag, segment = parts
        else:
            print(f"  [skip] unexpected path shape: {csv_path}")
            continue

        seg_path = interim_root / dataset / tag / f"{segment}.parquet"
        if not seg_path.exists():
            print(f"  [skip] no matching interim segment for {csv_path}")
            continue

        pred_df = pd.read_csv(csv_path)
        gt = pred_df["label"].to_numpy(dtype=bool)
        probs = pred_df["prob"].to_numpy(dtype=np.float32)

        seg = pd.read_parquet(seg_path, columns=ENVELOPE_CHANNELS)
        signal_mag = np.sqrt((seg[ENVELOPE_CHANNELS].to_numpy(dtype=np.float64) ** 2).sum(axis=1))
        envelope = compute_envelope(signal_mag, fs=FS, window_s=envelope_window_s)

        # The prediction CSV and the interim segment describe the same
        # recording but come from independently-read files -- guard rather
        # than silently misalign if they ever desync by a few rows.
        n = min(len(gt), len(envelope))
        recordings.append(
            {
                "dataset": dataset,
                "participant": tag,
                "tag": tag,
                "segment": segment,
                "gt": gt[:n],
                "probs": probs[:n],
                "envelope": envelope[:n],
            }
        )
    return recordings


def plot_comparison(
    interim_root: Path,
    dataset: str,
    tag: str,
    segment: str,
    ml_result: dict,
    ssl_result: dict,
    match_mode: str,
    ml_run_name: str,
    ssl_run_name: str,
    out_path: Path,
    ylim: tuple[float, float] | None = None,
) -> None:
    """3-panel figure for one recording shared by both pipelines: ground
    truth (detected by either pipeline) / classical ML's prediction / SSL's
    prediction -- the comparison neither single-pipeline plot_predictions.py
    can produce alone, since neither has access to the other pipeline's
    predictions."""
    seg_path = interim_root / dataset / tag / f"{segment}.parquet"
    if not seg_path.exists():
        print(f"  [skip plot] no interim segment for {dataset}/{tag}/{segment}")
        return
    df = pd.read_parquet(seg_path, columns=ENVELOPE_CHANNELS)
    t = pd.Series(np.arange(len(df)) / FS)

    panels = [
        gt_panel(ml_result["gt_bouts"], title="Ground truth"),
        pred_panel(ml_result["pred_bouts"], title=f"Classical ML ({ml_run_name})"),
        pred_panel(ssl_result["pred_bouts"], title=f"SSL masked-reconstruction ({ssl_run_name})"),
    ]
    suptitle = f"{dataset}/{tag}/{segment}  --  classical ML vs. SSL masked-reconstruction"
    plot_bout_panels(
        t, df["nav_x_left_temple"], df["nav_y_left_temple"], panels, suptitle, out_path, ylim=ylim
    )
    print(f"  Saved  ->  {out_path}")


def print_summary(pipeline: str, run_name: str, dual: dict) -> None:
    print(f"\n  {pipeline} ({run_name})  GT={dual['n_gt_bouts']}  Pred={dual['n_pred_bouts']}")
    for mode in ("iou", "tolerance"):
        m = dual[mode]
        print(
            f"    {mode:<10} prec={m['precision']:.3f}  rec={m['recall']:.3f}  f1={m['f1']:.3f}  "
            f"median_iou={m['median_iou']:.3f}  (tp={m['tp']} fp={m['fp']} fn={m['fn']})"
        )


# ── Entry point ──────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interim-root", type=Path, default=Path("data/interim_normalized"))

    parser.add_argument(
        "--classical-ml-model-path",
        type=Path,
        default=Path("checkpoints/train_ml/best_model.joblib"),
        help="Predictions are computed fresh against this file every run -- no cached artifact.",
    )
    parser.add_argument(
        "--classical-ml-features-path",
        type=Path,
        default=Path("data/processed/ml_features/test.parquet"),
    )
    parser.add_argument(
        "--classical-ml-window-seconds",
        type=float,
        default=None,
        help="Must match whatever segment.py --window-seconds built --classical-ml-features-path "
        "with. Default: configs/preprocessing.yaml's value.",
    )
    parser.add_argument("--classical-ml-run-name", default="rf_4s")

    parser.add_argument(
        "--ssl-pred-dir",
        type=Path,
        default=Path("checkpoints/predictions_masked/span_bigru_w200"),
    )
    parser.add_argument("--ssl-run-name", default="span_bigru_w200")
    parser.add_argument(
        "--ssl-smooth-window-s",
        type=float,
        default=2.0,
        help="Overrides --smooth-window-s for the SSL side only. SSL's per-timestep trace "
        "has no equivalent overlap-averaging smoothing -- 2.0s is the empirically-validated "
        "default (see module docstring); F1 improves 0.674->0.731 IoU-mode, "
        "0.489->0.527 tolerance-mode on span_bigru_w200.",
    )

    # Every OTHER postprocessing parameter is shared/identical for both pipelines.
    # --smooth-window-s (added here, default 0.0) is classical ML's value --
    # its trace is already smoothed by upscale_predictions.py, so 0.0 is the
    # validated default; --ssl-smooth-window-s above overrides it for SSL only.
    add_postprocess_args(parser)
    add_match_args(parser, default_match_mode="tolerance")

    parser.add_argument(
        "--out-ledger", type=Path, default=Path("checkpoints/bite_comparison_ledger.csv")
    )
    parser.add_argument(
        "--out-summary", type=Path, default=Path("checkpoints/bite_comparison_summary.csv")
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Also write a 3-panel comparison PNG (ground truth / classical ML / SSL) for every "
        "recording BOTH pipelines have output for, to --plot-out-dir.",
    )
    parser.add_argument(
        "--plot-out-dir", type=Path, default=Path("checkpoints/plots/bite_comparison/all")
    )
    parser.add_argument(
        "--ylim",
        type=float,
        nargs=2,
        default=None,
        metavar=("LO", "HI"),
        help="--plot only: fix every panel's y-axis to [LO, HI] instead of auto-scale, "
        "e.g. --ylim -2 2. Default: auto-scale per panel.",
    )
    args = parser.parse_args()

    cfg = postprocess_config_from_args(args)  # smooth_window_s here is classical ML's value
    ml_cfg = cfg
    ssl_cfg = cfg.__class__(**{**cfg.__dict__, "smooth_window_s": args.ssl_smooth_window_s})

    print("\nBite-by-bite comparison: classical ML vs. SSL masked-reconstruction")
    print("=" * 78)
    print(
        f"  Shared postprocessing: hysteresis({cfg.high}/{cfg.low})  gap={cfg.max_gap_s}s  "
        f"dur=[{cfg.min_dur_s},{cfg.max_dur_s}]s  valley={cfg.valley_thresh}  "
        f"envelope-split={cfg.envelope_split} (prominence={cfg.envelope_prominence}, "
        f"min-piece={cfg.envelope_min_piece_dur}s)"
    )
    print(
        f"  Smoothing (the one asymmetric parameter): "
        f"classical_ml={ml_cfg.smooth_window_s}s  ssl_masked={ssl_cfg.smooth_window_s}s"
    )

    ml_window_len = (
        int(args.classical_ml_window_seconds * TARGET_HZ)
        if args.classical_ml_window_seconds
        else WINDOW_LEN
    )
    print(f"  Computing classical ML predictions fresh from {args.classical_ml_model_path} ...")
    ml_recordings = compute_upscaled_recordings(
        args.classical_ml_model_path,
        args.classical_ml_features_path,
        args.interim_root,
        ml_window_len,
        envelope_window_s=args.envelope_window_s,
    )
    ssl_recordings = load_ssl_recordings(
        args.ssl_pred_dir, args.interim_root, envelope_window_s=args.envelope_window_s
    )
    print(
        f"  Classical ML ({args.classical_ml_run_name}): {len(ml_recordings)} recordings\n"
        f"  SSL masked   ({args.ssl_run_name}): {len(ssl_recordings)} recordings"
    )

    ml_results = [
        evaluate_recording(
            r, ml_cfg, args.min_bout_dur, args.match_mode, args.iou_thresh, args.tolerance_s
        )
        for r in ml_recordings
    ]
    ssl_results = [
        evaluate_recording(
            r, ssl_cfg, args.min_bout_dur, args.match_mode, args.iou_thresh, args.tolerance_s
        )
        for r in ssl_recordings
    ]
    ml_dual = aggregate_dual_mode(ml_results)
    ssl_dual = aggregate_dual_mode(ssl_results)
    print_summary("classical_ml", args.classical_ml_run_name, ml_dual)
    print_summary("ssl_masked", args.ssl_run_name, ssl_dual)

    print(f"\n  --- {args.classical_ml_run_name} (classical_ml) ---")
    print_alignment_summary(aggregate_alignment(ml_results))
    print(f"\n  --- {args.ssl_run_name} (ssl_masked) ---")
    print_alignment_summary(aggregate_alignment(ssl_results))

    if args.out_summary:
        rows = []
        for pipeline, run_name, dual in (
            ("classical_ml", args.classical_ml_run_name, ml_dual),
            ("ssl_masked", args.ssl_run_name, ssl_dual),
        ):
            rows.append(
                {
                    "pipeline": pipeline,
                    "run_name": run_name,
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
            )
        args.out_summary.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(args.out_summary, index=False)
        print(f"\n  Summary (both pipelines, shared postprocessing) -> {args.out_summary}")

    if args.out_ledger:
        # Built directly from the results already computed above -- no need
        # to re-run evaluate_recording a second time (build_bite_ledger's
        # convenience-wrapper path would otherwise do exactly that).
        def _ledger_rows(recordings, results, extra_ident):
            rows = []
            for rec, result in zip(recordings, results):
                ident = {
                    **extra_ident,
                    **{k: rec[k] for k in ("dataset", "participant", "tag", "segment")},
                }
                rows += ledger_rows_for_recording(ident, result, args.match_mode)
            return rows

        ml_rows = _ledger_rows(
            ml_recordings,
            ml_results,
            {"pipeline": "classical_ml", "run_name": args.classical_ml_run_name},
        )
        ssl_rows = _ledger_rows(
            ssl_recordings, ssl_results, {"pipeline": "ssl_masked", "run_name": args.ssl_run_name}
        )
        combined = pd.DataFrame(ml_rows + ssl_rows)
        args.out_ledger.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(args.out_ledger, index=False)
        print(
            f"  Combined per-bite ledger ({len(combined)} rows, match_mode={args.match_mode}) "
            f"-> {args.out_ledger}"
        )

    if args.plot:
        ml_by_key = {
            (r["dataset"], r["tag"], r["segment"]): res for r, res in zip(ml_recordings, ml_results)
        }
        ssl_by_key = {
            (r["dataset"], r["tag"], r["segment"]): res
            for r, res in zip(ssl_recordings, ssl_results)
        }
        common_keys = sorted(set(ml_by_key) & set(ssl_by_key))
        print(
            f"\n  Plotting {len(common_keys)} recording(s) shared by both pipelines -> {args.plot_out_dir}/"
        )
        for dataset, tag, segment in common_keys:
            out_path = args.plot_out_dir / f"{dataset}_{tag}_{segment}.png"
            plot_comparison(
                args.interim_root,
                dataset,
                tag,
                segment,
                ml_by_key[(dataset, tag, segment)],
                ssl_by_key[(dataset, tag, segment)],
                args.match_mode,
                args.classical_ml_run_name,
                args.ssl_run_name,
                out_path,
                ylim=tuple(args.ylim) if args.ylim else None,
            )
        print(f"  Done. {len(common_keys)} plot(s) written -> {args.plot_out_dir}/")

    print("\n" + "=" * 78 + "\n")


if __name__ == "__main__":
    main()
