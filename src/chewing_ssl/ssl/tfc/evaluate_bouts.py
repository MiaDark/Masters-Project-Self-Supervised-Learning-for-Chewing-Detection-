"""Bout-by-bout evaluation of a finetune_window.py (TF-C / TF-FC) checkpoint.

The window-level analogue of chewing_ssl.ssl.masked.evaluate_bouts, filling
the gap left by finetune_window.py / visualize_window_inference.py: neither
produces a bout-level number, only window-level classification metrics
(accuracy/F1/AUC over independent 4s-or-whatever-window-length chunks). Bout-
level detection quality -- did we find the right chewing episodes at
roughly the right times -- is what's actually comparable to
chewing_ssl.classical_ml.evaluate_bouts and chewing_ssl.ssl.masked.evaluate_bouts,
neither of which this pretext task had a counterpart for before this script.

Bridging window-level -> per-timestep, for a WindowClassifier
------------------------------------------------------------
Unlike the masked-reconstruction ChewingDetector (a per-timestep sequence
head), finetune_window.py's WindowClassifier emits exactly ONE probability
per window (its embeddings are pooled by construction -- see
chewing_ssl.ssl.tfc.pretrain_tfc's module docstring). To get a continuous
per-timestep timeline for bout extraction, this script slides
non-overlapping windows over the full recording (matching finetune_test's
own no-overlap eval convention, chewing_ssl.common.data.windowing) and PAINTS
each window's single probability across every timestep it covers --
analogous to chewing_ssl.classical_ml.upscale_predictions, but simpler since
non-overlapping eval windows never need the overlap-averaging that
script's 50%-overlapping windows require.

Run from the project root, after fine-tuning at least one checkpoint:
  python -m chewing_ssl.ssl.tfc.evaluate_bouts \\
      --checkpoint checkpoints/finetune_window_tfc/tfc_best.pt
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from chewing_ssl.common.data.preprocess import SENSOR_CHANNELS
from chewing_ssl.common.data.windowing import TARGET_HZ, WINDOW_LEN
from chewing_ssl.common.evaluation.bouts import extract_bouts, match_bouts, match_bouts_tolerant
from chewing_ssl.common.evaluation.postprocess import postprocess
from chewing_ssl.common.evaluation.reporting import fmt, print_matched_pairs, print_summary_table
from chewing_ssl.ssl.tfc.finetune_window import WindowClassifier
from chewing_ssl.ssl.tfc.visualize_window_inference import build_encoder, infer_pretext_task

FS = 50.0  # Hz


# ── Inference ──────────────────────────────────────────────────────────────────


@torch.no_grad()
def infer_recording(model: WindowClassifier, x_full: np.ndarray, window_len: int) -> np.ndarray:
    """Slide non-overlapping window_len windows over (C, N), and paint each
    window's single scalar probability across every timestep it covers.

    No overlap-averaging needed (unlike classical_ml.upscale_predictions):
    finetune_test/val windows never overlap by construction, so each covered
    timestep belongs to exactly one window. Any tail shorter than window_len
    is left as NaN (uncovered), same convention as masked.evaluate_bouts.
    """
    C, N = x_full.shape
    probs = np.full(N, np.nan, dtype=np.float32)
    starts = list(range(0, N - window_len + 1, window_len))
    if not starts:
        return probs
    batch = np.stack([x_full[:, s : s + window_len] for s in starts])  # (n_windows, C, window_len)
    logits = model(torch.from_numpy(batch))  # (n_windows,)
    win_probs = torch.sigmoid(logits).numpy()
    for s, p in zip(starts, win_probs):
        probs[s : s + window_len] = p
    return probs


# ── Per-recording evaluation ────────────────────────────────────────────────────


def evaluate_recording(
    model: WindowClassifier,
    seg_path: Path,
    label: str,
    pred_dir: Path,
    window_len: int,
    match_mode: str,
    iou_thresh: float,
    tolerance_s: float,
    min_bout_dur: float,
    threshold: float,
    post_high: float,
    post_low: float,
    post_max_gap_s: float,
    post_min_dur_s: float,
) -> dict | None:
    if not seg_path.exists():
        print(f"  [skip] segment not found: {seg_path}")
        return None

    df = pd.read_parquet(seg_path)
    if "label" not in df.columns:
        print(f"  [skip] no label column in {seg_path}")
        return None

    x = df[SENSOR_CHANNELS].to_numpy(dtype=np.float32).T  # (C, N)
    y = df["label"].to_numpy(dtype=np.float32)  # (N,)

    probs = infer_recording(model, x, window_len=window_len)

    preds_post = postprocess(
        probs, high=post_high, low=post_low, max_gap_s=post_max_gap_s, min_dur_s=post_min_dur_s
    ).astype(np.float32)
    preds_raw = (probs >= threshold).astype(np.float32)
    preds_raw[np.isnan(probs)] = 0.0

    out_csv = pred_dir / f"{label}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time_s", "prob", "pred_raw", "pred_post", "label"])
        for i, (pr, raw, post, gt) in enumerate(zip(probs, preds_raw, preds_post, y)):
            writer.writerow(
                [f"{i / FS:.4f}", "" if np.isnan(pr) else f"{pr:.6f}", int(raw), int(post), int(gt)]
            )

    gt_bouts = extract_bouts(y.astype(bool), min_dur=min_bout_dur)
    pred_bouts = extract_bouts(preds_post.astype(bool), min_dur=min_bout_dur)

    if match_mode == "tolerance":
        result = match_bouts_tolerant(gt_bouts, pred_bouts, tolerance_s=tolerance_s)
    else:
        result = match_bouts(gt_bouts, pred_bouts, iou_thresh=iou_thresh)
    tp, fp, fn = result["tp"], result["fp"], result["fn"]

    prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    rec = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else float("nan")

    matched = result["matches"]
    ious = [m["iou"] for m in matched]
    onset_lag = [(m["pred"]["start_s"] - m["gt"]["start_s"]) for m in matched]
    dur_delta = [(m["pred"]["duration_s"] - m["gt"]["duration_s"]) for m in matched]

    return {
        "participant": label,
        "n_gt_bouts": len(gt_bouts),
        "n_pred_bouts": len(pred_bouts),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "median_iou": float(np.median(ious)) if ious else float("nan"),
        "median_onset_lag_s": float(np.median(onset_lag)) if onset_lag else float("nan"),
        "median_dur_delta_s": float(np.median(dur_delta)) if dur_delta else float("nan"),
        "matches": matched,
    }


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bout-by-bout evaluation of a TF-C/TF-FC finetune_window.py checkpoint."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--time-architecture",
        default="cnn",
        help="Only used if --checkpoint's pretext_task is tfc (default matches "
        "pretrain_tfc.py's own default).",
    )
    parser.add_argument("--resnet-base-width", type=int, default=32)
    parser.add_argument("--head-hidden-dim", type=int, default=64)
    parser.add_argument("--head-dropout", type=float, default=0.3)
    parser.add_argument(
        "--interim-root",
        type=Path,
        default=Path("data/interim_normalized"),
        help="Must be the NORMALIZED tree -- the model was trained on globally z-scored input.",
    )
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--dataset", default=None, help="Restrict to one dataset. Default: all datasets."
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=None,
        help="Window length the checkpoint was fine-tuned on, if different from "
        "configs/preprocessing.yaml's default (e.g. 2.0 for a 2s-window model). "
        "Must match --processed-root's window size.",
    )
    parser.add_argument("--pred-dir", type=Path, default=Path("checkpoints/predictions_tfc"))
    parser.add_argument("--match-mode", choices=["iou", "tolerance"], default="iou")
    parser.add_argument("--iou-thresh", type=float, default=0.3)
    parser.add_argument("--tolerance-s", type=float, default=5.0)
    parser.add_argument("--min-bout-dur", type=float, default=0.5)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--post-high", type=float, default=0.65)
    parser.add_argument("--post-low", type=float, default=0.35)
    parser.add_argument("--post-max-gap", type=float, default=2.0)
    parser.add_argument("--post-min-dur", type=float, default=1.0)
    parser.add_argument("--show-pairs", action="store_true")
    args = parser.parse_args()
    window_len = int(args.window_seconds * TARGET_HZ) if args.window_seconds else WINDOW_LEN

    if not args.checkpoint.exists():
        print(f"Checkpoint not found: {args.checkpoint}")
        return

    pretext_task = infer_pretext_task(args.checkpoint)
    n_channels = len(SENSOR_CHANNELS)
    encoder = build_encoder(pretext_task, n_channels, args.time_architecture, args.resnet_base_width)
    model = WindowClassifier(encoder, args.head_hidden_dim, args.head_dropout)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    print("\nBout-by-Bout Evaluation (TF-C / TF-FC)")
    print("=" * 60)
    print(f"  Checkpoint    : {args.checkpoint}  [{pretext_task}]")
    print(f"  Window length : {window_len} samples ({window_len / TARGET_HZ:.1f}s)")
    print(f"  Match mode    : {args.match_mode}")
    print(
        f"  Post-process  : hysteresis ({args.post_high}/{args.post_low})  "
        f"gap-fill {args.post_max_gap}s  min-dur {args.post_min_dur}s"
    )
    print(
        f"  Loaded        : val_f1={state.get('val_f1', float('nan')):.3f}  "
        f"epoch={state.get('epoch', 0) + 1}\n"
    )

    idx = pd.read_parquet(args.processed_root / "index" / "finetune_test.parquet")
    if args.dataset:
        idx = idx[idx["dataset"] == args.dataset]
    recordings = sorted(idx[["dataset", "tag"]].drop_duplicates().itertuples(index=False))
    print(f"  Test recordings: {len(recordings)}  (datasets: {sorted(idx['dataset'].unique())})\n")

    results = []
    for dataset, tag in recordings:
        seg_dir = args.interim_root / dataset / tag
        seg_paths = sorted(seg_dir.glob("segment_*.parquet"))
        for seg_path in seg_paths:
            label = f"{dataset}/{tag}" + (f"/{seg_path.stem}" if len(seg_paths) > 1 else "")
            print(f"  Processing {label} ...", end=" ", flush=True)
            r = evaluate_recording(
                model,
                seg_path,
                label,
                args.pred_dir,
                window_len=window_len,
                match_mode=args.match_mode,
                iou_thresh=args.iou_thresh,
                tolerance_s=args.tolerance_s,
                min_bout_dur=args.min_bout_dur,
                threshold=args.threshold,
                post_high=args.post_high,
                post_low=args.post_low,
                post_max_gap_s=args.post_max_gap,
                post_min_dur_s=args.post_min_dur,
            )
            if r is not None:
                r["_iou_thresh"] = args.iou_thresh
                r["_tolerance_s"] = args.tolerance_s
                r["_min_dur"] = args.min_bout_dur
                results.append(r)
                print(
                    f"GT={r['n_gt_bouts']} bouts  Pred={r['n_pred_bouts']} bouts  "
                    f"TP={r['tp']} FP={r['fp']} FN={r['fn']}  F1={fmt(r['f1'])}"
                )

    if results:
        print_summary_table(results, match_mode=args.match_mode)
        if args.show_pairs:
            print_matched_pairs(results)
        print(f"\n  Per-timestep predictions saved to: {args.pred_dir}/")

    print()


if __name__ == "__main__":
    main()
