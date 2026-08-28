"""Illustrate ground-truth under-segmentation: real test recordings where a
single ground-truth-labelled "bite" actually spans several distinct chewing
episodes the model correctly resolves into separate predicted bouts.
Companion figures for the bite-outcome-taxonomy discussion (dissertation.tex,
sec:results-bitebybite) -- evidence that "split" is not purely a model
failure mode, since the ground truth itself under-segments in these cases.

Two figures, both masked-SSL predictions on the S11 cohort (same checkpoint/
predictions behind checkpoints/predict_new/s11_feit_eating_masked_ssl/):
  1. Two zoomed split examples from one recording, side by side.
  2. One long, mostly-continuous stretch from a second, much more heavily
     under-segmented recording, showing the same pattern repeating five times.

No dataset name, participant name, GT/prediction counts, false-alarm counts,
tolerance, or min-bout-duration anywhere in either figure -- titles are
deliberately minimal ("Masked SSL" only).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from chewing_ssl.common.evaluation.bouts import extract_bouts
from chewing_ssl.common.evaluation.pipeline import add_postprocess_args, postprocess_config_from_args
from chewing_ssl.common.evaluation.postprocess import apply_full_postprocessing, compute_envelope
from chewing_ssl.ssl.masked.evaluate_bouts import ENVELOPE_CHANNELS, FS, infer_recording, load_finetuned_model

PALETTE = {"nav_x": "#1565C0", "nav_y": "#6A1B9A", "gt": "#757575", "pred": "#2E7D32"}
CHECKPOINT = Path("checkpoints/finetune_masked_80_10_10/span_bigru_w200_best.pt")
INTERIM_ROOT = Path("checkpoints/predict_new/s11_feit_eating_masked_ssl")


def _run_inference(tag: str, model, meta, cfg):
    path = INTERIM_ROOT / tag / "interim_normalized" / "segment_00.parquet"
    df = pd.read_parquet(path)
    t = np.arange(len(df)) / FS
    x = df[meta["channels"]].to_numpy(dtype=np.float32).T
    gt_binary = df["label"].to_numpy(dtype=bool)
    probs = infer_recording(model, x, window_len=meta["window_len"])
    signal_mag = np.sqrt((df[ENVELOPE_CHANNELS].to_numpy(dtype=np.float64) ** 2).sum(axis=1))
    envelope = compute_envelope(signal_mag, fs=FS, window_s=cfg.envelope_window_s)
    post_binary = apply_full_postprocessing(probs, envelope, cfg)
    gt_bouts = extract_bouts(gt_binary, fs=FS, min_dur=0.5)
    pred_bouts = extract_bouts(post_binary, fs=FS, min_dur=0.5)
    return df, t, gt_binary, post_binary, gt_bouts, pred_bouts


def _shade(ax, bouts, color, alpha, y_lo, y_hi):
    for b in bouts:
        ax.axvspan(b["start_s"], b["end_s"], y_lo, y_hi, color=color, alpha=alpha, linewidth=0)


def _panel(ax, t, df, lo, hi, gt_bouts, pred_bouts):
    mask = (t >= lo) & (t <= hi)
    ax.plot(t[mask], df["nav_x_left_temple"].to_numpy()[mask], color=PALETTE["nav_x"], linewidth=0.7)
    ax.plot(t[mask], df["nav_y_left_temple"].to_numpy()[mask], color=PALETTE["nav_y"], linewidth=0.7)
    gt_win = [b for b in gt_bouts if b["end_s"] >= lo and b["start_s"] <= hi]
    pred_win = [b for b in pred_bouts if b["end_s"] >= lo and b["start_s"] <= hi]
    _shade(ax, gt_win, PALETTE["gt"], 0.30, 0.0, 0.5)
    _shade(ax, pred_win, PALETTE["pred"], 0.30, 0.5, 1.0)
    ax.set_xlim(lo, hi)
    ax.set_ylabel("nav (normalized)", fontsize=8)
    ax.grid(alpha=0.2)


def _legend(fig):
    handles = [
        mpatches.Patch(color=PALETTE["gt"], alpha=0.30, label="Ground truth (lower half)"),
        mpatches.Patch(color=PALETTE["pred"], alpha=0.30, label="Predicted bout (upper half)"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, 1.03))


def _full_recording_figure(tag: str, model, meta, cfg, out_path: Path) -> None:
    df, t, _, _, gt_bouts, pred_bouts = _run_inference(tag, model, meta, cfg)
    lo, hi = 0.0, len(df) / FS

    fig, ax = plt.subplots(1, 1, figsize=(20, 3.5))
    _panel(ax, t, df, lo, hi, gt_bouts, pred_bouts)
    ax.set_xlabel("Time (s)", fontsize=9)
    _legend(fig)
    fig.suptitle("Masked SSL", fontsize=13, y=1.16)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Wrote {out_path}")


def figure_one(model, meta, cfg, out_path: Path) -> None:
    _full_recording_figure("Teodora _eating", model, meta, cfg, out_path)


def figure_two(model, meta, cfg, out_path: Path) -> None:
    _full_recording_figure("PavelT_eating", model, meta, cfg, out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_postprocess_args(parser)
    for action in parser._actions:
        if action.dest == "smooth_window_s":
            action.default = 2.0
    args = parser.parse_args()
    cfg = postprocess_config_from_args(args)

    model, meta = load_finetuned_model(CHECKPOINT)
    model.eval()

    figure_one(model, meta, cfg, Path("checkpoints/plots/gt_undersegmentation_example1.png"))
    figure_two(model, meta, cfg, Path("checkpoints/plots/gt_undersegmentation_example2.png"))


if __name__ == "__main__":
    main()
