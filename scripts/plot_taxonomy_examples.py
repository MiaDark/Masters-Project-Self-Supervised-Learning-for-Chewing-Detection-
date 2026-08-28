"""Illustrate the six bite-outcome taxonomy labels (match/merge/split/
tangle/missed/false_alarm, common/evaluation/bouts.py's
align_bouts_sequential) with one real example of each, boxed and
annotated. Companion figure for the taxonomy definition in
Methods/Results (dissertation.tex).

Reuses the same inference/postprocessing/alignment functions
ssl/masked/evaluate_bouts.py and compare_bites.py already use -- no new
prediction, postprocessing, or clustering logic, only a new annotated view
of it. One real recording (FEIT_eating_S05/p10/segment_00) happens to
contain a clean instance of all six labels, so all six panels come from
the same recording rather than being stitched together from several.
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

from chewing_ssl.common.evaluation.bouts import align_bouts_sequential, extract_bouts
from chewing_ssl.common.evaluation.pipeline import add_postprocess_args, postprocess_config_from_args
from chewing_ssl.common.evaluation.postprocess import apply_full_postprocessing, compute_envelope
from chewing_ssl.ssl.masked.evaluate_bouts import ENVELOPE_CHANNELS, FS, infer_recording, load_finetuned_model

PALETTE = {"nav_x": "#1565C0", "nav_y": "#6A1B9A", "gt": "#757575", "pred": "#2E7D32", "box": "#C62828"}

GLOSS = {
    "match": "MATCH -- one real bite, one predicted bout, cleanly overlapping",
    "merge": "MERGE -- two real bites covered by a single predicted bout",
    "split": "SPLIT -- one real bite covered by two predicted bouts",
    "tangle": "TANGLE -- several real bites and several predicted bouts, all overlapping",
    "missed": "MISSED -- a real bite with no overlapping prediction at all",
    "false_alarm": "FALSE ALARM -- a predicted bout with no overlapping real bite",
}

# (label, cluster span_start_s, span_end_s, left_pad_s, right_pad_s), hand-picked
# for a clean, compact, non-overlapping-with-neighbours example of each label --
# see the exploratory align_bouts_sequential dump this was chosen from.
EXAMPLES = [
    ("match", 101.12, 112.04, 2.0, 4.0),
    ("merge", 347.14, 360.44, 3.0, 2.0),
    ("split", 224.74, 249.78, 0.8, 3.0),
    ("tangle", 362.66, 389.48, 1.5, 4.0),
    ("missed", 327.70, 332.18, 3.0, 2.0),
    ("false_alarm", 32.92, 38.86, 5.0, 5.0),
]


def _shade(ax, bouts: list[dict], color: str, alpha: float, y_lo: float, y_hi: float) -> None:
    for b in bouts:
        ax.axvspan(b["start_s"], b["end_s"], y_lo, y_hi, color=color, alpha=alpha, linewidth=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/finetune_masked_80_10_10/span_bigru_w200_best.pt"))
    parser.add_argument("--interim-root", type=Path, default=Path("data/interim_normalized_80_10_10"))
    parser.add_argument("--dataset", default="FEIT_eating_S05")
    parser.add_argument("--tag", default="p10")
    parser.add_argument("--segment", default="segment_00")
    parser.add_argument("--out", type=Path, default=Path("checkpoints/plots/taxonomy_examples.png"))
    add_postprocess_args(parser)
    for action in parser._actions:
        if action.dest == "smooth_window_s":
            action.default = 2.0
    args = parser.parse_args()
    cfg = postprocess_config_from_args(args)

    model, meta = load_finetuned_model(args.checkpoint)
    model.eval()

    seg_path = args.interim_root / args.dataset / args.tag / f"{args.segment}.parquet"
    df = pd.read_parquet(seg_path)
    t = np.arange(len(df)) / FS
    x = df[meta["channels"]].to_numpy(dtype=np.float32).T
    gt_binary = df["label"].to_numpy(dtype=bool)

    probs = infer_recording(model, x, window_len=meta["window_len"])
    signal_mag = np.sqrt((df[ENVELOPE_CHANNELS].to_numpy(dtype=np.float64) ** 2).sum(axis=1))
    envelope = compute_envelope(signal_mag, fs=FS, window_s=cfg.envelope_window_s)
    post_binary = apply_full_postprocessing(probs, envelope, cfg)

    gt_bouts = extract_bouts(gt_binary, fs=FS, min_dur=0.5)
    pred_bouts = extract_bouts(post_binary, fs=FS, min_dur=0.5)
    clusters = align_bouts_sequential(gt_bouts, pred_bouts)
    print(f"n_gt={len(gt_bouts)}  n_pred={len(pred_bouts)}  n_clusters={len(clusters)}")

    # Sanity-check the hand-picked EXAMPLES windows still match a real
    # cluster of that label at that span -- fails loudly if the checkpoint
    # or postprocessing recipe ever changes and shifts the bout boundaries.
    cluster_spans = {
        (c["label"], round(min(b["start_s"] for _, _, b in c["cluster"]), 1), round(max(b["end_s"] for _, _, b in c["cluster"]), 1))
        for c in clusters
    }
    for label, start_s, end_s, _, _ in EXAMPLES:
        assert (label, round(start_s, 1), round(end_s, 1)) in cluster_spans, (
            f"No {label} cluster at [{start_s}, {end_s}] any more -- re-run the exploratory "
            f"align_bouts_sequential dump and update EXAMPLES."
        )

    fig, axes = plt.subplots(3, 2, figsize=(14, 11))
    for ax, (label, start_s, end_s, left_pad, right_pad) in zip(axes.flat, EXAMPLES):
        lo, hi = start_s - left_pad, end_s + right_pad
        mask = (t >= lo) & (t <= hi)
        t_win, df_win = t[mask], df.iloc[mask]

        gt_win = [b for b in gt_bouts if b["end_s"] >= lo and b["start_s"] <= hi]
        pred_win = [b for b in pred_bouts if b["end_s"] >= lo and b["start_s"] <= hi]

        ax.plot(t_win, df_win["nav_x_left_temple"], color=PALETTE["nav_x"], linewidth=0.8, label="nav_x_left_temple")
        ax.plot(t_win, df_win["nav_y_left_temple"], color=PALETTE["nav_y"], linewidth=0.8, label="nav_y_left_temple")
        _shade(ax, gt_win, PALETTE["gt"], 0.30, 0.0, 0.5)
        _shade(ax, pred_win, PALETTE["pred"], 0.30, 0.5, 1.0)

        box = mpatches.Rectangle(
            (start_s, 0.0), end_s - start_s, 1.0,
            transform=ax.get_xaxis_transform(),
            fill=False, edgecolor=PALETTE["box"], linewidth=1.8, linestyle="--", zorder=5,
        )
        ax.add_patch(box)

        ax.set_xlim(lo, hi)
        ax.set_title(GLOSS[label], fontsize=10, loc="left")
        ax.set_ylabel("nav (normalized)", fontsize=8)
        ax.set_xlabel("Time (s)", fontsize=8)
        ax.grid(alpha=0.2)

    legend_handles = [
        mpatches.Patch(color=PALETTE["gt"], alpha=0.30, label="Ground truth (lower half)"),
        mpatches.Patch(color=PALETTE["pred"], alpha=0.30, label="Predicted bout (upper half)"),
        mpatches.Patch(facecolor="none", edgecolor=PALETTE["box"], linewidth=1.8, linestyle="--", label="Cluster extent"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Bite-outcome taxonomy: one real example of each label (masked-SSL detection)", fontsize=13, y=1.06)
    fig.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
