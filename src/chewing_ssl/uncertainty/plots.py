"""Figures for docs/uncertainty.md §8, built directly from the per-recording
`.npz` arrays (`checkpoints/uncertainty/<run_name>_<participant>.npz`) and
`checkpoints/uncertainty_summary.csv` that `evaluate_uncertainty.py`/
`evaluate_uncertainty_rf.py` already produce -- no new evaluation runs.

Color roles follow the project's dataviz convention: categorical slot 1
(blue, `#2a78d6`) = masked_ssl, slot 2 (orange, `#eb6834`) = classical_ml,
assigned in fixed order and never swapped between figures; status colors
(`#0ca30c` good / `#d03b3b` critical) for the binary correct/wrong split,
never reused as a third series color. Chart chrome (surface/ink/gridline)
matches the same reference palette. `common/evaluation/plotting.PALETTE`
(nav_x/nav_y/bout) is reused as-is for the recording-trace figure so it
matches every other per-recording plot already in this project
(`plot_predictions.py`).

Run from the project root:
  python -m chewing_ssl.uncertainty.plots --out-dir checkpoints/plots/uncertainty
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from chewing_ssl.common.evaluation.plotting import PALETTE as TRACE_PALETTE
from chewing_ssl.uncertainty.calibration import reliability_diagram

# -- dataviz reference palette (references/palette.md) --
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"
SSL_COLOR = "#2a78d6"  # categorical slot 1 (blue)
RF_COLOR = "#eb6834"  # categorical slot 2 (orange)
GOOD_COLOR = "#0ca30c"  # status: correct
CRITICAL_COLOR = "#d03b3b"  # status: wrong

PIPELINE_COLOR = {"masked_ssl": SSL_COLOR, "classical_ml": RF_COLOR}
PIPELINE_LABEL = {"masked_ssl": "masked_ssl (MC-Dropout)", "classical_ml": "classical_ml (tree ensemble)"}

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "axes.edgecolor": BASELINE,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "axes.grid": True,
        "grid.color": GRIDLINE,
        "grid.linewidth": 0.7,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
    }
)


def _save(fig: plt.Figure, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out_path.with_suffix('.pdf')}  (+ .png)")


def _pool_run(uncertainty_dir: Path, run_name: str) -> dict:
    """Concatenate every recording's covered timesteps for one run_name's
    .npz files into flat arrays -- the same pooling
    evaluate_uncertainty[_rf].py does internally, just recomputed from the
    saved per-recording arrays instead of held in memory."""
    files = sorted(uncertainty_dir.glob(f"{run_name}_*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files matching {run_name}_*.npz in {uncertainty_dir}")
    means, labels, mis, ents = [], [], [], []
    for f in files:
        d = np.load(f)
        covered = ~np.isnan(d["mean"])
        means.append(d["mean"][covered])
        labels.append(d["label"][covered])
        mis.append(d["mutual_info"][covered])
        ents.append(d["entropy"][covered])
    return {
        "mean": np.concatenate(means),
        "label": np.concatenate(labels),
        "mutual_info": np.concatenate(mis),
        "entropy": np.concatenate(ents),
    }


# -- Figure 1: reliability diagram --------------------------------------------------


def plot_reliability_diagram(
    uncertainty_dir: Path, run_names: dict[str, str], out_path: Path, n_bins: int = 15
) -> None:
    """run_names: {pipeline_key: run_name}, e.g. {"masked_ssl": "ssl_global_all",
    "classical_ml": "rf_global_all"}. One line per pipeline against the
    perfect-calibration diagonal."""
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.plot([0, 1], [0, 1], color=BASELINE, linewidth=1.5, linestyle="--", label="Perfect calibration", zorder=1)

    for pipeline, run_name in run_names.items():
        pooled = _pool_run(uncertainty_dir, run_name)
        rel = reliability_diagram(pooled["mean"], pooled["label"], n_bins=n_bins)
        valid = rel["bin_count"] > 0
        ax.plot(
            rel["bin_confidence"][valid],
            rel["bin_accuracy"][valid],
            marker="o",
            markersize=5,
            linewidth=2,
            color=PIPELINE_COLOR[pipeline],
            label=f"{PIPELINE_LABEL[pipeline]}  (ECE={rel['ece']:.3f})",
            zorder=2,
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted confidence (bin)")
    ax.set_ylabel("Observed positive rate (bin)")
    ax.set_title("Calibration: confidence vs. observed accuracy", fontsize=11, loc="left")
    ax.legend(fontsize=8.5, loc="upper left", frameon=False)
    ax.set_aspect("equal")
    _save(fig, out_path)


# -- Figure 2: 12-config grid --------------------------------------------------------


def plot_config_grid(summary_csv: Path, out_path: Path) -> None:
    """3 side-by-side panels (bout F1 / ECE / AUROC-epistemic), one grouped
    bar chart each, x-axis = the 6 norm x channel configs, paired bars per
    pipeline -- three different-scale metrics, so three panels (one axis
    each), never one dual-axis chart."""
    df = pd.read_csv(summary_csv)
    df = df[df["run_name"] != "span_bigru_w200_mc30"]  # exclude the pre-grid demo row
    config_order = ["global_all", "global_nav", "global_accel", "per_file_all", "per_file_nav", "per_file_accel"]
    df["config"] = df["norm"] + "_" + df["channels"]
    df["config"] = pd.Categorical(df["config"], categories=config_order, ordered=True)
    df = df.sort_values("config")

    metrics = [
        ("bout_iou_f1", "Bout F1 (IoU >= 0.3)", (0, 1)),
        ("ece", "Expected calibration error (lower better)", None),
        ("auroc_mi_vs_error", "AUROC(epistemic uncertainty -> error)", (0.5, 1)),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    x = np.arange(len(config_order))
    width = 0.36

    for ax, (col, title, ylim) in zip(axes, metrics):
        for i, pipeline in enumerate(["masked_ssl", "classical_ml"]):
            sub = df[df["pipeline"] == pipeline].set_index("config").reindex(config_order)
            offset = (i - 0.5) * width
            ax.bar(
                x + offset, sub[col].to_numpy(), width=width * 0.94,
                color=PIPELINE_COLOR[pipeline], label=PIPELINE_LABEL[pipeline],
            )
        ax.set_xticks(x)
        ax.set_xticklabels([c.replace("_", "\n") for c in config_order], fontsize=8)
        ax.set_title(title, fontsize=9.5, loc="left")
        if ylim:
            ax.set_ylim(*ylim)
        ax.grid(axis="y", alpha=0.6)
        ax.grid(axis="x", visible=False)
        ax.set_axisbelow(True)

    axes[0].legend(fontsize=8, loc="lower left", frameon=False)
    fig.suptitle("All 12 configs: accuracy, calibration, and error-detection side by side", fontsize=11, x=0.01, ha="left")
    _save(fig, out_path)


# -- Figure 3: uncertainty distribution, correct vs. wrong --------------------------


def plot_uncertainty_distributions(uncertainty_dir: Path, run_names: dict[str, str], out_path: Path) -> None:
    """One panel per pipeline (different absolute MI scale -- see
    docs/uncertainty.md finding 1 -- so shared x-axis would flatten one of
    them); density histogram of epistemic uncertainty (mutual_info) split
    by whether that timestep's point prediction was correct."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, (pipeline, run_name) in zip(axes, run_names.items()):
        pooled = _pool_run(uncertainty_dir, run_name)
        preds = (pooled["mean"] >= 0.5).astype(int)
        correct = preds == pooled["label"].astype(int)
        mi = pooled["mutual_info"]

        bins = np.linspace(0, np.quantile(mi, 0.995), 60)
        ax.hist(mi[correct], bins=bins, density=True, color=GOOD_COLOR, alpha=0.55, label=f"correct (n={correct.sum():,})")
        ax.hist(mi[~correct], bins=bins, density=True, color=CRITICAL_COLOR, alpha=0.55, label=f"wrong (n={(~correct).sum():,})")
        ax.axvline(mi[correct].mean(), color=GOOD_COLOR, linewidth=1.5, linestyle="--")
        ax.axvline(mi[~correct].mean(), color=CRITICAL_COLOR, linewidth=1.5, linestyle="--")
        ax.set_title(PIPELINE_LABEL[pipeline], fontsize=10, loc="left")
        ax.set_xlabel("Epistemic uncertainty (mutual information)")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8, frameon=False)

    fig.suptitle("Epistemic uncertainty is higher where the model is wrong (dashed = group mean)", fontsize=11, x=0.01, ha="left")
    _save(fig, out_path)


# -- Figure 4: per-recording trace (the EMTEQRAP15 case) ----------------------------


def plot_recording_trace(npz_path: Path, out_path: Path, title: str) -> None:
    """Two stacked panels sharing a time axis: (a) mean probability with GT
    shaded underneath, same visual language as plot_predictions.py's own
    GT-shading convention (common.evaluation.plotting.PALETTE["bout"]); (b)
    epistemic (mutual_info) and total (entropy) uncertainty traces. Point of
    the figure: show a "confidently wrong" stretch directly, rather than
    just quoting its summary statistic (docs/uncertainty.md finding 5)."""
    d = np.load(npz_path)
    fs = 50.0
    t = np.arange(len(d["mean"])) / fs

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True, height_ratios=[1.3, 1])

    gt = d["label"].astype(bool)
    edges = np.flatnonzero(np.diff(np.concatenate([[0], gt.astype(int), [0]])))
    for start, end in zip(edges[0::2], edges[1::2]):
        ax1.axvspan(t[start], t[min(end, len(t) - 1)], color=TRACE_PALETTE["bout"], alpha=0.28, linewidth=0)

    ax1.plot(t, d["mean"], color=SSL_COLOR, linewidth=1.0, label="mean predicted probability")
    ax1.fill_between(t, d["mean"] - d["std"], d["mean"] + d["std"], color=SSL_COLOR, alpha=0.2, linewidth=0, label="+/- 1 std (MC-Dropout)")
    ax1.axhline(0.5, color=BASELINE, linewidth=1, linestyle=":")
    ax1.set_ylabel("P(chewing)")
    ax1.set_ylim(-0.02, 1.02)
    ax1.set_title(title, fontsize=11, loc="left")
    legend_handles = [
        mpatches.Patch(color=TRACE_PALETTE["bout"], alpha=0.28, label="ground-truth chewing bout"),
    ]
    ax1.legend(handles=ax1.get_legend_handles_labels()[0] + legend_handles, fontsize=8, loc="upper right")

    ax2.plot(t, d["entropy"], color=INK_SECONDARY, linewidth=1.0, label="total uncertainty (entropy)")
    ax2.plot(t, d["mutual_info"], color=CRITICAL_COLOR, linewidth=1.2, label="epistemic uncertainty (mutual info)")
    ax2.set_ylabel("Uncertainty")
    ax2.set_xlabel("Time (s)")
    ax2.legend(fontsize=8, loc="upper right")

    _save(fig, out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uncertainty-dir", type=Path, default=Path("checkpoints/uncertainty"))
    parser.add_argument("--summary-csv", type=Path, default=Path("checkpoints/uncertainty_summary.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("checkpoints/plots/uncertainty"))
    args = parser.parse_args()

    best = {"masked_ssl": "ssl_global_all", "classical_ml": "rf_global_all"}

    print("Figure 1: reliability diagram")
    plot_reliability_diagram(args.uncertainty_dir, best, args.out_dir / "01_reliability_diagram")

    print("Figure 2: 12-config grid")
    plot_config_grid(args.summary_csv, args.out_dir / "02_config_grid")

    print("Figure 3: uncertainty distribution, correct vs. wrong")
    plot_uncertainty_distributions(args.uncertainty_dir, best, args.out_dir / "03_uncertainty_distributions")

    print("Figure 4: EMTEQRAP15 trace (masked_ssl, global/all)")
    plot_recording_trace(
        args.uncertainty_dir / "ssl_global_all_SussexEatingUK_EMTEQRAP15_segment_00.npz",
        args.out_dir / "04_emteqrap15_trace",
        title="SussexEatingUK/EMTEQRAP15 -- masked_ssl, global/all (22/23 bites missed, confidently)",
    )

    print(f"\nAll figures written to {args.out_dir}/ (.pdf + .png)")


if __name__ == "__main__":
    main()
