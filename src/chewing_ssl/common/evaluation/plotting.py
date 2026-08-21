from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

PALETTE = {
    "nav_x": "#1565C0",
    "nav_y": "#6A1B9A",
    "bout": "#2E7D32",
}


def shade_bout(ax: plt.Axes, bout: dict, t: pd.Series, color: str, alpha: float = 0.28) -> None:
    end = min(bout["end_idx"], len(t)) - 1
    ax.axvspan(t.iloc[bout["start_idx"]], t.iloc[end], color=color, alpha=alpha, linewidth=0)


def gt_panel(bouts: list[dict], title: str = "Ground truth") -> dict:
    """A panel dict for plot_bout_panels: ground-truth bouts, plainly shaded."""
    return {"bouts": bouts, "title": title}


def pred_panel(bouts: list[dict], title: str) -> dict:
    """A panel dict for plot_bout_panels: predicted bouts, plainly shaded."""
    return {"bouts": bouts, "title": title}


def plot_bout_panels(
    t: pd.Series,
    nav_x: pd.Series,
    nav_y: pd.Series,
    panels: list[dict],
    suptitle: str,
    out_path: Path,
    ylim: tuple[float, float] | None = None,
    ylabel: str = "nav (normalized)",
) -> None:
    """One figure, `len(panels)` stacked panels sharing a time axis. Each
    panel is a dict from `gt_panel`/`pred_panel`: the same raw nav_x/nav_y
    signal is plotted in every panel, with that panel's own bouts shaded
    plainly (not color-coded by match outcome). 2 panels = the original
    single-pipeline plot (GT + one prediction); 3 panels = GT + two
    pipelines' predictions side by side.

    `ylim`, if given, fixes every panel to the same (lo, hi) y-range instead
    of matplotlib's per-panel auto-scale -- makes bout SHAPE/POSITION
    comparable panel-to-panel and plot-to-plot (a recording with one large-
    amplitude spike no longer silently rescales that one panel and throws
    off the visual comparison). Default None preserves auto-scale.

    `ylabel` describes the scale of nav_x/nav_y as actually passed in --
    default assumes normalized data (the historical convention here); pass
    "nav (raw, clipped)" or similar when plotting a raw/unnormalized tree.
    """
    fig, axes = plt.subplots(len(panels), 1, figsize=(16, 3.5 * len(panels)), sharex=True)
    if len(panels) == 1:
        axes = [axes]
    fig.suptitle(suptitle, fontsize=11)

    for ax, panel in zip(axes, panels):
        ax.plot(t, nav_x, color=PALETTE["nav_x"], linewidth=0.8, label="nav_x_left_temple")
        ax.plot(t, nav_y, color=PALETTE["nav_y"], linewidth=0.8, label="nav_y_left_temple")
        for bout in panel["bouts"]:
            shade_bout(ax, bout, t, PALETTE["bout"])
        ax.set_title(panel["title"], fontsize=10, loc="left")
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(alpha=0.2)
        if ylim is not None:
            ax.set_ylim(*ylim)

    legend_handles = [mpatches.Patch(color=PALETTE["bout"], alpha=0.28, label="chewing bout")]
    axes[0].legend(handles=legend_handles, fontsize=8, loc="upper right")
    axes[-1].set_xlabel("Time", fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
