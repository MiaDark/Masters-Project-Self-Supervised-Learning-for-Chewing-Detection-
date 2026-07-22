from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from chewing_ssl.common.data.loaders import (
    ACCEL_COLS,
    PROX_BROW_COLS,
    PROX_CHEEK_COLS,
    PROX_TEMPLE_COLS,
    load_recording,
)

CORRELATION_CHANNELS = (
    ACCEL_COLS
    + ["nav_x_left_temple", "nav_y_left_temple", "nav_x_right_temple", "nav_y_right_temple"]
    + PROX_CHEEK_COLS
    + PROX_BROW_COLS
    + PROX_TEMPLE_COLS
)

SIGNAL_CLIP_RANGES: dict[str, tuple[float, float]] = {
    "accel_": (-5000, 5000),
    "nav_": (-50, 50),
}


def clip_signal_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """Clip accelerometer/navigation/proximity columns to valid ranges."""
    for prefix, (lo, hi) in SIGNAL_CLIP_RANGES.items():
        cols = [c for c in df.columns if c.startswith(prefix)]
        if cols:
            df[cols] = df[cols].clip(lower=lo, upper=hi)
    return df


def find_bouts(label: pd.Series, datetime: pd.Series) -> pd.DataFrame:
    """Contiguous runs of label == 1 -> one row per bout with start/end/duration."""
    lab = label.fillna(0).to_numpy() == 1
    if not lab.any():
        return pd.DataFrame(columns=["start", "end", "duration_s", "n_samples"])

    change = np.diff(lab.astype(int))
    starts = np.where(change == 1)[0] + 1
    ends = np.where(change == -1)[0] + 1
    if lab[0]:
        starts = np.insert(starts, 0, 0)
    if lab[-1]:
        ends = np.append(ends, len(lab))

    dt = datetime.to_numpy()
    rows = [
        {
            "start": dt[s],
            "end": dt[e - 1],
            "duration_s": (dt[e - 1] - dt[s]) / np.timedelta64(1, "s"),
            "n_samples": e - s,
        }
        for s, e in zip(starts, ends)
    ]
    return pd.DataFrame(rows)


def sampling_diagnostics(datetime: pd.Series) -> dict:
    t = datetime.to_numpy().astype("datetime64[ns]").astype(np.int64) / 1e9
    dt = np.diff(t)
    dt = dt[dt > 0]
    if len(dt) == 0:
        return {}
    median_dt = float(np.median(dt))
    return {
        "median_dt_s": median_dt,
        "estimated_hz": 1.0 / median_dt if median_dt > 0 else None,
        "n_gaps_gt_3x_median": int(np.sum(dt > 3 * median_dt)),
        "max_gap_s": float(dt.max()),
        "p99_dt_s": float(np.percentile(dt, 99)),
    }


def dead_channel_report(df: pd.DataFrame, cols: list[str]) -> dict:
    out = {}
    for c in cols:
        if c not in df.columns:
            continue
        s = df[c].dropna()
        if s.empty:
            continue
        std = float(s.std())
        out[c] = {
            "std": std,
            "frac_zero": float((s == 0).mean()),
            "likely_dead_or_disconnected": bool(std < 1e-6),
        }
    return out


def summarize(df: pd.DataFrame, label_col: str = "label") -> dict:
    bouts = find_bouts(df[label_col], df["datetime"]) if label_col in df.columns else pd.DataFrame()
    summary = {
        "n_rows": len(df),
        "duration_minutes": (df["datetime"].max() - df["datetime"].min()).total_seconds() / 60,
        "columns": sorted(df.columns),
        #"sampling": sampling_diagnostics(df["datetime"]),
        "missing_fraction": {
            c: float(df[c].isna().mean()) for c in df.columns if df[c].isna().any()
        },
        "dead_channels": dead_channel_report(df, ACCEL_COLS + CORRELATION_CHANNELS),
    }
    if label_col in df.columns:
        counts = df[label_col].fillna(0).value_counts().to_dict()
        summary["label_balance"] = {str(k): int(v) for k, v in counts.items()}
        summary["n_bouts"] = len(bouts)
        if not bouts.empty:
            summary["bout_duration_s"] = {
                "mean": float(bouts["duration_s"].mean()),
                "median": float(bouts["duration_s"].median()),
                "min": float(bouts["duration_s"].min()),
                "max": float(bouts["duration_s"].max()),
            }
    return summary


def _panels_for(df: pd.DataFrame) -> list[tuple[str, list[str]]]:
    panels = []
    if all(c in df.columns for c in ACCEL_COLS):
        panels.append(("Accelerometer (X/Y/Z)", ACCEL_COLS))
    nav_x = [c for c in ["nav_x_left_temple", "nav_x_right_temple"] if c in df.columns]
    nav_y = [c for c in ["nav_y_left_temple", "nav_y_right_temple"] if c in df.columns]
    if nav_x:
        panels.append(("Nav Temple X", nav_x))
    if nav_y:
        panels.append(("Nav Temple Y", nav_y))
    prox_cheek = [c for c in PROX_CHEEK_COLS if c in df.columns]
    if prox_cheek:
        panels.append(("Proximity Cheek", prox_cheek))
    prox_brow = [c for c in PROX_BROW_COLS if c in df.columns]
    if prox_brow:
        panels.append(("Proximity Brow", prox_brow))
    prox_temple = [c for c in PROX_TEMPLE_COLS if c in df.columns]
    if prox_temple:
        panels.append(("Proximity Temple", prox_temple))
    return panels


def plot_timeseries(
    df: pd.DataFrame, out_path: Path, title: str, label_col: str = "label", max_points: int = 200_000
) -> None:
    panels = _panels_for(df)
    if not panels:
        return

    plot_df = df
    if len(df) > max_points:
        stride = len(df) // max_points + 1
        plot_df = df.iloc[::stride]

    bouts = find_bouts(df[label_col], df["datetime"]) if label_col in df.columns else pd.DataFrame()

    fig, axes = plt.subplots(len(panels), 1, sharex=True, figsize=(14, 2.2 * len(panels)))
    axes = np.atleast_1d(axes)
    for ax, (panel_title, cols) in zip(axes, panels):
        for c in cols:
            ax.plot(plot_df["datetime"], plot_df[c], linewidth=0.6, label=c)
        for _, b in bouts.iterrows():
            ax.axvspan(b["start"], b["end"], color="orange", alpha=0.25, linewidth=0)
        ax.set_ylabel(panel_title, fontsize=8)
        ax.legend(fontsize=6, loc="upper right", ncol=len(cols))
    axes[-1].set_xlabel("Time")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_zoom(
    df: pd.DataFrame, out_path: Path, title: str, label_col: str = "label", pad_minutes: float = 3.0
) -> None:
    """Full-resolution view around the longest bout, to see fine chewing rhythm."""
    bouts = find_bouts(df[label_col], df["datetime"]) if label_col in df.columns else pd.DataFrame()
    if bouts.empty:
        return
    longest = bouts.loc[bouts["duration_s"].idxmax()]
    center = longest["start"] + (longest["end"] - longest["start"]) / 2
    lo, hi = center - pd.Timedelta(minutes=pad_minutes), center + pd.Timedelta(minutes=pad_minutes)
    window = df[(df["datetime"] >= lo) & (df["datetime"] <= hi)]
    if window.empty:
        return
    plot_timeseries(window, out_path, title, label_col=label_col, max_points=200_000)


def plot_class_balance(df: pd.DataFrame, out_path: Path, title: str, label_col: str = "label") -> None:
    if label_col not in df.columns:
        return
    counts = df[label_col].fillna(0).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.bar([str(int(i)) for i in counts.index], counts.values, color=["steelblue", "darkorange"])
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v}\n({v / counts.sum():.1%})", ha="center", va="bottom", fontsize=9)
    ax.set_xlabel(label_col)
    ax.set_ylabel("count")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_bout_durations(df: pd.DataFrame, out_path: Path, title: str, label_col: str = "label") -> None:
    if label_col not in df.columns:
        return
    bouts = find_bouts(df[label_col], df["datetime"])
    if bouts.empty:
        return
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(bouts["duration_s"], bins=30, color="darkorange", edgecolor="black")
    ax.set_xlabel("Bout duration (s)")
    ax.set_ylabel("count")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_sampling_intervals(df: pd.DataFrame, out_path: Path, title: str) -> None:
    t = df["datetime"].to_numpy().astype("datetime64[ns]").astype(np.int64) / 1e9
    dt = np.diff(t)
    dt = dt[dt > 0]
    if len(dt) == 0:
        return
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(dt, bins=60, color="slategray", edgecolor="black")
    ax.set_yscale("log")
    ax.set_xlabel("Δt between samples (s)")
    ax.set_ylabel("count (log)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_psd_comparison(
    df: pd.DataFrame,
    out_path: Path,
    title: str,
    signal_col: str,
    fs: float,
    label_col: str = "label",
) -> None:
    from scipy.signal import welch

    if signal_col not in df.columns or label_col not in df.columns or not fs:
        return
    chew = df.loc[df[label_col] == 1, signal_col].dropna().to_numpy()
    rest = df.loc[df[label_col] == 0, signal_col].dropna().to_numpy()
    if len(chew) < fs * 2 or len(rest) < fs * 2:
        return
    f1, p1 = welch(chew, fs=fs, nperseg=min(len(chew), int(fs * 4)))
    f2, p2 = welch(rest, fs=fs, nperseg=min(len(rest), int(fs * 4)))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(f1, p1, label="chewing")
    ax.semilogy(f2, p2, label="non-chewing")
    ax.set_xlim(0, min(10, fs / 2))
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_correlation_heatmaps(
    df: pd.DataFrame, out_path: Path, title: str, channel_cols: list[str], label_col: str = "label"
) -> None:
    cols = [c for c in channel_cols if c in df.columns]
    if len(cols) < 2 or label_col not in df.columns:
        return
    chew = df.loc[df[label_col] == 1, cols].dropna()
    rest = df.loc[df[label_col] == 0, cols].dropna()
    if len(chew) < 10 or len(rest) < 10:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    im = None
    for ax, sub_df, sub_title in zip(axes, [chew, rest], ["chewing", "non-chewing"]):
        corr = sub_df.corr()
        im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(cols, rotation=90, fontsize=6)
        ax.set_yticks(range(len(cols)))
        ax.set_yticklabels(cols, fontsize=6)
        ax.set_title(sub_title)
    fig.colorbar(im, ax=axes.tolist(), shrink=0.7)
    fig.suptitle(title)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def discover_recordings(dataset: str, root: Path) -> list[tuple[Path, str]]:
    """Find every recording CSV for a dataset under root, paired with a short tag.

    Mirrors the per-dataset folder layouts documented in docs/data.md:
    SIPM is one file per participant folder, Sussex one file per participant's
    Eating/ folder, Spectacle one file per participant per recording day.
    """
    items: list[tuple[Path, str]] = []
    if dataset == "SIPMEatingStudy":
        for p in sorted((root / dataset).glob("p*")):
            csvs = sorted(p.glob("sensor_data_*.csv"))
            if csvs:
                items.append((csvs[0], p.name))
    elif dataset in ("SussexEatingUK", "FEIT_eating_S05", "Office_eating_s05"):
        for p in sorted((root / dataset).glob("*")):
            csvs = sorted((p / "Eating").glob("*.csv")) if (p / "Eating").is_dir() else []
            if csvs:
                items.append((csvs[0], p.name))
    elif dataset == "DiverseNormative":
        for p in sorted((root / dataset).glob("p*")):
            csvs = sorted(p.glob("*.csv"))
            if csvs:
                items.append((csvs[0], p.name))
    elif dataset == "Spectacle":
        for p in sorted((root / dataset).glob("week*/p*")):
            for day in sorted(p.glob("year_*")):
                csv = day / "daily_annotated.csv"
                if csv.exists():
                    items.append((csv, f"{p.parent.name}_{p.name}_{day.name}"))
    elif dataset == "Spectacle_NonMeal":
        # Unlike DiverseNormative, each participant folder holds multiple
        # snippet CSVs (see scripts/extract_spectacle_nonmeal.py) -> take all.
        for p in sorted((root / dataset).glob("p*")):
            for csv in sorted(p.glob("*.csv")):
                items.append((csv, f"{p.name}_{csv.stem}"))
    else:
        raise ValueError(f"Unknown dataset {dataset!r}")
    return items


def _flatten_summary(dataset: str, tag: str, summary: dict) -> dict:
    """One row per recording, for the cross-recording aggregate_summary.csv."""
    sampling = summary.get("sampling", {})
    bout = summary.get("bout_duration_s", {})
    dead = [
        c for c, v in summary.get("dead_channels", {}).items() if v.get("likely_dead_or_disconnected")
    ]
    balance = summary.get("label_balance", {})
    n_pos = sum(v for k, v in balance.items() if float(k) == 1.0)
    n_total = sum(balance.values()) if balance else None
    return {
        "dataset": dataset,
        "tag": tag,
        "n_rows": summary.get("n_rows"),
        "duration_minutes": summary.get("duration_minutes"),
        "estimated_hz": sampling.get("estimated_hz"),
        "n_gaps_gt_3x_median": sampling.get("n_gaps_gt_3x_median"),
        "max_gap_s": sampling.get("max_gap_s"),
        "label_positive_frac": (n_pos / n_total) if n_total else None,
        "n_bouts": summary.get("n_bouts"),
        "bout_mean_s": bout.get("mean"),
        "bout_median_s": bout.get("median"),
        "dead_channels": ";".join(dead),
    }


def run_batch(datasets: list[str], root: Path, out_root: Path, zoom_minutes: float) -> None:
    """Run the single-file pipeline over every recording found for each dataset."""
    for dataset in datasets:
        recordings = discover_recordings(dataset, root)
        print(f"\n=== {dataset}: {len(recordings)} file(s) found under {root / dataset} ===")
        rows = []
        for i, (path, tag) in enumerate(recordings, 1):
            out_dir = out_root / dataset / tag
            print(f"[{i}/{len(recordings)}] {dataset}/{tag}")
            try:
                summary = run(dataset, path, out_dir, tag, zoom_minutes, verbose=False)
                rows.append(_flatten_summary(dataset, tag, summary))
            except Exception as exc:
                print(f"  FAILED: {exc}")
                rows.append({"dataset": dataset, "tag": tag, "error": str(exc)})
        if rows:
            agg_path = out_root / dataset / "aggregate_summary.csv"
            pd.DataFrame(rows).to_csv(agg_path, index=False)
            print(f"Wrote {agg_path}")


def run(dataset: str, path: Path, out_dir: Path, tag: str, zoom_minutes: float, verbose: bool = True) -> dict:
    df = load_recording(dataset, path)
    df = clip_signal_ranges(df)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = summarize(df)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    title_prefix = f"{dataset} / {tag}"
    plot_timeseries(df, out_dir / "overview.png", f"{title_prefix} - overview")
    #plot_zoom(df, out_dir / "zoom.png", f"{title_prefix} - zoom on longest bout", pad_minutes=zoom_minutes)
    #plot_class_balance(df, out_dir / "class_balance.png", f"{title_prefix} - label balance")
    #plot_bout_durations(df, out_dir / "bout_durations.png", f"{title_prefix} - bout durations")
    #plot_sampling_intervals(df, out_dir / "sampling_intervals.png", f"{title_prefix} - sampling intervals")

    fs = summary.get("sampling", {}).get("estimated_hz")
    
    if verbose:
        print(json.dumps(summary, indent=2, default=str))
    print(f"Wrote plots and summary.json to {out_dir}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="EDA plots + summary for OCOsense recording CSVs.")
    parser.add_argument(
        "--dataset",
        choices=[
            "SIPMEatingStudy",
            "SussexEatingUK",
            "Spectacle",
            "Spectacle_NonMeal",
            "DiverseNormative",
            "FEIT_eating_S05",
            "Office_eating_s05",
        ],
        help="Required unless --all is given.",
    )
    parser.add_argument("--path", type=Path, help="Path to one recording CSV (single-file mode).")
    parser.add_argument("--out", type=Path, default=Path("src/chewing_ssl/data/reports/eda"))
    parser.add_argument("--tag", default=None, help="Subfolder name; defaults to the file stem.")
    parser.add_argument("--zoom-minutes", type=float, default=3.0)
    parser.add_argument(
        "--all", action="store_true", help="Run over every recording found under --root for --dataset(s)."
    )
    parser.add_argument(
        "--root", type=Path, default=Path("data/raw"), help="Root containing the per-dataset folders."
    )
    args = parser.parse_args()

    if args.all:
        datasets = [args.dataset] if args.dataset else [
            "SIPMEatingStudy",
            "SussexEatingUK",
            "Spectacle",
            "DiverseNormative",
            "FEIT_eating_S05",
            "Office_eating_s05",
        ]
        run_batch(datasets, args.root, args.out, args.zoom_minutes)
        return

    if not args.dataset or not args.path:
        parser.error("--dataset and --path are required unless --all is given.")

    tag = args.tag or args.path.stem
    out_dir = args.out / args.dataset / tag
    run(args.dataset, args.path, out_dir, tag, args.zoom_minutes)


if __name__ == "__main__":
    main()
