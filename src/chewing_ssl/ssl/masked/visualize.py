"""Visualize SSL pretext tasks on real sensor data from a single participant.

Produces two PNG files in --out-dir:

  transforms_demo.png
      Five-panel figure showing one real chewing window and what each pretext
      transform does to it: original signal, after masking, Arrow of Time
      (temporal flip), Permutation (shuffled chunks), Time-warping.
      Can be generated without any trained checkpoint.

  reconstruction_demo.png
      For each trained checkpoint found in --checkpoint-dir, shows the original
      signal, the masked input the model saw, and the model's reconstruction at
      every timestep (with masked positions highlighted).
      Only produced if at least one checkpoint exists.

Run from the project root:

  # Before training (transforms only):
  python -m chewing_ssl.ssl.masked.visualize

  # After training (adds reconstruction panel):
  python -m chewing_ssl.ssl.masked.visualize \\
      --checkpoint-dir checkpoints/pretrain \\
      --architectures cnn bigru
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from chewing_ssl.common.data.preprocess import SENSOR_CHANNELS
from chewing_ssl.ssl.models import ARCHITECTURES
from chewing_ssl.ssl.models.masking import apply_mask_token, sample_mask_batch


CHANNEL_LABELS = {
    "nav_x_left_temple": "Nav X\nLeft Temple",
    "nav_y_left_temple": "Nav Y\nLeft Temple",
    "prox_left_cheek": "Prox\nLeft Cheek",
    "prox_right_cheek": "Prox\nRight Cheek",
    "accel_x": "Accel X",
    "accel_y": "Accel Y",
    "accel_z": "Accel Z",
}

# Three channels most visually distinctive during chewing
REPR_CHANNELS = ["nav_x_left_temple", "prox_left_cheek", "accel_z"]

MODEL_KWARGS = {
    "cnn": {"base_width": 12},
    "bigru": {},
}

ARCH_COLORS = {"cnn": "#C62828", "bigru": "#E65100"}
ARCH_LABELS = {"cnn": "CNN U-Net", "bigru": "Bi-GRU"}


# ── Data loading ──────────────────────────────────────────────────────────────


def find_chewing_window(
    interim_root: Path,
    dataset: str,
    participant: str,
    variant: str,
    window_len: int = 200,
    min_chew_frac: float = 0.75,
) -> tuple[np.ndarray, str, int]:
    """Return (x, participant, start_row) for the first high-chewing window found.

    Falls back to the first available participant/segment if the requested one
    is not found or has no labels.
    """
    seg_path = interim_root / variant / dataset / participant / "segment_00.parquet"

    if not seg_path.exists():
        print(f"  {seg_path} not found — scanning for another participant ...")
        for p_dir in sorted((interim_root / variant / dataset).iterdir()):
            if p_dir.is_dir() and list(p_dir.glob("segment_*.parquet")):
                participant = p_dir.name
                seg_path = p_dir / "segment_00.parquet"
                break

    df = pd.read_parquet(seg_path)

    if "label" in df.columns:
        for start in range(0, len(df) - window_len, 10):
            win = df.iloc[start : start + window_len]
            if win["label"].mean() >= min_chew_frac:
                x = win[SENSOR_CHANNELS].to_numpy(dtype=np.float32).T  # (C, T)
                print(
                    f"  Chewing window found at row {start}  "
                    f"(chewing fraction = {win['label'].mean():.0%})"
                )
                return x, participant, start

    # No label column or no high-chew window: use first available window
    x = df[SENSOR_CHANNELS].iloc[:window_len].to_numpy(dtype=np.float32).T
    print(
        f"  No high-chew window found in {seg_path.name} — using first window (no-label fallback)"
    )
    return x, participant, 0


# ── Masking ───────────────────────────────────────────────────────────────────


def make_mask(x: np.ndarray, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Return (x_masked, bool_mask) where True = masked position."""
    C, T = x.shape
    g = torch.Generator()
    g.manual_seed(seed)
    mask_t = sample_mask_batch(
        1, C, T, span_mask_ratio=0.4, channel_mask_prob=0.15, generator=g
    )  # (1, C, T) BoolTensor
    x_t = torch.from_numpy(x).unsqueeze(0)  # (1, C, T)
    token = torch.zeros(C)
    x_masked_t = apply_mask_token(x_t, mask_t, token)
    return x_masked_t.squeeze(0).numpy(), mask_t.squeeze(0).numpy().astype(bool)


# ── Transforms (illustrative only -- not used by any pretraining script) ─────


def transform_arrow_of_time(x: np.ndarray) -> np.ndarray:
    return np.flip(x, axis=1).copy()


def transform_permutation(x: np.ndarray, n_chunks: int = 4, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    C, T = x.shape
    chunk = T // n_chunks
    order = rng.permutation(n_chunks)
    return np.concatenate([x[:, i * chunk : (i + 1) * chunk] for i in order], axis=1)


def transform_time_warp(
    x: np.ndarray, seed: int = 3, n_ctrl: int = 6, max_warp: float = 0.2
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    C, T = x.shape
    ctrl = np.linspace(0, T - 1, n_ctrl)
    perturb = rng.uniform(-max_warp * T / n_ctrl, max_warp * T / n_ctrl, n_ctrl)
    perturb[[0, -1]] = 0.0
    ctrl_warped = np.sort(np.clip(ctrl + perturb, 0, T - 1))
    t_query = np.linspace(0, T - 1, T)
    t_src = np.interp(t_query, ctrl_warped, ctrl)
    out = np.empty_like(x)
    for c in range(C):
        out[c] = np.interp(t_src, np.arange(T), x[c])
    return out.astype(np.float32)


# ── Helper: shade masked spans on an axis ────────────────────────────────────


def _shade_mask(
    ax: plt.Axes, mask_row: np.ndarray, t: np.ndarray, color: str = "grey", alpha: float = 0.25
) -> None:
    in_mask, span_start = False, 0.0
    for ti, mv in enumerate(mask_row):
        if mv and not in_mask:
            span_start = t[ti]
            in_mask = True
        elif not mv and in_mask:
            ax.axvspan(span_start, t[ti - 1], alpha=alpha, color=color, linewidth=0)
            in_mask = False
    if in_mask:
        ax.axvspan(span_start, t[-1], alpha=alpha, color=color, linewidth=0)


# ── Figure 1: Transforms demo ─────────────────────────────────────────────────


def plot_transforms(
    x: np.ndarray,
    mask: np.ndarray,
    participant: str,
    out_path: Path,
) -> None:
    """Five-column × 3-row figure: original + four transforms on representative channels."""
    t = np.arange(x.shape[1]) / 50.0  # seconds axis

    col_defs = [
        # (title, signal_array, highlight_mask_row, extra_fn)
        ("Original\n(z-scored per-recording)", x, None, None),
        ("After masking\n(grey = hidden from model)", x, mask, None),
        ("Arrow of Time\n(temporal flip)", transform_arrow_of_time(x), None, None),
        ("Permutation\n(4 chunks shuffled)", transform_permutation(x), None, "permutation"),
        ("Time-warping\n(non-linear time stretch)", transform_time_warp(x), None, None),
    ]

    col_colors = ["#1565C0", "#37474F", "#B71C1C", "#1B5E20", "#4A148C"]

    repr_indices = [SENSOR_CHANNELS.index(c) for c in REPR_CHANNELS]
    n_rows, n_cols = len(REPR_CHANNELS), len(col_defs)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(19, 6.5), sharey="row")
    fig.suptitle(
        f"SSL pretext task transforms  —  SIPMEatingStudy / {participant} "
        f"(chewing window, 4 s @ 50 Hz, per-recording z-scored)",
        fontsize=12,
        y=1.02,
    )

    for col_i, (col_title, x_show, mask_show, extra) in enumerate(col_defs):
        for row_i, (ch_name, ch_idx) in enumerate(zip(REPR_CHANNELS, repr_indices)):
            ax = axes[row_i, col_i]
            signal = x_show[ch_idx]

            if mask_show is not None:
                # Background: original signal in very light color
                ax.plot(t, x[ch_idx], color="#CFD8DC", linewidth=1.0, zorder=1)
                # Shade masked positions
                _shade_mask(ax, mask_show[ch_idx], t, color="dimgrey", alpha=0.35)
                # Visible (unmasked) signal in full color
                visible = signal.copy().astype(float)
                visible[mask_show[ch_idx]] = np.nan
                ax.plot(t, visible, color=col_colors[col_i], linewidth=1.5, zorder=3)
            elif col_i == 0:
                ax.plot(t, signal, color=col_colors[col_i], linewidth=1.5)
            else:
                # Overlay: original faint, transform bold
                ax.plot(t, x[ch_idx], color="#CFD8DC", linewidth=1.0, label="original")
                ax.plot(t, signal, color=col_colors[col_i], linewidth=1.5, label="transformed")

            # Permutation: mark chunk boundaries
            if extra == "permutation":
                chunk = x.shape[1] // 4
                for b_idx in range(1, 4):
                    ax.axvline(
                        b_idx * chunk / 50.0,
                        color="#777777",
                        linestyle="--",
                        linewidth=0.8,
                        alpha=0.9,
                    )

            ax.set_xlim(0, t[-1])
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.18)

            if row_i == 0:
                ax.set_title(col_title, fontsize=8.5, color=col_colors[col_i], pad=5)
            if col_i == 0:
                ax.set_ylabel(CHANNEL_LABELS[ch_name], fontsize=8)
            if row_i == n_rows - 1:
                ax.set_xlabel("Time (s)", fontsize=8)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved  →  {out_path}")


# ── Figure 2: Reconstruction demo ────────────────────────────────────────────


def plot_reconstruction(
    x: np.ndarray,
    mask: np.ndarray,
    recons: dict[str, np.ndarray],
    out_path: Path,
) -> None:
    """All 7 channels × (2 + n_arch) columns: original | masked input | reconstructions."""
    t = np.arange(x.shape[1]) / 50.0
    n_arch = len(recons)
    n_rows = len(SENSOR_CHANNELS)
    n_cols = 2 + n_arch

    arch_names = list(recons.keys())
    col_colors = ["#1565C0", "#37474F"] + [ARCH_COLORS[a] for a in arch_names]
    col_titles = ["Original\n(ground truth)", "Masked input\n(what the model sees)"] + [
        f"{ARCH_LABELS.get(a, a)}\nreconstruction" for a in arch_names
    ]

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 2.0 * n_rows), sharey="row")
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(
        "Masked reconstruction — original vs. model output on a held-out chewing window\n"
        "(bold segments = where the model had to reconstruct; light = unmasked context it could see)",
        fontsize=11,
        y=1.01,
    )

    for row_i, ch_name in enumerate(SENSOR_CHANNELS):
        ch_idx = row_i
        m = mask[ch_idx]  # (T,) bool

        for col_i in range(n_cols):
            ax = axes[row_i, col_i]

            if col_i == 0:
                # Ground truth
                ax.plot(t, x[ch_idx], color=col_colors[0], linewidth=1.5)

            elif col_i == 1:
                # Masked input: show visible signal in color, shade masked regions
                visible = x[ch_idx].astype(float).copy()
                visible[m] = np.nan
                ax.plot(t, x[ch_idx], color="#CFD8DC", linewidth=1.0, alpha=0.7)
                ax.plot(t, visible, color=col_colors[1], linewidth=1.5)
                _shade_mask(ax, m, t, color="dimgrey", alpha=0.25)

            else:
                arch = arch_names[col_i - 2]
                recon = recons[arch]
                # Context (unmasked): original faint underneath
                ax.plot(t, x[ch_idx], color="#CFD8DC", linewidth=1.0, alpha=0.7)
                # Full reconstruction curve
                ax.plot(t, recon[ch_idx], color=col_colors[col_i], linewidth=1.3, alpha=0.6)
                # Highlight the reconstructed-at-masked positions in bold
                recon_masked = recon[ch_idx].astype(float).copy()
                recon_masked[~m] = np.nan
                ax.plot(t, recon_masked, color=col_colors[col_i], linewidth=2.5)
                # Light shading over masked region
                _shade_mask(ax, m, t, color=col_colors[col_i], alpha=0.10)

            ax.set_xlim(0, t[-1])
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.18)

            if row_i == 0:
                ax.set_title(col_titles[col_i], fontsize=9, color=col_colors[col_i], pad=5)
            if col_i == 0:
                ax.set_ylabel(CHANNEL_LABELS[ch_name], fontsize=8)
            if row_i == n_rows - 1:
                ax.set_xlabel("Time (s)", fontsize=8)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved  →  {out_path}")


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize SSL transforms and reconstruction on real sensor data."
    )
    parser.add_argument(
        "--interim-root",
        type=Path,
        default=Path("src/chewing_ssl/data/data/interim"),
        help="Root of per-recording z-scored interim segments",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints/pretrain"),
        help="Directory containing *_best.pt checkpoints from pretrain.py",
    )
    parser.add_argument(
        "--architectures",
        nargs="+",
        default=["cnn", "bigru"],
        choices=list(ARCHITECTURES),
        help="Which architectures to load for reconstruction (skipped if checkpoint missing)",
    )
    parser.add_argument("--variant", default="personal")
    parser.add_argument("--dataset", default="SIPMEatingStudy")
    parser.add_argument(
        "--participant", default="p01", help="Participant to load the example window from"
    )
    parser.add_argument("--out-dir", type=Path, default=Path("checkpoints/plots"))
    parser.add_argument(
        "--mask-seed",
        type=int,
        default=42,
        help="Seed for reproducible masking pattern in visualizations",
    )
    args = parser.parse_args()

    print("\nSSL Visualization")
    print("=" * 60)

    # ── Load one real chewing window ──
    print(f"\n  Loading: {args.dataset} / {args.participant}  (variant={args.variant})")
    x, participant, start_row = find_chewing_window(
        args.interim_root, args.dataset, args.participant, args.variant
    )
    print(
        f"  Signal: {x.shape[0]} channels × {x.shape[1]} timesteps  "
        f"({x.shape[1] / 50:.1f}s @ 50 Hz)"
    )

    # ── Build the mask ──
    x_masked, mask = make_mask(x, seed=args.mask_seed)
    print(f"  Mask: {mask.mean() * 100:.1f}% of positions hidden")

    # ── Figure 1: Transforms demo ──
    print("\n  [1/2] Generating transforms_demo.png ...")
    plot_transforms(x, mask, participant, args.out_dir / "transforms_demo.png")

    # ── Figure 2: Reconstruction demo ──
    print("\n  [2/2] Checking for trained checkpoints ...")
    recons: dict[str, np.ndarray] = {}

    for arch in args.architectures:
        ckpt_path = args.checkpoint_dir / f"{arch}_{args.variant}_best.pt"
        if not ckpt_path.exists():
            print(f"  [{arch}] no checkpoint at {ckpt_path}  (run pretrain.py first)")
            continue

        print(f"  [{arch}] loading {ckpt_path} ...")
        model = ARCHITECTURES[arch](n_channels=len(SENSOR_CHANNELS), **MODEL_KWARGS[arch])
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model_state_dict"])
        model.eval()
        print(
            f"  [{arch}] checkpoint epoch={state.get('epoch', '?')}  "
            f"val_loss={state.get('val_loss', float('nan')):.4f}"
        )

        x_t = torch.from_numpy(x).unsqueeze(0)  # (1, C, T)
        mask_t = torch.from_numpy(mask).unsqueeze(0)  # (1, C, T)
        x_in = apply_mask_token(x_t, mask_t, model.mask_token)
        with torch.no_grad():
            recon = model(x_in).squeeze(0).numpy()  # (C, T)
        recons[arch] = recon

    if recons:
        print("\n  Generating reconstruction_demo.png ...")
        plot_reconstruction(x, mask, recons, args.out_dir / "reconstruction_demo.png")
    else:
        print("\n  No checkpoints found — skipping reconstruction_demo.png")
        print("  Train first, then re-run this script to see reconstruction quality.")

    print(f"\n  Plots saved to  {args.out_dir}/")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
