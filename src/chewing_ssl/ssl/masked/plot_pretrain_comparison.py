"""Plot masked-reconstruction pretraining results across the CNN and BiGRU encoder architectures.

Reads the history.json files written by chewing_ssl.ssl.masked.pretrain and produces:
  1. loss_curves.png    -- train + val loss per epoch, all architectures overlaid
  2. channel_losses.png -- per-channel val MSE at the final epoch (bar chart)
  3. summary table printed to stdout
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ARCH_COLOURS = {"cnn": "#2196F3", "bigru": "#FF9800"}
ARCH_LABELS = {"cnn": "CNN U-Net", "bigru": "Bi-GRU"}
CHANNEL_LABELS = {
    "nav_x_left_temple": "Nav LeftTemple X",
    "nav_y_left_temple": "Nav LeftTemple Y",
    "prox_left_cheek": "Prox Left Cheek",
    "prox_right_cheek": "Prox Right Cheek",
    "accel_x": "Accel X",
    "accel_y": "Accel Y",
    "accel_z": "Accel Z",
}


def load_history(checkpoint_dir: Path, variant: str = "personal") -> dict[str, list[dict]]:
    histories = {}
    for arch in ("cnn", "bigru"):
        path = checkpoint_dir / f"{arch}_{variant}_history.json"
        if path.exists():
            histories[arch] = json.loads(path.read_text())
    return histories


def plot_loss_curves(histories: dict, out_path: Path) -> None:
    fig, (ax_train, ax_val) = plt.subplots(1, 2, figsize=(12, 5))

    for arch, history in histories.items():
        colour = ARCH_COLOURS[arch]
        label = ARCH_LABELS[arch]
        epochs = [r["epoch"] + 1 for r in history]
        train_loss = [r["loss"] for r in history]
        val_loss = [r["val_loss"] for r in history]
        ax_train.plot(epochs, train_loss, color=colour, label=label, linewidth=2)
        ax_val.plot(epochs, val_loss, color=colour, label=label, linewidth=2)

    for ax, title in zip(
        [ax_train, ax_val], ["Training loss (masked MSE)", "Validation loss (masked MSE)"]
    ):
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE (masked positions only, z-scored signal)")
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.3)

    fig.suptitle(
        "Masked reconstruction pretraining — loss curves\n(personal normalization, 200K train windows)"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_channel_losses(histories: dict, out_path: Path) -> None:
    channels = list(CHANNEL_LABELS.keys())
    x = np.arange(len(channels))
    n_arch = len(histories)
    width = 0.25
    offsets = np.linspace(-(n_arch - 1) * width / 2, (n_arch - 1) * width / 2, n_arch)

    fig, ax = plt.subplots(figsize=(11, 5))
    for offset, (arch, history) in zip(offsets, histories.items()):
        final = history[-1]["val_per_channel"]
        vals = [final.get(c, float("nan")) for c in channels]
        ax.bar(
            x + offset, vals, width, label=ARCH_LABELS[arch], color=ARCH_COLOURS[arch], alpha=0.85
        )

    ax.set_xticks(x)
    ax.set_xticklabels([CHANNEL_LABELS[c] for c in channels], rotation=30, ha="right")
    ax.set_ylabel("Val MSE (masked positions only)")
    ax.set_title(
        "Per-channel reconstruction error at final epoch\n(lower = this channel was easier to reconstruct from context)"
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def print_summary(histories: dict) -> None:
    print("\n" + "=" * 70)
    print(f"{'Architecture':<18} {'Best val loss':>15} {'Final val loss':>15} {'Epochs':>8}")
    print("-" * 70)
    for arch, history in sorted(
        histories.items(), key=lambda kv: min(r["val_loss"] for r in kv[1])
    ):
        best = min(r["val_loss"] for r in history)
        final = history[-1]["val_loss"]
        n = len(history)
        print(f"{ARCH_LABELS[arch]:<18} {best:>15.4f} {final:>15.4f} {n:>8}")
    print("=" * 70)

    print("\nPer-channel val MSE at final epoch (↓ easier to reconstruct):")
    all_channels = list(CHANNEL_LABELS.keys())
    print(f"{'Channel':<22} " + " ".join(f"{ARCH_LABELS[a]:>18}" for a in histories))
    print("-" * (22 + 19 * len(histories)))
    for c in all_channels:
        vals = {
            arch: history[-1]["val_per_channel"].get(c, float("nan"))
            for arch, history in histories.items()
        }
        best_arch = min(vals, key=vals.get)
        row = f"{CHANNEL_LABELS[c]:<22} "
        for arch in histories:
            marker = "*" if arch == best_arch else " "
            row += f"{vals[arch]:>17.4f}{marker}"
        print(row)
    print("  * = best (lowest) for that channel across architectures")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot pretraining comparison results.")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/pretrain"))
    parser.add_argument("--out", type=Path, default=Path("checkpoints/pretrain"))
    parser.add_argument("--variant", default="personal")
    args = parser.parse_args()

    histories = load_history(args.checkpoint_dir, args.variant)
    if not histories:
        raise SystemExit(
            f"No history.json files found under {args.checkpoint_dir}. Run pretrain.py first."
        )

    print(f"Loaded histories for: {list(histories)}")
    plot_loss_curves(histories, args.out / "loss_curves.png")
    plot_channel_losses(histories, args.out / "channel_losses.png")
    print_summary(histories)


if __name__ == "__main__":
    main()
