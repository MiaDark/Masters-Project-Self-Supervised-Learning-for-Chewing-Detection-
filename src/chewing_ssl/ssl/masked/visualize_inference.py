"""Visualize every step of chewing detection inference.

Produces three PNG files in --out-dir:

  inference_pipeline.png
      One test window, step-by-step through the model:
        Step 1 — Input: 7-channel z-scored sensor signal (heatmap)
        Step 2 — GRU encoding: hidden-state activations at every timestep (heatmap)
        Step 3 — Chewing probability: sigmoid of per-timestep logits (line plot)
        Step 4 — Decision: binary prediction vs ground truth (color bar)

  inference_recording.png
      ~30 seconds of continuous inference on a test participant, showing
      the model's rolling predictions against the actual chewing labels.

  inference_metrics.png
      Confusion matrix, ROC curve, and Precision-Recall curve evaluated
      on the full finetune_test set.

Run from the project root after running finetune.py:
  python -m chewing_ssl.ssl.masked.visualize_inference
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
from matplotlib import gridspec
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

from chewing_ssl.common.data.preprocess import SENSOR_CHANNELS
from chewing_ssl.common.data.windowing import WINDOW_LEN, WindowDataset
from chewing_ssl.common.evaluation.postprocess import postprocess
from chewing_ssl.ssl.masked.finetune import ChewingDetector, load_index

CHANNEL_LABELS = {
    "nav_x_left_temple": "Nav X (L Temple)",
    "nav_y_left_temple": "Nav Y (L Temple)",
    "prox_left_cheek": "Prox Left Cheek",
    "prox_right_cheek": "Prox Right Cheek",
    "accel_x": "Accel X",
    "accel_y": "Accel Y",
    "accel_z": "Accel Z",
}

PALETTE = {
    "signal": "#1565C0",
    "chewing": "#E65100",
    "no_chewing": "#B0BEC5",
    "correct": "#2E7D32",
    "error": "#C62828",
    "prob": "#6A1B9A",
    "threshold": "#FF6F00",
}


def load_finetuned_model(checkpoint_path: Path) -> ChewingDetector:
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = ChewingDetector()
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    print(
        f"  Loaded checkpoint: phase={state.get('phase')}  "
        f"epoch={state.get('epoch', 0) + 1}  val_f1={state.get('val_f1', float('nan')):.3f}"
    )
    return model


# ── Inference helpers ─────────────────────────────────────────────────────────


@torch.no_grad()
def infer_window(model: ChewingDetector, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Run one (C, T) window through the model.  Returns (probs, hidden_states)."""
    x_t = torch.from_numpy(x).unsqueeze(0)  # (1, C, T)
    logits = model(x_t).squeeze(0)  # (T,)
    h = model.hidden_states(x_t).squeeze(0)  # (T, 2*hidden)
    return torch.sigmoid(logits).numpy(), h.numpy()


@torch.no_grad()
def infer_dataset(
    model: ChewingDetector, loader: DataLoader
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run full dataset through model.  Returns (probs, preds, labels) all (N*T,)."""
    all_probs, all_preds, all_labels = [], [], []
    for x, y in loader:
        logits = model(x)  # (B, T)
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()
        all_probs.append(probs.cpu().numpy().ravel())
        all_preds.append(preds.cpu().numpy().ravel())
        all_labels.append(y.cpu().numpy().ravel())
    return (
        np.concatenate(all_probs),
        np.concatenate(all_preds).astype(int),
        np.concatenate(all_labels).astype(int),
    )


# ── Figure helpers ────────────────────────────────────────────────────────────


def _label_bar(ax: plt.Axes, labels: np.ndarray, t: np.ndarray, title: str, colors: tuple) -> None:
    """Draw a compact binary label bar (filled spans) on ax."""
    in_region, start = False, 0.0
    for i, lv in enumerate(labels):
        if lv and not in_region:
            start = t[i]
            in_region = True
        elif not lv and in_region:
            ax.axvspan(start, t[i - 1], color=colors[1], alpha=0.85, linewidth=0)
            in_region = False
    if in_region:
        ax.axvspan(start, t[-1], color=colors[1], alpha=0.85, linewidth=0)
    ax.set_yticks([])
    ax.set_xlim(t[0], t[-1])
    ax.set_facecolor(colors[0])
    ax.set_ylabel(title, fontsize=8, labelpad=4)
    ax.tick_params(labelsize=7)


def _shade_chewing(ax: plt.Axes, labels: np.ndarray, t: np.ndarray, alpha: float = 0.15) -> None:
    """Shade ground-truth chewing regions on ax."""
    in_chew, start = False, 0.0
    for i, lv in enumerate(labels):
        if lv and not in_chew:
            start = t[i]
            in_chew = True
        elif not lv and in_chew:
            ax.axvspan(start, t[i - 1], color=PALETTE["chewing"], alpha=alpha, linewidth=0)
            in_chew = False
    if in_chew:
        ax.axvspan(start, t[-1], color=PALETTE["chewing"], alpha=alpha, linewidth=0)


# ── Figure 1: Inference pipeline ──────────────────────────────────────────────


def plot_inference_pipeline(
    x: np.ndarray,
    y: np.ndarray,
    probs: np.ndarray,
    hidden: np.ndarray,
    participant: str,
    out_path: Path,
) -> None:
    """4-row figure: input heatmap | GRU activations | probability | decision."""
    T = x.shape[1]
    t = np.arange(T) / 50.0
    preds = (probs >= 0.5).astype(int)

    fig = plt.figure(figsize=(14, 11))
    gs = gridspec.GridSpec(
        5,
        1,
        height_ratios=[3, 3, 1.5, 1, 1],
        hspace=0.45,
        left=0.10,
        right=0.97,
        top=0.93,
        bottom=0.06,
    )
    ax_sig = fig.add_subplot(gs[0])
    ax_gru = fig.add_subplot(gs[1])
    ax_prob = fig.add_subplot(gs[2])
    ax_gt = fig.add_subplot(gs[3])
    ax_pred = fig.add_subplot(gs[4])

    fig.suptitle(
        f"Inference pipeline — SIPMEatingStudy / {participant}  (4 s @ 50 Hz, per-recording z-scored)\n"
        f"Fine-tuned BiGRU (pretrained with Arrow of Time)",
        fontsize=11,
    )

    # ── Step 1: Input signal ──
    im1 = ax_sig.imshow(
        x,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-3,
        vmax=3,
        extent=[0, T / 50.0, -0.5, len(SENSOR_CHANNELS) - 0.5],
        origin="lower",
    )
    ax_sig.set_yticks(range(len(SENSOR_CHANNELS)))
    ax_sig.set_yticklabels([CHANNEL_LABELS[c] for c in SENSOR_CHANNELS], fontsize=7)
    ax_sig.set_xlabel("Time (s)", fontsize=8)
    ax_sig.set_title(
        "Step 1 — Input: 7-channel sensor signal  (colour = z-score)", fontsize=9, loc="left"
    )
    plt.colorbar(im1, ax=ax_sig, fraction=0.02, pad=0.01, label="z-score")
    _shade_chewing(ax_sig, y, t)

    # ── Step 2: GRU hidden states ──
    # Show absolute activation magnitude of all 128 units, transposed so time is x-axis
    im2 = ax_gru.imshow(
        np.abs(hidden.T),  # (128, T)
        aspect="auto",
        cmap="viridis",
        extent=[0, T / 50.0, -0.5, hidden.shape[1] - 0.5],
        origin="lower",
    )
    ax_gru.set_yticks([0, 31, 63, 95, 127])
    ax_gru.set_yticklabels(["unit 0", "31", "63", "95", "127"], fontsize=7)
    ax_gru.set_xlabel("Time (s)", fontsize=8)
    ax_gru.set_title(
        "Step 2 — GRU encoding: |hidden state| per unit at each timestep  "
        "(bright = strongly activated)",
        fontsize=9,
        loc="left",
    )
    plt.colorbar(im2, ax=ax_gru, fraction=0.02, pad=0.01, label="|activation|")
    _shade_chewing(ax_gru, y, t)

    # ── Step 3: Chewing probability ──
    ax_prob.plot(t, probs, color=PALETTE["prob"], linewidth=1.8, label="P(chewing)")
    ax_prob.axhline(
        0.5, color=PALETTE["threshold"], linestyle="--", linewidth=1.2, label="threshold 0.5"
    )
    ax_prob.fill_between(
        t,
        probs,
        0.5,
        where=(probs >= 0.5),
        alpha=0.25,
        color=PALETTE["chewing"],
        label="predicted chewing",
    )
    ax_prob.set_ylim(-0.05, 1.05)
    ax_prob.set_ylabel("P(chewing)", fontsize=8)
    ax_prob.set_xlabel("Time (s)", fontsize=8)
    ax_prob.legend(fontsize=7, loc="upper right", framealpha=0.8)
    ax_prob.set_title(
        "Step 3 — Head output: per-timestep chewing probability  (sigmoid of logit)",
        fontsize=9,
        loc="left",
    )
    ax_prob.grid(alpha=0.2)
    _shade_chewing(ax_prob, y, t)

    # ── Step 4a: Ground truth ──
    _label_bar(ax_gt, y, t, "Ground\ntruth", colors=(PALETTE["no_chewing"], PALETTE["chewing"]))
    ax_gt.set_title("Step 4 — Decision vs ground truth", fontsize=9, loc="left")

    # ── Step 4b: Prediction (colour by correct/wrong) ──
    correct = (preds == y).astype(int)
    _label_bar(
        ax_pred,
        preds,
        t,
        "Prediction\n(green=✓)",
        colors=(PALETTE["no_chewing"], PALETTE["chewing"]),
    )
    # Overlay errors in red
    in_err, err_start = False, 0.0
    for i, cv in enumerate(correct):
        if not cv and not in_err:
            err_start = t[i]
            in_err = True
        elif cv and in_err:
            ax_pred.axvspan(err_start, t[i - 1], color=PALETTE["error"], alpha=0.7, linewidth=0)
            in_err = False
    if in_err:
        ax_pred.axvspan(err_start, t[-1], color=PALETTE["error"], alpha=0.7, linewidth=0)
    ax_pred.set_xlabel("Time (s)", fontsize=8)

    # Sync x-axis limits
    for ax in [ax_sig, ax_gru, ax_prob, ax_gt, ax_pred]:
        ax.set_xlim(0, T / 50.0)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved  →  {out_path}")


# ── Figure 2: Full recording ───────────────────────────────────────────────────


def plot_inference_recording(
    model: ChewingDetector,
    interim_root: Path,
    dataset: str,
    participant: str,
    variant: str,
    out_path: Path,
    n_seconds: float = 900.0,
) -> None:
    """Plot continuous model predictions over ~n_seconds of a test recording."""
    seg_path = interim_root / variant / dataset / participant / "segment_00.parquet"
    if not seg_path.exists():
        print(f"  Segment not found: {seg_path}  (skipping recording figure)")
        return

    df = pd.read_parquet(seg_path)
    n_samples = min(len(df), int(n_seconds * 50))
    df = df.iloc[:n_samples]

    # Slide non-overlapping WINDOW_LEN windows; concatenate predictions
    x_all = df[SENSOR_CHANNELS].to_numpy(dtype=np.float32).T  # (C, N)
    y_all = df["label"].to_numpy(dtype=np.float32) if "label" in df.columns else None
    t_all = np.arange(n_samples) / 50.0

    probs_all = np.full(n_samples, np.nan)
    with torch.no_grad():
        for start in range(0, n_samples - WINDOW_LEN + 1, WINDOW_LEN):
            xw = torch.from_numpy(x_all[:, start : start + WINDOW_LEN]).unsqueeze(0)
            lgt = model(xw).squeeze(0)
            probs_all[start : start + WINDOW_LEN] = torch.sigmoid(lgt).numpy()

    preds_all = (probs_all >= 0.5).astype(float)
    preds_all[np.isnan(probs_all)] = np.nan

    # Post-processed predictions: hysteresis → gap fill → min duration
    preds_post = postprocess(probs_all).astype(float)

    repr_ch = ["nav_x_left_temple", "prox_left_cheek", "accel_z"]
    repr_idx = [SENSOR_CHANNELS.index(c) for c in repr_ch]

    fig, axes = plt.subplots(
        7,
        1,
        figsize=(16, 11),
        gridspec_kw={"height_ratios": [2, 2, 2, 1.5, 0.8, 0.8, 0.8]},
        sharex=True,
    )
    fig.suptitle(
        f"Continuous inference — SIPMEatingStudy / {participant}  "
        f"(first {n_seconds:.0f}s, non-overlapping windows)",
        fontsize=11,
    )

    colors_sig = [PALETTE["signal"], PALETTE["chewing"], "#2E7D32"]
    for i, (ch_name, ch_idx) in enumerate(zip(repr_ch, repr_idx)):
        ax = axes[i]
        ax.plot(t_all, x_all[ch_idx], color=colors_sig[i], linewidth=0.9)
        ax.set_ylabel(CHANNEL_LABELS[ch_name], fontsize=8)
        ax.grid(alpha=0.2)
        if y_all is not None:
            _shade_chewing(ax, y_all, t_all)

    # Probability — add hysteresis bands so the reader can see how the two
    # thresholds carve out bouts
    ax_p = axes[3]
    ax_p.plot(t_all, probs_all, color=PALETTE["prob"], linewidth=1.5)
    ax_p.axhline(
        0.65, color=PALETTE["threshold"], linestyle=":", linewidth=1.0, label="high (0.65)"
    )
    ax_p.axhline(0.35, color=PALETTE["correct"], linestyle=":", linewidth=1.0, label="low  (0.35)")
    ax_p.axhline(0.50, color="grey", linestyle="--", linewidth=0.8, label="raw threshold (0.5)")
    ax_p.fill_between(
        t_all, probs_all, 0.5, where=(probs_all >= 0.5), alpha=0.2, color=PALETTE["chewing"]
    )
    ax_p.set_ylim(-0.05, 1.05)
    ax_p.set_ylabel("P(chewing)", fontsize=8)
    ax_p.legend(fontsize=7, loc="upper right", framealpha=0.7, ncol=3)
    ax_p.grid(alpha=0.2)
    ax_p.set_title("Predicted chewing probability", fontsize=8, loc="left")

    # Raw predicted bouts (simple >= 0.5)
    ax_raw = axes[4]
    raw_binary = np.nan_to_num(preds_all, nan=0.0)
    _label_bar(
        ax_raw, raw_binary, t_all, "Raw\n(≥0.5)", colors=(PALETTE["no_chewing"], PALETTE["chewing"])
    )

    # Post-processed bouts (hysteresis + gap fill + min duration)
    ax_post = axes[5]
    _label_bar(
        ax_post, preds_post, t_all, "Post-\nprocessed", colors=(PALETTE["no_chewing"], "#1B5E20")
    )

    # Ground truth bar
    ax_gt = axes[6]
    if y_all is not None:
        _label_bar(
            ax_gt, y_all, t_all, "Ground\ntruth", colors=(PALETTE["no_chewing"], PALETTE["chewing"])
        )
    ax_gt.set_xlabel("Time (s)", fontsize=9)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved  →  {out_path}")


# ── Figure 3: Metrics ─────────────────────────────────────────────────────────


def plot_metrics(
    probs: np.ndarray,
    preds: np.ndarray,
    labels: np.ndarray,
    out_path: Path,
) -> None:
    """Confusion matrix + ROC curve + Precision-Recall curve."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Fine-tuned BiGRU — evaluation on held-out test participants", fontsize=11)

    f1 = f1_score(labels, preds, zero_division=0.0)
    auc = roc_auc_score(labels, probs)

    # ── Confusion matrix ──
    ax = axes[0]
    cm = confusion_matrix(labels, preds)
    total = cm.sum()
    cm_norm = cm / total

    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    for r in range(2):
        for c in range(2):
            color = "white" if cm_norm[r, c] > 0.5 else "black"
            ax.text(
                c,
                r,
                f"{cm[r, c]:,}\n({cm_norm[r, c]:.1%})",
                ha="center",
                va="center",
                fontsize=10,
                color=color,
            )
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred: no chew", "Pred: chewing"], fontsize=9)
    ax.set_yticklabels(["True: no chew", "True: chewing"], fontsize=9)
    ax.set_title(f"Confusion Matrix\n(F1={f1:.3f})", fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.04, label="fraction of all timesteps")

    # ── ROC curve ──
    ax = axes[1]
    fpr, tpr, _ = roc_curve(labels, probs)
    ax.plot(fpr, tpr, color=PALETTE["signal"], linewidth=2, label=f"BiGRU AoT (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="random (AUC=0.5)")
    ax.fill_between(fpr, tpr, alpha=0.12, color=PALETTE["signal"])
    ax.set_xlabel("False Positive Rate", fontsize=9)
    ax.set_ylabel("True Positive Rate", fontsize=9)
    ax.set_title("ROC Curve", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    # ── Precision-Recall curve ──
    ax = axes[2]
    prec, rec, thresholds = precision_recall_curve(labels, probs)
    ax.plot(rec, prec, color=PALETTE["prob"], linewidth=2)
    ax.fill_between(rec, prec, alpha=0.12, color=PALETTE["prob"])

    # Mark the 0.5 threshold operating point
    idx = np.searchsorted(thresholds, 0.5)
    if 0 <= idx < len(prec) - 1:
        ax.scatter(
            [rec[idx]],
            [prec[idx]],
            color=PALETTE["threshold"],
            s=80,
            zorder=5,
            label=f"threshold=0.5\n(P={prec[idx]:.2f}, R={rec[idx]:.2f})",
        )
        ax.legend(fontsize=8)

    chew_frac = labels.mean()
    ax.axhline(
        chew_frac,
        color="grey",
        linestyle="--",
        linewidth=0.8,
        label=f"no-skill baseline ({chew_frac:.1%})",
    )
    ax.set_xlabel("Recall", fontsize=9)
    ax.set_ylabel("Precision", fontsize=9)
    ax.set_title("Precision-Recall Curve", fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.2)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved  →  {out_path}")


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize every step of chewing detection inference."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/finetune/arrow_of_time_bigru_personal_finetune_best.pt"),
        help="Fine-tuned ChewingDetector checkpoint from finetune.py",
    )
    parser.add_argument(
        "--interim-root", type=Path, default=Path("src/chewing_ssl/data/data/interim")
    )
    parser.add_argument(
        "--processed-root", type=Path, default=Path("src/chewing_ssl/data/data/processed")
    )
    parser.add_argument("--dataset", default="SIPMEatingStudy")
    parser.add_argument("--variant", default="personal")
    parser.add_argument(
        "--example-participant",
        default="all",
        help="Participant ID (e.g. p01) or 'all' to iterate every test participant",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("checkpoints/plots/masked"))
    args = parser.parse_args()

    if not args.checkpoint.exists():
        print(f"\nCheckpoint not found: {args.checkpoint}")
        print("Run finetune.py first:\n")
        print("  python -m chewing_ssl.ssl.masked.finetune")
        return

    print("\nInference Visualization")
    print("=" * 60)
    print(f"  Checkpoint : {args.checkpoint}")

    # ── Load model ──
    model = load_finetuned_model(args.checkpoint)
    model.eval()

    # ── Resolve participant list ──
    test_index = load_index(
        args.processed_root, args.variant, "finetune_test", None, 42, args.dataset
    )
    all_participants = sorted(
        test_index["segment_path"].str.extract(f"/{args.dataset}/([^/]+)/")[0].dropna().unique()
    )
    if args.example_participant == "all":
        participants = all_participants
    else:
        participants = [args.example_participant]

    print(f"  Participants: {', '.join(participants)}\n")

    # ── Per-participant pipeline and recording figures ──
    for participant in participants:
        print(f"\n  ── {participant} ──")
        seg_path = (
            args.interim_root / args.variant / args.dataset / participant / "segment_00.parquet"
        )

        # Pipeline figure
        print(f"  [pipeline] {participant} ...")
        df = pd.read_parquet(seg_path)
        has_label = "label" in df.columns
        chosen_start = 0
        if has_label:
            for start in range(0, len(df) - WINDOW_LEN, 10):
                win_labels = df["label"].iloc[start : start + WINDOW_LEN].to_numpy()
                frac = win_labels.mean()
                if 0.25 <= frac <= 0.75:
                    chosen_start = start
                    break
            if chosen_start == 0:
                for start in range(0, len(df) - WINDOW_LEN, 10):
                    if df["label"].iloc[start : start + WINDOW_LEN].mean() > 0.5:
                        chosen_start = start
                        break

        win_df = df.iloc[chosen_start : chosen_start + WINDOW_LEN]
        x_win = win_df[SENSOR_CHANNELS].to_numpy(dtype=np.float32).T
        y_win = win_df["label"].to_numpy(dtype=np.float32) if has_label else np.zeros(WINDOW_LEN)

        probs_win, hidden_win = infer_window(model, x_win)
        pct_chew = y_win.mean()
        pred_f1 = f1_score(y_win.astype(int), (probs_win >= 0.5).astype(int), zero_division=0.0)
        print(f"    Row {chosen_start}  chewing={pct_chew:.0%}  window-F1={pred_f1:.3f}")

        pipeline_name = (
            f"inference_pipeline_{participant}.png"
            if len(participants) > 1
            else "inference_pipeline.png"
        )
        plot_inference_pipeline(
            x_win,
            y_win,
            probs_win,
            hidden_win,
            participant=participant,
            out_path=args.out_dir / pipeline_name,
        )

        # Recording figure
        print(f"  [recording] {participant} ...")
        recording_name = (
            f"inference_recording_{participant}.png"
            if len(participants) > 1
            else "inference_recording.png"
        )
        plot_inference_recording(
            model,
            args.interim_root,
            args.dataset,
            participant,
            args.variant,
            out_path=args.out_dir / recording_name,
        )

    # ── Metrics figure (full test set) ──
    print("\n  [3/3] Running full test set evaluation ...")
    test_index = load_index(
        args.processed_root, args.variant, "finetune_test", None, 42, args.dataset
    )
    test_loader = DataLoader(
        WindowDataset(
            test_index.sort_values("segment_path").reset_index(drop=True), with_label=True
        ),
        batch_size=256,
        shuffle=False,
        num_workers=0,
    )
    probs_all, preds_all, labels_all = infer_dataset(model, test_loader)
    f1 = f1_score(labels_all, preds_all, zero_division=0.0)
    auc = roc_auc_score(labels_all, probs_all)
    print(
        f"  Test set: F1={f1:.3f}  AUC={auc:.3f}  "
        f"(N={len(labels_all):,} timesteps, {labels_all.mean():.1%} chewing)"
    )

    plot_metrics(
        probs_all,
        preds_all,
        labels_all,
        out_path=args.out_dir / "inference_metrics.png",
    )

    print(f"\n  Plots saved to  {args.out_dir}/")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
