"""Evaluate one or more finetune_window.py checkpoints on the held-out
finetune_test split, and plot the results. The window-level analogue of
visualize_inference.py (which is per-timestep and BiGRU-specific).

Produces three PNG files in --out-dir:

  window_metrics_comparison.png
      Confusion matrix per checkpoint, plus a shared ROC curve and a shared
      Precision-Recall curve with every checkpoint overlaid -- the direct
      TF-C vs TF-FC comparison.

  embedding_space.png
      2D PCA of each checkpoint's pooled test-set embeddings (encoder.embed
      output, before the classification head), colored by true label. Shows
      whether the pretrained representation actually separates chewing from
      non-chewing before you even look at head accuracy.

  window_predictions_recording.png
      One test participant's windows in chronological order: ground truth vs.
      each checkpoint's prediction, so you can see where each approach agrees
      or disagrees with reality over a real stretch of time (the per-window
      analogue of visualize_inference.py's inference_recording.png).

finetune_window.py's checkpoint does not store which architecture built it
(only the filename does, by its own naming convention: <pretext_task>_best.pt)
-- so --checkpoints' pretext_task is inferred from each file's stem. Pass
--time-architecture / --resnet-base-width to override the (cnn / 32)
defaults if you fine-tuned with something else.

Run from the project root, after fine-tuning at least one checkpoint:

  python -m chewing_ssl.ssl.tfc.visualize_window_inference \\
      --checkpoints checkpoints/finetune_window_tfc/tfc_best.pt \\
                    checkpoints/finetune_window_tffc/tffc_best.pt
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
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

from chewing_ssl.common.data.preprocess import SENSOR_CHANNELS
from chewing_ssl.common.data.windowing import WindowDataset
from chewing_ssl.ssl.models import ARCHITECTURES
from chewing_ssl.ssl.models.frequency_encoder import FrequencyEncoder
from chewing_ssl.ssl.models.resnet1d import ResNet1D
from chewing_ssl.ssl.tfc.finetune_window import ContrastivePretrainedEncoder, WindowClassifier, window_labels
from chewing_ssl.ssl.masked.pretrain import MODEL_KWARGS, load_index

PALETTE = {"non_chewing": "#90A4AE", "chewing": "#E65100"}
CKPT_COLORS = ["#1565C0", "#C62828", "#2E7D32", "#6A1B9A"]

def infer_pretext_task(checkpoint_path: Path) -> str:
    stem = checkpoint_path.stem
    if "tffc" in stem:
        return "tffc"
    if "tfc" in stem:
        return "tfc"
    raise ValueError(
        f"Can't infer pretext_task from {checkpoint_path.name!r} -- expected a name containing "
        "'tfc' or 'tffc' (finetune_window.py's own naming convention)."
    )


def build_encoder(pretext_task: str, n_channels: int, time_architecture: str, resnet_base_width: int) -> ContrastivePretrainedEncoder:
    """A freshly-initialized encoder with the right shapes for pretext_task,
    ready to have a real fine-tuned state_dict loaded onto it -- reuses
    ContrastivePretrainedEncoder's real construction logic by feeding it a
    synthetic "checkpoint" dict built from freshly-constructed sub-modules,
    rather than duplicating that logic here."""
    if pretext_task == "tfc":
        time_enc = ARCHITECTURES[time_architecture](n_channels=n_channels, **MODEL_KWARGS[time_architecture])
        freq_enc = FrequencyEncoder(n_channels=n_channels)
        fake_checkpoint = {
            "pretext_task": "tfc",
            "time_architecture": time_architecture,
            "time_encoder_state_dict": time_enc.state_dict(),
            "freq_encoder_state_dict": freq_enc.state_dict(),
            "time_embedding_dim": time_enc.embedding_dim,
            "freq_embedding_dim": freq_enc.embedding_dim,
        }
    elif pretext_task == "tffc":
        enc = ResNet1D(n_channels=n_channels, base_width=resnet_base_width)
        fake_checkpoint = {
            "pretext_task": "tffc",
            "resnet_base_width": resnet_base_width,
            "encoder_state_dict": enc.state_dict(),
            "embedding_dim": enc.embedding_dim,
        }
    else:
        raise ValueError(f"Unknown pretext_task {pretext_task!r}")
    return ContrastivePretrainedEncoder(fake_checkpoint, n_channels)


def load_window_classifier(
    checkpoint_path: Path, n_channels: int, time_architecture: str, resnet_base_width: int,
    head_hidden_dim: int, head_dropout: float,
) -> tuple[WindowClassifier, str]:
    pretext_task = infer_pretext_task(checkpoint_path)
    encoder = build_encoder(pretext_task, n_channels, time_architecture, resnet_base_width)
    model = WindowClassifier(encoder, head_hidden_dim, head_dropout)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    print(f"  [{pretext_task}] loaded {checkpoint_path}  (val_f1={state.get('val_f1', float('nan')):.3f})")
    return model, pretext_task


# ── Test-set inference ───────────────────────────────────────────────────────


def metrics_from_probs(probs: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> dict:
    preds = (probs >= threshold).astype(int)
    n_unique = len(np.unique(labels))
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0.0)),
        "recall": float(recall_score(labels, preds, zero_division=0.0)),
        "f1": float(f1_score(labels, preds, zero_division=0.0)),
        "auc": float(roc_auc_score(labels, probs)) if n_unique > 1 else float("nan"),
    }


@torch.no_grad()
def infer_test_set(model: WindowClassifier, loader: DataLoader) -> dict:
    """Run the full test set through model. Returns probs/preds/labels/embeddings, all (N, ...)."""
    all_probs, all_labels, all_embeddings = [], [], []
    for x, y in loader:
        y = window_labels(y)
        embed = model.encoder.embed(x)
        logits = model.head(embed)
        all_probs.append(torch.sigmoid(logits).numpy())
        all_labels.append(y.numpy())
        all_embeddings.append(embed.numpy())
    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels).astype(int)
    embeddings = np.concatenate(all_embeddings)
    return {"probs": probs, "preds": (probs >= 0.5).astype(int), "labels": labels, "embeddings": embeddings}


# ── Figure 1: metrics comparison ─────────────────────────────────────────────


def plot_metrics_comparison(results: dict[str, dict], out_path: Path, threshold: float = 0.5) -> None:
    names = list(results)
    n = len(names)
    fig, axes = plt.subplots(1, n + 2, figsize=(4.2 * (n + 2), 4.5))
    fig.suptitle(f"Window-level evaluation on finetune_test  (decision threshold={threshold:.2f})", fontsize=12)

    for i, name in enumerate(names):
        r = results[name]
        ax = axes[i]
        preds = (r["probs"] >= threshold).astype(int)
        cm = confusion_matrix(r["labels"], preds)
        cm_norm = cm / cm.sum()
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        for row in range(2):
            for col in range(2):
                color = "white" if cm_norm[row, col] > 0.5 else "black"
                ax.text(col, row, f"{cm[row, col]:,}\n({cm_norm[row, col]:.1%})", ha="center", va="center", fontsize=9, color=color)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred: no chew", "Pred: chewing"], fontsize=8)
        ax.set_yticklabels(["True: no chew", "True: chewing"], fontsize=8)
        f1 = metrics_from_probs(r["probs"], r["labels"], threshold)["f1"]
        ax.set_title(f"{name}\nConfusion Matrix (F1={f1:.3f})", fontsize=9)

    ax_roc = axes[n]
    for i, name in enumerate(names):
        r = results[name]
        fpr, tpr, _ = roc_curve(r["labels"], r["probs"])
        auc = roc_auc_score(r["labels"], r["probs"])
        ax_roc.plot(fpr, tpr, color=CKPT_COLORS[i % len(CKPT_COLORS)], linewidth=2, label=f"{name} (AUC={auc:.3f})")
    ax_roc.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="random")
    ax_roc.set_xlabel("False Positive Rate", fontsize=9)
    ax_roc.set_ylabel("True Positive Rate", fontsize=9)
    ax_roc.set_title("ROC Curve", fontsize=10)
    ax_roc.legend(fontsize=8)
    ax_roc.grid(alpha=0.2)

    ax_pr = axes[n + 1]
    for i, name in enumerate(names):
        r = results[name]
        prec, rec, _ = precision_recall_curve(r["labels"], r["probs"])
        ax_pr.plot(rec, prec, color=CKPT_COLORS[i % len(CKPT_COLORS)], linewidth=2, label=name)
    chew_frac = list(results.values())[0]["labels"].mean()
    ax_pr.axhline(chew_frac, color="grey", linestyle="--", linewidth=0.8, label=f"no-skill ({chew_frac:.1%})")
    ax_pr.set_xlabel("Recall", fontsize=9)
    ax_pr.set_ylabel("Precision", fontsize=9)
    ax_pr.set_title("Precision-Recall Curve", fontsize=10)
    ax_pr.set_xlim(0, 1); ax_pr.set_ylim(0, 1)
    ax_pr.legend(fontsize=8)
    ax_pr.grid(alpha=0.2)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved  →  {out_path}")


# ── Figure 2: embedding space ────────────────────────────────────────────────


def plot_embedding_space(results: dict[str, dict], out_path: Path, max_points: int = 2000, seed: int = 42) -> None:
    names = list(results)
    fig, axes = plt.subplots(1, len(names), figsize=(6 * len(names), 5.5))
    axes = np.atleast_1d(axes)
    fig.suptitle("Pretrained embedding space (2D PCA of pooled test-set embeddings)", fontsize=12)

    rng = np.random.default_rng(seed)
    for ax, name in zip(axes, names):
        r = results[name]
        n = len(r["labels"])
        idx = rng.choice(n, size=min(max_points, n), replace=False)
        emb2d = PCA(n_components=2, random_state=seed).fit_transform(r["embeddings"][idx])
        labels = r["labels"][idx]
        for lab, color, lab_name in [(0, PALETTE["non_chewing"], "non-chewing"), (1, PALETTE["chewing"], "chewing")]:
            mask = labels == lab
            ax.scatter(emb2d[mask, 0], emb2d[mask, 1], color=color, alpha=0.5, s=12, label=lab_name)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("PC 1", fontsize=8)
        ax.set_ylabel("PC 2", fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved  →  {out_path}")


# ── Figure 3: one recording's windows, chronologically ───────────────────────


def _label_bar(ax: plt.Axes, values: np.ndarray, title: str) -> None:
    ax.imshow(values[np.newaxis, :], aspect="auto", cmap="Oranges", vmin=0, vmax=1)
    ax.set_yticks([])
    ax.set_ylabel(title, fontsize=8, rotation=0, ha="right", va="center", labelpad=6)


def _plot_one_recording(
    models: dict[str, WindowClassifier], part_index: pd.DataFrame,
    dataset: str, participant: str, tag: str, out_path: Path, threshold: float = 0.5,
) -> None:
    loader = DataLoader(WindowDataset(part_index, with_label=True), batch_size=64, shuffle=False)
    labels_all = []
    preds_by_model: dict[str, list[np.ndarray]] = {name: [] for name in models}
    with torch.no_grad():
        for x, y in loader:
            labels_all.append(window_labels(y).numpy())
            for name, model in models.items():
                probs = torch.sigmoid(model(x)).numpy()
                preds_by_model[name].append(probs)
    labels_all = np.concatenate(labels_all)

    # 2 rows per model: raw probability (gradient, shows confidence) and the
    # thresholded 0/1 decision (binary, directly comparable to ground truth).
    n_rows = 1 + 2 * len(models)
    fig, axes = plt.subplots(n_rows, 1, figsize=(14, 1.0 * n_rows + 1), sharex=True)
    fig.suptitle(
        f"Window-level predictions over test recording -- {dataset} / {participant} / {tag}\n"
        f"({len(labels_all)} windows, chronological order; decision threshold={threshold:.2f})",
        fontsize=11,
    )

    _label_bar(axes[0], labels_all, "Ground\ntruth")
    row = 1
    for name, prob_chunks in preds_by_model.items():
        probs = np.concatenate(prob_chunks)
        _label_bar(axes[row], probs, f"{name}\nP(chew)\n(raw)")
        _label_bar(axes[row + 1], (probs >= threshold).astype(float), f"{name}\n≥{threshold:.2f}\n(decision)")
        row += 2
    axes[-1].set_xlabel("Window index (chronological)", fontsize=9)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved  →  {out_path}")


def plot_recording_predictions(
    models: dict[str, WindowClassifier], test_index: pd.DataFrame, out_dir: Path,
    dataset: str | None, participant: str | None, threshold: float = 0.5,
) -> None:
    """Plots one or every genuinely continuous test recording -- (dataset,
    participant, tag) is the key that guarantees "continuous", not
    participant alone. Participant IDs repeat across datasets (different
    physical people, see docs/architecture.md), and even within one
    (dataset, participant), Spectacle_NonMeal participants can span multiple
    non-contiguous recording sessions (different tags, different days) --
    filtering on anything looser would silently concatenate unrelated
    recordings into a fake "chronological" strip.

    participant="all" -> one PNG per distinct (dataset, participant, tag) in
    test_index (mirrors visualize_inference.py's --example-participant all).
    """
    if participant == "all":
        recordings = sorted(test_index.groupby(["dataset", "participant", "tag"]).size().index)
        print(f"  {len(recordings)} distinct test recording(s) -- one PNG each")
        for ds, part, tag in recordings:
            part_index = test_index[
                (test_index["dataset"] == ds) & (test_index["participant"] == part) & (test_index["tag"] == tag)
            ].sort_values("start").reset_index(drop=True)
            out_path = out_dir / f"window_predictions_recording_{ds}_{part}_{tag}.png"
            _plot_one_recording(models, part_index, ds, part, tag, out_path, threshold)
        return

    if participant is not None and dataset is None:
        raise ValueError("--participant requires --recording-dataset too (participant IDs repeat across datasets).")

    if dataset is not None and participant is not None:
        candidates = test_index[(test_index["dataset"] == dataset) & (test_index["participant"] == participant)]
        if candidates.empty:
            print(f"  No test windows for {dataset}/{participant} -- skipping window_predictions_recording.png")
            return
        tag = candidates.groupby("tag").size().idxmax()
        if candidates["tag"].nunique() > 1:
            print(f"  {dataset}/{participant} spans {candidates['tag'].nunique()} separate recordings -- using the longest: {tag}")
    else:
        counts = test_index.groupby(["dataset", "participant", "tag"]).size()
        dataset, participant, tag = counts.idxmax()  # most windows -> longest, most informative single recording

    part_index = test_index[
        (test_index["dataset"] == dataset) & (test_index["participant"] == participant) & (test_index["tag"] == tag)
    ].sort_values("start").reset_index(drop=True)
    if part_index.empty:
        print(f"  No test windows found for {dataset}/{participant}/{tag} -- skipping window_predictions_recording.png")
        return

    _plot_one_recording(
        models, part_index, dataset, participant, tag, out_dir / "window_predictions_recording.png", threshold
    )


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate + visualize finetune_window.py checkpoint(s) on finetune_test.")
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True,
                         help="One or more finetune_window.py checkpoints (pretext_task inferred from filename).")
    parser.add_argument("--names", nargs="+", default=None,
                         help="Display names for each checkpoint, same order as --checkpoints. Default: pretext_task.")
    parser.add_argument("--time-architecture", default="cnn", help="For any tfc checkpoint (default matches pretrain_tfc.py's default).")
    parser.add_argument("--resnet-base-width", type=int, default=32, help="For any tffc checkpoint (default matches pretrain_tffc.py's default).")
    parser.add_argument("--head-hidden-dim", type=int, default=64)
    parser.add_argument("--head-dropout", type=float, default=0.3)
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--dataset", default=None, help="Restrict finetune_test (all figures/metrics) to one dataset. Default: all.")
    parser.add_argument("--recording-dataset", default=None,
                         help="Dataset for the recording-strip figure's example (required if --participant is given "
                              "and isn't 'all'; participant IDs repeat across datasets). Ignored if --participant=all.")
    parser.add_argument("--participant", default=None,
                         help="Participant for the recording-strip figure, or 'all' to plot every distinct test "
                              "recording (one PNG each, mirrors visualize_inference.py's --example-participant all). "
                              "Default: the single (dataset, participant, tag) with the most test windows.")
    parser.add_argument("--threshold", type=float, default=0.5,
                         help="Decision threshold for probability->0/1 (e.g. 0.4). Applied to the confusion "
                              "matrix, printed metrics, and the recording-strip 'decision' row. ROC/PR curves "
                              "are threshold-independent by construction and unaffected.")
    parser.add_argument("--out-dir", type=Path, default=Path("checkpoints/plots/window_inference"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    n_channels = len(SENSOR_CHANNELS)
    names = args.names or [infer_pretext_task(p) for p in args.checkpoints]
    if len(names) != len(args.checkpoints):
        raise ValueError("--names must have the same length as --checkpoints if given.")

    print("\nWindow Inference Visualization")

    print("\n  Loading checkpoint(s) ...")
    models: dict[str, WindowClassifier] = {}
    for path, name in zip(args.checkpoints, names):
        model, _ = load_window_classifier(
            path, n_channels, args.time_architecture, args.resnet_base_width, args.head_hidden_dim, args.head_dropout
        )
        models[name] = model

    print("\n  Loading finetune_test ...")
    test_index = load_index(args.processed_root, "finetune_test", None, args.seed, args.dataset)
    print(f"  Test: {len(test_index):,} windows  |  {test_index['participant'].nunique()} participants")
    test_loader = DataLoader(
        WindowDataset(test_index.sort_values("segment_path").reset_index(drop=True), with_label=True),
        batch_size=128, shuffle=False,
    )

    print("\n  [1/3] Running inference on the full test set ...")
    results = {}
    for name, model in models.items():
        r = infer_test_set(model, test_loader)
        m = metrics_from_probs(r["probs"], r["labels"], args.threshold)
        print(f"    [{name}] f1={m['f1']:.3f}  prec={m['precision']:.3f}  rec={m['recall']:.3f}  auc={m['auc']:.3f}  acc={m['accuracy']:.3f}")
        results[name] = r

    print(f"\n  [2/3] Plotting metrics comparison + embedding space  (threshold={args.threshold:.2f}) ...")
    plot_metrics_comparison(results, args.out_dir / "window_metrics_comparison.png", args.threshold)
    plot_embedding_space(results, args.out_dir / "embedding_space.png")

    print("\n  [3/3] Plotting recording(s)' window-level predictions ...")
    plot_recording_predictions(
        models, test_index, args.out_dir,
        args.recording_dataset, args.participant, args.threshold,
    )

    print(f"\n  Plots saved to  {args.out_dir}/")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
