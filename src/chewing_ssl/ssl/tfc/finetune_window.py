"""Fine-tune a window-level chewing classifier on top of a TF-C- or
TF-FC-pretrained encoder (chewing_ssl.ssl.tfc.pretrain_tfc / pretrain_tffc).

Checkpoint format is auto-detected from the saved `pretext_task` key ("tfc" or
"tffc") -- same auto-detection spirit as the existing finetune.py's format
sniffing for the masked-reconstruction checkpoints, just simpler since both
new pretraining scripts stamp their format explicitly.

Window-level, not per-timestep (see pretrain_tfc.py's module docstring for
why): each window's dense per-timestep label array is collapsed to a single
target via majority vote (mean >= 0.5) before training.

Two-phase strategy, matching finetune.py's convention:
  Phase 1 (--phase1-epochs, default 5): encoder(s) frozen, only the
      classification head trains -- fast, stable convergence before touching
      pretrained weights.
  Phase 2 (--phase2-epochs, default 10): encoder(s) unfrozen at a 10x lower
      learning rate than the head, so the pretrained representations adapt
      without catastrophic forgetting.

Checkpoint is saved on best val_f1 (not val_loss) -- F1 is what matters for
the downstream chewing-detection task, same convention as finetune.py.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", message="An output with one or more elements was resized")


from chewing_ssl.common.data.preprocess import SENSOR_CHANNELS
from chewing_ssl.common.data.windowing import WindowDataset
from chewing_ssl.ssl.models import ARCHITECTURES
from chewing_ssl.ssl.models.augmentation import to_frequency
from chewing_ssl.ssl.models.finetune_head import WindowClassifierHead
from chewing_ssl.ssl.models.frequency_encoder import FrequencyEncoder
from chewing_ssl.ssl.models.resnet1d import ResNet1D
from chewing_ssl.ssl.masked.pretrain import MODEL_KWARGS, load_index, pick_device, shuffle_by_segment
from chewing_ssl.common.utils.seed import set_seed


class ContrastivePretrainedEncoder(nn.Module):
    """Wraps either the TF-C pair (time + frequency encoders) or the TF-FC
    single ResNet encoder behind one uniform `.embed(x) -> (B, embedding_dim)`
    interface, so the rest of this script doesn't need to branch on which
    pretext task produced the checkpoint.
    """

    def __init__(self, checkpoint: dict, n_channels: int):
        super().__init__()
        self.pretext_task = checkpoint["pretext_task"]

        if self.pretext_task == "tfc":
            arch = checkpoint["time_architecture"]
            self.time_encoder = ARCHITECTURES[arch](n_channels=n_channels, **MODEL_KWARGS[arch])
            self.time_encoder.load_state_dict(checkpoint["time_encoder_state_dict"])
            self.freq_encoder = FrequencyEncoder(n_channels=n_channels)
            self.freq_encoder.load_state_dict(checkpoint["freq_encoder_state_dict"])
            self.embedding_dim = checkpoint["time_embedding_dim"] + checkpoint["freq_embedding_dim"]
        elif self.pretext_task == "tffc":
            self.encoder = ResNet1D(n_channels=n_channels, base_width=checkpoint["resnet_base_width"])
            self.encoder.load_state_dict(checkpoint["encoder_state_dict"])
            self.embedding_dim = checkpoint["embedding_dim"]
        else:
            raise ValueError(f"Unknown pretext_task {self.pretext_task!r} in checkpoint.")

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T) -> (B, embedding_dim), pooled (and, for tfc, concatenated)."""
        if self.pretext_task == "tfc":
            h_time = self.time_encoder.encode(x).mean(dim=1)
            h_freq = self.freq_encoder.encode(to_frequency(x)).mean(dim=1)
            return torch.cat([h_time, h_freq], dim=1)
        h = self.encoder.encode(x).mean(dim=1)
        return h

    def freeze(self) -> None:
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze(self) -> None:
        for p in self.parameters():
            p.requires_grad = True


class WindowClassifier(nn.Module):
    def __init__(self, encoder: ContrastivePretrainedEncoder, hidden_dim: int, dropout: float):
        super().__init__()
        self.encoder = encoder
        self.head = WindowClassifierHead(encoder.embedding_dim, hidden_dim=hidden_dim, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder.embed(x))


def window_labels(y: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """(B, window_len) dense per-timestep labels -> (B,) window-level target,
    majority vote (fraction of chewing timesteps >= threshold)."""
    return (y.float().mean(dim=1) >= threshold).float()


def compute_metrics(logits: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5) -> dict:
    probs = torch.sigmoid(logits).cpu().numpy().ravel()
    preds = (probs >= threshold).astype(int)
    y = labels.cpu().numpy().ravel().astype(int)
    n_unique = len(np.unique(y))
    return {
        "accuracy": float(accuracy_score(y, preds)),
        "precision": float(precision_score(y, preds, zero_division=0.0)),
        "recall": float(recall_score(y, preds, zero_division=0.0)),
        "f1": float(f1_score(y, preds, zero_division=0.0)),
        "auc": float(roc_auc_score(y, probs)) if n_unique > 1 else float("nan"),
    }


def run_epoch(
    model: WindowClassifier,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    pos_weight: torch.Tensor,
    epoch_label: str = "",
) -> dict:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss, n_batches = 0.0, 0
    all_logits, all_labels = [], []
    pw = pos_weight.to(device)
    desc = f"  [{'train' if is_train else 'val  '}] {epoch_label}"
    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        pbar = tqdm(loader, desc=desc, leave=False, unit="batch")
        for x, y in pbar:
            x, y = x.to(device), window_labels(y).to(device)
            logits = model(x)  # (B,)
            loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pw)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            all_logits.append(logits.detach().cpu())
            all_labels.append(y.cpu())
            pbar.set_postfix(loss=f"{total_loss / n_batches:.4f}")

    metrics = compute_metrics(torch.cat(all_logits), torch.cat(all_labels))
    metrics["loss"] = total_loss / max(n_batches, 1)
    return metrics


def print_metrics(metrics: dict, prefix: str = "") -> None:
    print(
        f"{prefix}loss={metrics['loss']:.4f}  f1={metrics['f1']:.3f}  "
        f"prec={metrics['precision']:.3f}  rec={metrics['recall']:.3f}  "
        f"auc={metrics['auc']:.3f}  acc={metrics['accuracy']:.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a window-level classifier on a TF-C/TF-FC checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True,
                         help="From pretrain_tfc.py or pretrain_tffc.py (pretext_task auto-detected).")
    parser.add_argument("--dataset", default=None, help="Restrict to one dataset. Default: all datasets.")
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--out-dir", type=Path, default=Path("checkpoints/finetune_window"))
    parser.add_argument("--head-hidden-dim", type=int, default=64)
    parser.add_argument("--head-dropout", type=float, default=0.3)
    parser.add_argument("--phase1-epochs", type=int, default=5, help="Epochs with encoder(s) frozen.")
    parser.add_argument("--phase2-epochs", type=int, default=10, help="Epochs with encoder(s) unfrozen.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--encoder-lr", type=float, default=1e-4, help="10x lower than head-lr by default.")
    parser.add_argument("--max-train-windows", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    device = pick_device(args.device)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    encoder = ContrastivePretrainedEncoder(checkpoint, n_channels=len(SENSOR_CHANNELS))
    encoder.freeze()
    model = WindowClassifier(encoder, args.head_hidden_dim, args.head_dropout).to(device)

    print(f"\n{'=' * 60}")
    print("  Fine-tuning window-level chewing classifier")
    print(f"  Pretext task : {encoder.pretext_task}  (from {args.checkpoint})")
    print(f"  Embedding dim: {encoder.embedding_dim}")
    print(f"  Dataset      : {args.dataset or 'all'}")
    print(f"  Device       : {device}")
    print(f"  Phase 1      : {args.phase1_epochs} epochs (encoder frozen, head_lr={args.head_lr})")
    print(f"  Phase 2      : {args.phase2_epochs} epochs (encoder_lr={args.encoder_lr}, head_lr={args.head_lr})")
    print(f"{'=' * 60}\n")

    train_index = load_index(args.processed_root, "finetune_train", args.max_train_windows, args.seed, args.dataset)
    val_index = load_index(args.processed_root, "finetune_val", None, args.seed, args.dataset)
    print(f"  Train : {len(train_index):>7,} windows  |  {train_index['participant'].nunique()} participants")
    print(f"  Val   : {len(val_index):>7,} windows  |  {val_index['participant'].nunique()} participants\n")

    val_loader = DataLoader(
        WindowDataset(val_index.sort_values("segment_path").reset_index(drop=True), with_label=True),
        batch_size=args.batch_size, shuffle=False,
    )

    # Class balance from the train split's window-level (majority-vote) labels,
    # for BCE pos_weight -- same convention as finetune.py.
    train_loader_probe = DataLoader(
        WindowDataset(train_index.sort_values("segment_path").reset_index(drop=True), with_label=True),
        batch_size=512, shuffle=False,
    )
    n_pos, n_total = 0, 0
    for _, y in train_loader_probe:
        wl = window_labels(y)
        n_pos += int(wl.sum().item())
        n_total += len(wl)
    pos_weight = torch.tensor([(n_total - n_pos) / max(n_pos, 1)], dtype=torch.float32)
    print(f"  Train window-level class balance: {n_pos}/{n_total} positive  (pos_weight={pos_weight.item():.2f})\n")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    best_val_f1 = -1.0
    history = []
    total_epochs = args.phase1_epochs + args.phase2_epochs

    for epoch in range(total_epochs):
        phase = 1 if epoch < args.phase1_epochs else 2
        if epoch == 0:
            optimizer = torch.optim.AdamW(model.head.parameters(), lr=args.head_lr)
        elif epoch == args.phase1_epochs:
            encoder.unfreeze()
            optimizer = torch.optim.AdamW(
                [
                    {"params": model.encoder.parameters(), "lr": args.encoder_lr},
                    {"params": model.head.parameters(), "lr": args.head_lr},
                ]
            )
            print(f"  → Phase 2: encoder unfrozen (encoder_lr={args.encoder_lr})\n")

        t0 = time.time()
        epoch_label = f"epoch {epoch + 1}/{total_epochs} (phase {phase})"
        print(f"Epoch {epoch + 1}/{total_epochs}  [phase {phase}]")

        shuffled = shuffle_by_segment(train_index, args.seed + epoch)
        train_loader = DataLoader(
            WindowDataset(shuffled, with_label=True), batch_size=args.batch_size, shuffle=False,
        )

        train_m = run_epoch(model, train_loader, device, optimizer, pos_weight, epoch_label)
        val_m = run_epoch(model, val_loader, device, None, pos_weight, epoch_label)
        dt = time.time() - t0

        print_metrics(train_m, prefix="  train: ")
        print_metrics(val_m, prefix="  val  : ")
        print(f"  ({dt:.1f}s)\n")

        history.append({"epoch": epoch, "phase": phase, "train": train_m, "val": val_m})

        if val_m["f1"] > best_val_f1:
            best_val_f1 = val_m["f1"]
            ckpt_path = args.out_dir / f"{encoder.pretext_task}_best.pt"
            torch.save(
                {"model_state_dict": model.state_dict(), "epoch": epoch, "val_f1": best_val_f1},
                ckpt_path,
            )
            print(f"  ✓ New best val_f1={best_val_f1:.4f}  →  {ckpt_path}\n")

    history_path = args.out_dir / f"{encoder.pretext_task}_history.json"
    history_path.write_text(json.dumps(history, indent=2))
    print(f"{'=' * 60}")
    print(f"  Done.  Best val_f1={best_val_f1:.4f}")
    print(f"  Checkpoint + history  →  {args.out_dir}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
