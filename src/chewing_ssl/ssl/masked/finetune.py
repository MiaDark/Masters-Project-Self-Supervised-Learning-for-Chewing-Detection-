"""Fine-tune a pretrained BiGRU encoder for per-timestep chewing detection.

Two-phase strategy
------------------
Phase 1  (--phase1-epochs, default 5):
    Encoder frozen.  Only the classification head trains.  Fast convergence,
    no risk of corrupting pretrained representations before the head is stable.

Phase 2  (--phase2-epochs, default 10):
    Encoder unfrozen with a 10x lower learning rate than the head.
    Allows the GRU representations to adapt to the chewing task while
    preventing catastrophic forgetting of the pretraining structure.

Metrics (logged every epoch)
-----------------------------
  loss       -- binary cross-entropy on z-scored per-timestep labels
  accuracy   -- fraction of timesteps correctly classified (threshold 0.5)
  precision  -- of all timesteps predicted as chewing, fraction truly chewing
  recall     -- of all actual chewing timesteps, fraction correctly detected
  f1         -- harmonic mean of precision & recall  ← primary metric
  auc        -- ROC-AUC, threshold-free discrimination ability

The checkpoint is saved when val_f1 improves, not when val_loss improves.
F1 is the metric that matters for the downstream chewing detection task.

Loading checkpoints
-------------------
Expects a checkpoint saved by chewing_ssl.ssl.masked.pretrain as a
BiGRUAutoencoder (keys = 'mask_token', 'encoder.*', 'head.*'). Checkpoints
with double-nested 'encoder.encoder.*' keys (an older, now-removed pretext-task
pretraining format) are still auto-detected and unwrapped for backward
compatibility with checkpoints saved before that format was dropped.

Usage (run from project root)
------------------------------
  # Step 0 — pretrain BiGRU (if not done yet):
  python -m chewing_ssl.ssl.masked.pretrain \\
      --architecture bigru --dataset SIPMEatingStudy

  # Step 1 — fine-tune:
  python -m chewing_ssl.ssl.masked.finetune \\
      --checkpoint checkpoints/pretrain/bigru_personal_best.pt \\
      --dataset SIPMEatingStudy
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm
from torch.utils.data import DataLoader

from chewing_ssl.common.data.preprocess import SENSOR_CHANNELS
from chewing_ssl.common.data.windowing import WindowDataset
from chewing_ssl.ssl.models.bigru_autoencoder import BiGRUAutoencoder
from chewing_ssl.common.tracking.mlflow_utils import log_metrics, log_params, start_run
from chewing_ssl.common.utils.seed import set_seed

N_CHANNELS = len(SENSOR_CHANNELS)
HIDDEN_SIZE = 64  # matches BiGRUAutoencoder default
GRU_LAYERS = 2


# ── Model ─────────────────────────────────────────────────────────────────────


class ChewingDetector(nn.Module):
    """BiGRU encoder + per-timestep binary chewing head.

    The GRU produces a 128-dimensional context vector at every timestep
    (64 forward + 64 backward).  The head projects each timestep's vector
    independently to a single chewing logit, so the output is (B, T) —
    one probability per input sample.
    """

    def __init__(
        self,
        n_channels: int = N_CHANNELS,
        hidden_size: int = HIDDEN_SIZE,
        gru_layers: int = GRU_LAYERS,
        head_dropout: float = 0.3,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(
            input_size=n_channels,
            hidden_size=hidden_size,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1 if gru_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(head_dropout)
        self.head = nn.Linear(hidden_size * 2, 1)

    def copy_encoder_weights(self, gru_module: nn.GRU) -> None:
        self.gru.load_state_dict(gru_module.state_dict())

    def freeze_encoder(self) -> None:
        for p in self.gru.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self) -> None:
        for p in self.gru.parameters():
            p.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T)  →  logits: (B, T)"""
        h, _ = self.gru(x.transpose(1, 2))  # (B, T, 2*hidden)
        return self.head(self.dropout(h)).squeeze(-1)  # (B, T)

    def hidden_states(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw GRU hidden states (B, T, 2*hidden) without grad."""
        with torch.no_grad():
            h, _ = self.gru(x.transpose(1, 2))
        return h


# ── Utilities ─────────────────────────────────────────────────────────────────


def load_bigru_from_checkpoint(path: Path) -> BiGRUAutoencoder:
    """Load a BiGRUAutoencoder from a pretrain checkpoint.

    pretrain.py saves the BiGRUAutoencoder directly:
        keys = 'mask_token', 'encoder.*' (GRU), 'head.*'
    Checkpoints from the older, now-removed pretext-task pretraining format
    wrapped it as 'encoder.mask_token', 'encoder.encoder.*', 'encoder.head.*',
    identified by the double-nested 'encoder.encoder.*' keys, and are still
    auto-unwrapped here for backward compatibility.
    """
    state = torch.load(path, map_location="cpu", weights_only=True)["model_state_dict"]
    if any(k.startswith("encoder.encoder.") for k in state):
        # Legacy pretext-task format: strip the outer 'encoder.' prefix
        # to recover the BiGRUAutoencoder state dict.
        state = {k[len("encoder.") :]: v for k, v in state.items() if k.startswith("encoder.")}
    bigru = BiGRUAutoencoder(n_channels=N_CHANNELS)
    bigru.load_state_dict(state)
    return bigru


def pick_device(requested: str | None = None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_index(
    processed_root: Path,
    variant: str,
    purpose: str,
    max_windows: int | None,
    seed: int,
    dataset: str | None,
) -> pd.DataFrame:
    index = pd.read_parquet(processed_root / "index" / variant / f"{purpose}.parquet")
    if dataset:
        index = index[index["dataset"] == dataset].reset_index(drop=True)
    if max_windows is not None and len(index) > max_windows:
        index = index.sample(n=max_windows, random_state=seed).reset_index(drop=True)
    return index


def shuffle_by_segment(index: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    segs = index["segment_path"].unique().copy()
    rng.shuffle(segs)
    order = {s: i for i, s in enumerate(segs)}
    return (
        index.assign(_o=index["segment_path"].map(order))
        .sort_values("_o")
        .drop(columns="_o")
        .reset_index(drop=True)
    )


def compute_metrics(logits: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5) -> dict:
    """Compute classification metrics over flattened (B*T,) arrays."""
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


# ── Training loop ─────────────────────────────────────────────────────────────


def run_epoch(
    model: ChewingDetector,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    pos_weight: torch.Tensor,
    epoch_label: str = "",
) -> dict:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss, n_batches = 0.0, 0
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    pw = pos_weight.to(device)
    desc = f"  [{'train' if is_train else 'val  '}] {epoch_label}"
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        pbar = tqdm(loader, desc=desc, leave=False, unit="batch")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            logits = model(x)  # (B, T)
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

    all_logits_cat = torch.cat(all_logits)
    all_labels_cat = torch.cat(all_labels)
    metrics = compute_metrics(all_logits_cat, all_labels_cat)
    metrics["loss"] = total_loss / max(n_batches, 1)
    return metrics


def print_metrics(metrics: dict, prefix: str = "") -> None:
    print(
        f"{prefix}loss={metrics['loss']:.4f}  "
        f"f1={metrics['f1']:.3f}  "
        f"prec={metrics['precision']:.3f}  "
        f"rec={metrics['recall']:.3f}  "
        f"auc={metrics['auc']:.3f}  "
        f"acc={metrics['accuracy']:.3f}"
    )


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune BiGRU encoder for chewing detection.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/pretrain/bigru_personal_best.pt"),
        help="Pretrained BiGRU checkpoint (current or legacy pretext format, auto-detected)",
    )
    parser.add_argument("--variant", choices=["personal", "global"], default="personal")
    parser.add_argument("--dataset", default="SIPMEatingStudy")
    parser.add_argument(
        "--processed-root", type=Path, default=Path("src/chewing_ssl/data/data/processed")
    )
    parser.add_argument("--out-dir", type=Path, default=Path("checkpoints/finetune"))
    parser.add_argument(
        "--phase1-epochs",
        type=int,
        default=5,
        help="Epochs with encoder frozen (head-only training)",
    )
    parser.add_argument(
        "--phase2-epochs",
        type=int,
        default=10,
        help="Epochs with encoder unfrozen (full fine-tuning)",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--head-lr", type=float, default=1e-3, help="Learning rate for the classification head"
    )
    parser.add_argument(
        "--encoder-lr",
        type=float,
        default=1e-4,
        help="Learning rate for the encoder in phase 2 (10x lower by default)",
    )
    parser.add_argument("--max-train-windows", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    device = pick_device(args.device)

    print(f"\n{'=' * 60}")
    print("  Fine-tuning BiGRU for chewing detection")
    print(f"  Checkpoint   : {args.checkpoint}")
    print(f"  Dataset      : {args.dataset}")
    print(f"  Device       : {device}")
    print(f"  Phase 1      : {args.phase1_epochs} epochs  (encoder frozen, head LR={args.head_lr})")
    print(
        f"  Phase 2      : {args.phase2_epochs} epochs  (encoder LR={args.encoder_lr}, head LR={args.head_lr})"
    )
    print(f"{'=' * 60}\n")

    # ── Load pretrained encoder ──
    print("  Loading pretrained BiGRU encoder ...")
    bigru = load_bigru_from_checkpoint(args.checkpoint)
    model = ChewingDetector()
    model.copy_encoder_weights(bigru.encoder)
    model.freeze_encoder()
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params : {n_params:,}  |  Trainable (phase 1): {n_trainable:,}\n")

    # ── Data ──
    train_index = load_index(
        args.processed_root,
        args.variant,
        "finetune_train",
        args.max_train_windows,
        args.seed,
        args.dataset,
    )
    val_index = load_index(
        args.processed_root, args.variant, "finetune_val", None, args.seed, args.dataset
    )
    test_index = load_index(
        args.processed_root, args.variant, "finetune_test", None, args.seed, args.dataset
    )

    print(
        f"  Train : {len(train_index):>5,} windows | {train_index['participant'].nunique()} participants"
    )
    print(
        f"  Val   : {len(val_index):>5,} windows | {val_index['participant'].nunique()} participants"
    )
    print(
        f"  Test  : {len(test_index):>5,} windows | {test_index['participant'].nunique()} participants"
    )

    # Estimate chewing fraction from training index (no parquet load needed).
    # Used to compute pos_weight for class-balanced BCE.
    # Approximate: load a random sample of 10 segments to estimate.
    sample_segs = train_index["segment_path"].unique()[:10]
    chew_vals = []
    for sp in sample_segs:
        df = pd.read_parquet(sp, columns=["label"])
        chew_vals.append(df["label"].mean())
    chew_frac = float(np.mean(chew_vals))
    pos_weight = torch.tensor([(1.0 - chew_frac) / max(chew_frac, 1e-6)])
    print(f"\n  Chewing fraction (train sample): {chew_frac:.1%}")
    print(f"  pos_weight for BCE             : {pos_weight.item():.2f}\n")

    val_loader = DataLoader(
        WindowDataset(
            val_index.sort_values("segment_path").reset_index(drop=True), with_label=True
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        WindowDataset(
            test_index.sort_values("segment_path").reset_index(drop=True), with_label=True
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    best_val_f1 = 0.0
    history: list[dict] = []

    ckpt_name = f"{args.checkpoint.stem.replace('best', 'finetune')}"
    best_ckpt = args.out_dir / f"{ckpt_name}_best.pt"

    with start_run(run_name=f"finetune_{ckpt_name}"):
        log_params(
            {
                "pretrained_from": str(args.checkpoint),
                "dataset": args.dataset,
                "variant": args.variant,
                "phase1_epochs": args.phase1_epochs,
                "phase2_epochs": args.phase2_epochs,
                "head_lr": args.head_lr,
                "encoder_lr": args.encoder_lr,
                "batch_size": args.batch_size,
                "n_train_windows": len(train_index),
                "n_val_windows": len(val_index),
                "pos_weight": pos_weight.item(),
                "seed": args.seed,
            }
        )

        # ── Phase 1: encoder frozen ──────────────────────────────────────────
        if args.phase1_epochs > 0:
            print(f"{'─' * 60}")
            print(f"  Phase 1: {args.phase1_epochs} epochs  (encoder FROZEN)")
            print(f"{'─' * 60}")
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()), lr=args.head_lr
            )
            for epoch in range(args.phase1_epochs):
                t0 = time.time()
                lbl = f"P1 epoch {epoch + 1}/{args.phase1_epochs}"
                print(f"  {lbl}")
                shuffled = shuffle_by_segment(train_index, args.seed + epoch)
                train_loader = DataLoader(
                    WindowDataset(shuffled, with_label=True),
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=0,
                )
                train_m = run_epoch(model, train_loader, device, optimizer, pos_weight, lbl)
                val_m = run_epoch(model, val_loader, device, None, pos_weight, lbl)
                dt = time.time() - t0

                log_metrics({f"p1_train_{k}": v for k, v in train_m.items()}, step=epoch)
                log_metrics({f"p1_val_{k}": v for k, v in val_m.items()}, step=epoch)
                print_metrics(train_m, prefix="  train  ")
                print_metrics(val_m, prefix="  val    ")
                print(f"  ({dt:.1f}s)")

                row = {
                    "phase": 1,
                    "epoch": epoch,
                    **{f"train_{k}": v for k, v in train_m.items()},
                    **{f"val_{k}": v for k, v in val_m.items()},
                }
                history.append(row)

                if val_m["f1"] > best_val_f1:
                    best_val_f1 = val_m["f1"]
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "phase": 1,
                            "epoch": epoch,
                            "val_f1": best_val_f1,
                            "val_metrics": val_m,
                        },
                        best_ckpt,
                    )
                    print(f"  ✓ New best val_f1={best_val_f1:.3f}")
                print()

        # ── Phase 2: encoder unfrozen ────────────────────────────────────────
        if args.phase2_epochs > 0:
            model.unfreeze_encoder()
            n_trainable2 = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"{'─' * 60}")
            print(f"  Phase 2: {args.phase2_epochs} epochs  (encoder UNFROZEN)")
            print(f"  Trainable params now: {n_trainable2:,}")
            print(f"{'─' * 60}")
            optimizer = torch.optim.AdamW(
                [
                    {"params": model.gru.parameters(), "lr": args.encoder_lr, "name": "encoder"},
                    {
                        "params": list(model.dropout.parameters()) + list(model.head.parameters()),
                        "lr": args.head_lr,
                        "name": "head",
                    },
                ]
            )
            for epoch in range(args.phase2_epochs):
                t0 = time.time()
                lbl = f"P2 epoch {epoch + 1}/{args.phase2_epochs}"
                print(f"  {lbl}")
                shuffled = shuffle_by_segment(train_index, args.seed + 1000 + epoch)
                train_loader = DataLoader(
                    WindowDataset(shuffled, with_label=True),
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=0,
                )
                train_m = run_epoch(model, train_loader, device, optimizer, pos_weight, lbl)
                val_m = run_epoch(model, val_loader, device, None, pos_weight, lbl)
                dt = time.time() - t0

                log_metrics({f"p2_train_{k}": v for k, v in train_m.items()}, step=epoch)
                log_metrics({f"p2_val_{k}": v for k, v in val_m.items()}, step=epoch)
                print_metrics(train_m, prefix="  train  ")
                print_metrics(val_m, prefix="  val    ")
                print(f"  ({dt:.1f}s)")

                row = {
                    "phase": 2,
                    "epoch": epoch,
                    **{f"train_{k}": v for k, v in train_m.items()},
                    **{f"val_{k}": v for k, v in val_m.items()},
                }
                history.append(row)

                if val_m["f1"] > best_val_f1:
                    best_val_f1 = val_m["f1"]
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "phase": 2,
                            "epoch": epoch,
                            "val_f1": best_val_f1,
                            "val_metrics": val_m,
                        },
                        best_ckpt,
                    )
                    print(f"  ✓ New best val_f1={best_val_f1:.3f}")
                print()

        # ── Final test evaluation ────────────────────────────────────────────
        print(f"{'─' * 60}")
        print("  Loading best checkpoint for test evaluation ...")
        best_state = torch.load(best_ckpt, map_location="cpu", weights_only=True)
        model.load_state_dict(best_state["model_state_dict"])
        print(
            f"  Best model: phase={best_state['phase']}  epoch={best_state['epoch'] + 1}  "
            f"val_f1={best_state['val_f1']:.3f}\n"
        )

        test_m = run_epoch(model, test_loader, device, None, pos_weight, "test")
        print("  TEST SET RESULTS:")
        print_metrics(test_m, prefix="  ")
        log_metrics({f"test_{k}": v for k, v in test_m.items()})

        # Save history + test metrics
        result = {"history": history, "test_metrics": test_m, "best_val_f1": best_val_f1}
        hist_path = args.out_dir / f"{ckpt_name}_history.json"
        hist_path.write_text(json.dumps(result, indent=2))

    print(f"\n{'=' * 60}")
    print(f"  Done.  Best val_f1={best_val_f1:.3f}  Test F1={test_m['f1']:.3f}")
    print(f"  Checkpoint  →  {best_ckpt}")
    print(f"  History     →  {hist_path}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
