"""Fine-tune a per-timestep chewing detector on top of a masked-reconstruction
pretrained encoder -- architecture-agnostic (CNN or Bi-GRU) and pretext-task
-agnostic (channel-only / span-only / both masking), since pretrain.py now
stamps `architecture`, `pretext_task`, `window_len` and `model_kwargs`
directly into the checkpoint.

ChewingDetector wraps ANY of the two encoders behind one interface:
  - Bi-GRU's encode() already returns full temporal resolution (B, T, D).
  - CNN's encode() returns the downsampled bottleneck (B, T', D), T' < T --
    the head's output is linearly upsampled back to T (a small, decoupled
    step, not reusing the pretrained decoder/skip connections) so both
    architectures expose the same (B, T) per-timestep logit interface.

Three-phase gradual-unfreezing strategy:
  Phase 1 (--phase1-epochs): encoder fully frozen, head-only.
  Phase 2 (--phase2-epochs): only the encoder's LAST block unfrozen (the
      final Bi-GRU layer / the CNN bottleneck) at a low LR.
  Phase 3 (--phase3-epochs): entire encoder unfrozen at an even lower LR.
Each phase gets its own cosine LR schedule (--lr-schedule cosine, default)
decaying from that phase's starting LR to ~0 over that phase's epochs.

Checkpoint saved on best val_f1 (not val_loss) -- F1 is what matters for the
downstream chewing-detection task. Final held-out test evaluation runs
automatically once training finishes.
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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from chewing_ssl.common.data.preprocess import SENSOR_CHANNELS
from chewing_ssl.common.data.windowing import WindowDataset
from chewing_ssl.common.tracking.mlflow_utils import log_metrics, log_params, start_run
from chewing_ssl.common.utils.seed import set_seed
from chewing_ssl.ssl.masked.pretrain import MODEL_KWARGS, load_index, pick_device, shuffle_by_segment
from chewing_ssl.ssl.models import ARCHITECTURES

N_CHANNELS = len(SENSOR_CHANNELS)


# Model

class ChewingDetector(nn.Module):
    """Any pretrained encoder (CNN or Bi-GRU) + a per-timestep binary chewing head."""

    def __init__(
        self,
        encoder: nn.Module,
        architecture: str,
        embedding_dim: int,
        head_hidden_dim: int = 64,
        head_dropout: float = 0.3,
    ):
        super().__init__()
        self.encoder = encoder
        self.architecture = architecture
        self.head = nn.Sequential(
            nn.Linear(embedding_dim, head_hidden_dim),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden_dim, head_hidden_dim),
            nn.GELU(),
            nn.Linear(head_hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T) -> logits: (B, T)."""
        h = self.encoder.encode(x)  # (B, T', embedding_dim)
        logits = self.head(h).squeeze(-1)  # (B, T')
        t_target = x.shape[-1]
        if logits.shape[1] != t_target:
            # CNN's encode() is downsampled (T' < T) -- upsample the HEAD'S
            # OUTPUT back to full resolution.
            # Decoupled from the pretrained decoder/skip connections on
            # purpose: this head has to work for either architecture.
            logits = F.interpolate(
                logits.unsqueeze(1), size=t_target, mode="linear", align_corners=False
            ).squeeze(1)
        return logits

    def freeze_encoder(self) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = True

    def last_encoder_block_parameters(self) -> list[nn.Parameter]:
        """The encoder sub-module closest to the embedding output -- the
        final Bi-GRU layer, or the CNN's bottleneck block."""
        if self.architecture == "bigru":
            return self.encoder.layer_parameters(self.encoder.num_layers - 1)
        if self.architecture == "cnn":
            return list(self.encoder.bottleneck.parameters())
        raise ValueError(f"Unknown architecture {self.architecture!r}")

    def unfreeze_last_encoder_block(self) -> None:
        self.freeze_encoder()
        for p in self.last_encoder_block_parameters():
            p.requires_grad = True


def load_pretrained_encoder(path: Path) -> tuple[nn.Module, dict]:
    """Load an encoder from a pretrain.py checkpoint, auto-detecting
    architecture/model_kwargs from the metadata pretrain.py now stamps into
    every checkpoint (architecture, pretext_task, window_len, model_kwargs).
    Requires the new checkpoint format -- checkpoints from before this
    metadata existed need to be regenerated (see docs/ssl_masked.md)."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    missing = [k for k in ("architecture", "model_state_dict") if k not in state]
    if missing:
        raise ValueError(
            f"{path} is missing {missing} -- looks like a pre-ablation-study checkpoint "
            "(saved before pretrain.py stamped architecture/pretext_task/model_kwargs "
            "metadata). Re-run pretrain.py to regenerate it."
        )
    architecture = state["architecture"]
    model_kwargs = state.get("model_kwargs", MODEL_KWARGS[architecture])
    encoder = ARCHITECTURES[architecture](n_channels=N_CHANNELS, **model_kwargs)
    encoder.load_state_dict(state["model_state_dict"])
    meta = {
        "architecture": architecture,
        "pretext_task": state.get("pretext_task", "unknown"),
        "window_len": state.get("window_len"),
        "embedding_dim": encoder.embedding_dim,
        "model_kwargs": model_kwargs,
    }
    return encoder, meta


# Utilities


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


# Training loop


def run_epoch(
    model: ChewingDetector,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    pos_weight: torch.Tensor,
    epoch_label: str = "",
    grad_clip: float = 1.0,
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
                if grad_clip is not None:
                    nn.utils.clip_grad_norm_(
                        (p for p in model.parameters() if p.requires_grad), grad_clip
                    )
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


def run_phase(
    phase: int,
    n_epochs: int,
    model: ChewingDetector,
    optimizer: torch.optim.Optimizer,
    lr_schedule: str,
    train_index: pd.DataFrame,
    val_loader: DataLoader,
    device: torch.device,
    batch_size: int,
    pos_weight: torch.Tensor,
    seed: int,
    seed_offset: int,
    best_ckpt: Path,
    best_val_f1: float,
    history: list[dict],
    extra_meta: dict,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> float:
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(n_epochs, 1))
        if lr_schedule == "cosine"
        else None
    )
    for epoch in range(n_epochs):
        t0 = time.time()
        lbl = f"P{phase} epoch {epoch + 1}/{n_epochs}"
        print(f"  {lbl}")
        shuffled = shuffle_by_segment(train_index, seed + seed_offset + epoch)
        train_loader = DataLoader(
            WindowDataset(shuffled, with_label=True),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            # "spawn" avoids a fork-after-CUDA-init deadlock: the model is already
            # on the GPU by the time this loader is (re)built every epoch, and
            # forking a process with a live CUDA context is unsafe.
            multiprocessing_context="spawn" if num_workers > 0 else None,
        )
        train_m = run_epoch(model, train_loader, device, optimizer, pos_weight, lbl)
        val_m = run_epoch(model, val_loader, device, None, pos_weight, lbl)
        if scheduler is not None:
            scheduler.step()
        dt = time.time() - t0

        log_metrics({f"p{phase}_train_{k}": v for k, v in train_m.items()}, step=epoch)
        log_metrics({f"p{phase}_val_{k}": v for k, v in val_m.items()}, step=epoch)
        print_metrics(train_m, prefix="  train  ")
        print_metrics(val_m, prefix="  val    ")
        print(f"  ({dt:.1f}s)")

        history.append(
            {
                "phase": phase,
                "epoch": epoch,
                **{f"train_{k}": v for k, v in train_m.items()},
                **{f"val_{k}": v for k, v in val_m.items()},
            }
        )

        if val_m["f1"] > best_val_f1:
            best_val_f1 = val_m["f1"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "phase": phase,
                    "epoch": epoch,
                    "val_f1": best_val_f1,
                    **extra_meta,
                },
                best_ckpt,
            )
            print(f"  ✓ New best val_f1={best_val_f1:.3f}")
        print()
    return best_val_f1


# Entry point


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune a per-timestep chewing detector on a masked-reconstruction checkpoint."
    )
    parser.add_argument(
        "--checkpoint", type=Path, required=True, help="From pretrain.py (architecture auto-detected)."
    )
    parser.add_argument(
        "--dataset", default=None, help="Restrict to one dataset. Default: all datasets."
    )
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--out-dir", type=Path, default=Path("checkpoints/finetune"))
    parser.add_argument("--head-hidden-dim", type=int, default=64)
    parser.add_argument("--head-dropout", type=float, default=0.3)
    parser.add_argument("--phase1-epochs", type=int, default=10, help="Encoder frozen, head-only.")
    parser.add_argument(
        "--phase2-epochs", type=int, default=15, help="Only the encoder's last block unfrozen."
    )
    parser.add_argument("--phase3-epochs", type=int, default=20, help="Entire encoder unfrozen.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument(
        "--last-block-lr", type=float, default=1e-4, help="Phase 2 encoder LR (10x below head-lr)."
    )
    parser.add_argument(
        "--full-lr", type=float, default=3e-5, help="Phase 3 encoder LR (below last-block-lr)."
    )
    parser.add_argument("--lr-schedule", choices=["none", "cosine"], default="cosine")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--max-train-windows", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    device = pick_device(args.device)

    encoder, meta = load_pretrained_encoder(args.checkpoint)
    run_name = f"{meta['pretext_task']}_{meta['architecture']}_w{meta['window_len']}"
    model = ChewingDetector(
        encoder, meta["architecture"], meta["embedding_dim"], args.head_hidden_dim, args.head_dropout
    )
    model.freeze_encoder()
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n{'=' * 60}")
    print(f"  Fine-tuning chewing detector  [{run_name}]")
    print(f"  Checkpoint    : {args.checkpoint}")
    print(f"  Architecture  : {meta['architecture']}  |  Pretext task: {meta['pretext_task']}  |  "
          f"Window: {meta['window_len']} samples")
    print(f"  Dataset       : {args.dataset or 'all'}")
    print(f"  Device        : {device}")
    print(f"  Total params  : {n_params:,}  |  Trainable (phase 1): {n_trainable:,}")
    print(
        f"  Phase 1: {args.phase1_epochs} epochs (frozen)  |  "
        f"Phase 2: {args.phase2_epochs} epochs (last block, lr={args.last_block_lr})  |  "
        f"Phase 3: {args.phase3_epochs} epochs (full, lr={args.full_lr})"
    )
    print(f"{'=' * 60}\n")

    train_index = load_index(
        args.processed_root, "finetune_train", args.max_train_windows, args.seed, args.dataset
    )
    val_index = load_index(args.processed_root, "finetune_val", None, args.seed, args.dataset)
    test_index = load_index(args.processed_root, "finetune_test", None, args.seed, args.dataset)

    print(
        f"  Train : {len(train_index):>5,} windows | {train_index['participant'].nunique()} participants"
    )
    print(
        f"  Val   : {len(val_index):>5,} windows | {val_index['participant'].nunique()} participants"
    )
    print(
        f"  Test  : {len(test_index):>5,} windows | {test_index['participant'].nunique()} participants"
    )

    sample_segs = train_index["segment_path"].unique()[:10]
    chew_vals = [pd.read_parquet(sp, columns=["label"])["label"].mean() for sp in sample_segs]
    chew_frac = float(np.mean(chew_vals))
    pos_weight = torch.tensor([(1.0 - chew_frac) / max(chew_frac, 1e-6)])
    print(f"\n  Chewing fraction (train sample): {chew_frac:.1%}")
    print(f"  pos_weight for BCE             : {pos_weight.item():.2f}\n")

    pin_memory = device.type == "cuda"
    mp_context = "spawn" if args.num_workers > 0 else None
    val_loader = DataLoader(
        WindowDataset(val_index.sort_values("segment_path").reset_index(drop=True), with_label=True),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        multiprocessing_context=mp_context,
    )
    test_loader = DataLoader(
        WindowDataset(test_index.sort_values("segment_path").reset_index(drop=True), with_label=True),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        multiprocessing_context=mp_context,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    best_val_f1 = -1.0  # not 0.0 -- an all-zero-F1 epoch must still save a checkpoint
    history: list[dict] = []
    best_ckpt = args.out_dir / f"{run_name}_best.pt"
    # Stamped into every saved checkpoint so evaluate_bouts.py / plot_predictions.py
    # can reconstruct the model without needing the original pretrain checkpoint again.
    extra_meta = {
        "architecture": meta["architecture"],
        "pretext_task": meta["pretext_task"],
        "window_len": meta["window_len"],
        "embedding_dim": meta["embedding_dim"],
        "model_kwargs": meta["model_kwargs"],
        "head_hidden_dim": args.head_hidden_dim,
        "head_dropout": args.head_dropout,
    }

    with start_run(run_name=f"finetune_{run_name}"):
        log_params(
            {
                "pretrained_from": str(args.checkpoint),
                "architecture": meta["architecture"],
                "pretext_task": meta["pretext_task"],
                "window_len": meta["window_len"],
                "dataset": args.dataset or "all",
                "phase1_epochs": args.phase1_epochs,
                "phase2_epochs": args.phase2_epochs,
                "phase3_epochs": args.phase3_epochs,
                "head_lr": args.head_lr,
                "last_block_lr": args.last_block_lr,
                "full_lr": args.full_lr,
                "lr_schedule": args.lr_schedule,
                "batch_size": args.batch_size,
                "n_train_windows": len(train_index),
                "n_val_windows": len(val_index),
                "pos_weight": pos_weight.item(),
                "seed": args.seed,
            }
        )

        # ── Phase 1: encoder frozen, head-only ──
        if args.phase1_epochs > 0:
            print(f"{'─' * 60}\n  Phase 1: encoder FROZEN\n{'─' * 60}")
            optimizer = torch.optim.AdamW(model.head.parameters(), lr=args.head_lr)
            best_val_f1 = run_phase(
                1, args.phase1_epochs, model, optimizer, args.lr_schedule, train_index, val_loader,
                device, args.batch_size, pos_weight, args.seed, 0, best_ckpt, best_val_f1, history, extra_meta,
                num_workers=args.num_workers, pin_memory=pin_memory,
            )

        # ── Phase 2: only the last encoder block unfrozen ──
        if args.phase2_epochs > 0:
            model.unfreeze_last_encoder_block()
            n_trainable2 = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"{'─' * 60}\n  Phase 2: last encoder block UNFROZEN  ({n_trainable2:,} trainable)\n{'─' * 60}")
            optimizer = torch.optim.AdamW(
                [
                    {"params": model.last_encoder_block_parameters(), "lr": args.last_block_lr},
                    {"params": model.head.parameters(), "lr": args.head_lr},
                ]
            )
            best_val_f1 = run_phase(
                2, args.phase2_epochs, model, optimizer, args.lr_schedule, train_index, val_loader,
                device, args.batch_size, pos_weight, args.seed, 1000, best_ckpt, best_val_f1, history, extra_meta,
                num_workers=args.num_workers, pin_memory=pin_memory,
            )

        # ── Phase 3: entire encoder unfrozen ──
        if args.phase3_epochs > 0:
            model.unfreeze_encoder()
            n_trainable3 = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"{'─' * 60}\n  Phase 3: FULL encoder UNFROZEN  ({n_trainable3:,} trainable)\n{'─' * 60}")
            optimizer = torch.optim.AdamW(
                [
                    {"params": model.encoder.parameters(), "lr": args.full_lr},
                    {"params": model.head.parameters(), "lr": args.head_lr},
                ]
            )
            best_val_f1 = run_phase(
                3, args.phase3_epochs, model, optimizer, args.lr_schedule, train_index, val_loader,
                device, args.batch_size, pos_weight, args.seed, 2000, best_ckpt, best_val_f1, history, extra_meta,
                num_workers=args.num_workers, pin_memory=pin_memory,
            )

        # ── Final test evaluation ──
        print(f"{'─' * 60}")
        print("  Loading best checkpoint for test evaluation ...")
        best_state = torch.load(best_ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(best_state["model_state_dict"])
        print(
            f"  Best model: phase={best_state['phase']}  epoch={best_state['epoch'] + 1}  "
            f"val_f1={best_state['val_f1']:.3f}\n"
        )

        test_m = run_epoch(model, test_loader, device, None, pos_weight, "test")
        print("  TEST SET RESULTS:")
        print_metrics(test_m, prefix="  ")
        log_metrics({f"test_{k}": v for k, v in test_m.items()})

        result = {
            "run_name": run_name,
            "architecture": meta["architecture"],
            "pretext_task": meta["pretext_task"],
            "window_len": meta["window_len"],
            "history": history,
            "test_metrics": test_m,
            "best_val_f1": best_val_f1,
        }
        hist_path = args.out_dir / f"{run_name}_history.json"
        hist_path.write_text(json.dumps(result, indent=2))

    print(f"\n{'=' * 60}")
    print(f"  Done.  [{run_name}]  Best val_f1={best_val_f1:.3f}  Test F1={test_m['f1']:.3f}")
    print(f"  Checkpoint  →  {best_ckpt}")
    print(f"  History     →  {hist_path}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
