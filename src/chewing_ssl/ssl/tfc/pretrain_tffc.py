"""SSL pretraining: TF-FC (Time-Frequency Fusion Contrasting) approach, per
"Time-series representation learning via Time-Frequency Fusion Contrasting"
(Frontiers in Artificial Intelligence, 2024) -- the paper decision 004 cites.

A third pretext-task variant alongside pretrain.py (masked reconstruction)
and pretrain_tfc.py (the separate-encoders TF-C approach). This resolves the
specific open question left in chewing_ssl.ssl.models.augmentation's module docstring: how to
actually use `fuse_time_frequency`'s Kernel-PCA-fused embedding in a training
loop, since it's a flat (B, 2*n_components) vector, not a (C, T) sequence,
and so can't feed a sequence encoder directly.

Architecture:
  - Encoder: chewing_ssl.ssl.models.resnet1d.ResNet1D (3-layer 1D ResNet, per the
    paper), operating on the ORIGINAL (unaugmented) window -> pooled via
    ProjectionHead into z_orig.
  - Augmented view: time-domain augmentation + frequency-domain augmentation
    are each applied to the window, independently flattened, and Kernel-PCA
    transformed (chewing_ssl.ssl.models.augmentation.fuse_time_frequency,
    include_fusion=True), then concatenated -- exactly the paper's fusion
    step. This flat fused vector is projected (a small MLP, not the
    sequence-pooling ProjectionHead, since it's already flat) into z_fused.
  - Loss: NT-Xent(z_orig, z_fused) -- "this fused augmentation and original
    data constitute positive pairs in NT-Xent contrastive loss" (paper).

Kernel PCA has no learnable parameters -- it's refit via `.fit_transform` on
every batch (batch-level, transductive; see augmentation.py's module
docstring) -- so gradients only flow into the small fusion-projection MLP,
not "through" the KPCA step itself. That's expected: the ResNet encoder is
what actually learns from the contrastive signal.

At fine-tuning time the augmentation/KPCA path is discarded entirely (it
isn't even well-defined outside a batch), keeping only the ResNet encoder --
same convention as every other pretext task here (see
chewing_ssl.ssl.models.projection_head's docstring).

Fine-tuning granularity: like pretrain_tfc.py, the encoder's pooled output is
a window-level embedding (KPCA fusion flattens the whole window), so
downstream fine-tuning targets a single label per window, not per-timestep --
see chewing_ssl.ssl.tfc.finetune_window.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from chewing_ssl.common.data.preprocess import SENSOR_CHANNELS
from chewing_ssl.common.data.windowing import WindowDataset
from chewing_ssl.ssl.models.augmentation import make_augmented_views
from chewing_ssl.ssl.models.contrastive import nt_xent_loss
from chewing_ssl.ssl.models.projection_head import ProjectionHead
from chewing_ssl.ssl.models.resnet1d import ResNet1D
from chewing_ssl.common.tracking.mlflow_utils import log_metrics, log_params, start_run
from chewing_ssl.ssl.masked.pretrain import load_index, pick_device, shuffle_by_segment
from chewing_ssl.common.utils.config import load_augmentation_config
from chewing_ssl.common.utils.seed import set_seed

warnings.filterwarnings("ignore", message="An output with one or more elements was resized")

PROJECTION_OUTPUT_DIM = 128


class FusionProjectionHead(nn.Module):
    """Small MLP for the already-flat Kernel-PCA-fused vector -- unlike
    ProjectionHead, there's no (B, T, D) sequence here to mean-pool first."""

    def __init__(
        self, input_dim: int, output_dim: int = PROJECTION_OUTPUT_DIM, hidden_dim: int | None = None
    ):
        super().__init__()
        hidden_dim = hidden_dim or input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        return self.net(fused)


def run_epoch(
    encoder: torch.nn.Module,
    proj: torch.nn.Module,
    fusion_proj: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    augmentation_config: dict,
    temperature: float,
    optimizer: torch.optim.Optimizer | None,
    epoch_label: str = "",
) -> dict:
    is_train = optimizer is not None
    for m in (encoder, proj, fusion_proj):
        m.train(is_train)

    total_loss, n_batches = 0.0, 0
    desc = f"  [{'train' if is_train else 'val  '}] {epoch_label}"
    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        pbar = tqdm(loader, desc=desc, leave=False, unit="batch")
        for x in pbar:
            x = x.to(device)
            z_orig = proj(encoder.encode(x))

            views = make_augmented_views(x, augmentation_config, include_fusion=True)
            fused = torch.from_numpy(views.fused).to(device=device, dtype=z_orig.dtype)
            z_fused = fusion_proj(fused)

            loss = nt_xent_loss(z_orig, z_fused, temperature=temperature)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{total_loss / n_batches:.4f}")

    return {"loss": total_loss / max(n_batches, 1)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pretrain via the TF-FC (Kernel-PCA fusion) approach."
    )
    parser.add_argument(
        "--dataset", default=None, help="Restrict to one dataset. Default: all datasets."
    )
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--augmentation-config",
        type=Path,
        default=None,
        help="Defaults to configs/augmentation.yaml.",
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/pretrain_tffc"))
    parser.add_argument("--resnet-base-width", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=0.5, help="NT-Xent temperature.")
    parser.add_argument("--projection-dim", type=int, default=PROJECTION_OUTPUT_DIM)
    parser.add_argument("--max-train-windows", type=int, default=None)
    parser.add_argument("--max-val-windows", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    device = pick_device(args.device)
    augmentation_config = (
        load_augmentation_config(args.augmentation_config)
        if args.augmentation_config
        else load_augmentation_config()
    )
    n_components = augmentation_config["kernel_pca"]["n_components"]
    if args.batch_size <= n_components:
        raise ValueError(
            f"--batch-size ({args.batch_size}) must exceed kernel_pca.n_components ({n_components}): "
            "KernelPCA.fit_transform silently caps n_components at n_samples, so a batch this small "
            "would produce a smaller, wrong-shaped fused embedding instead of failing loudly."
        )
    fused_dim = 2 * n_components

    print(f"\n{'=' * 60}")
    print("  Pretext task   : TF-FC (Kernel-PCA fusion)")
    print(f"  Encoder        : resnet1d (base_width={args.resnet_base_width})")
    print(f"  Dataset        : {args.dataset or 'all'}")
    print(f"  Device         : {device}")
    print(f"  Epochs         : {args.epochs}  |  Batch size: {args.batch_size}  |  LR: {args.lr}")
    print(f"  Fused view dim : {fused_dim}  (2 * kernel_pca.n_components)")
    print(f"{'=' * 60}")

    train_index = load_index(
        args.processed_root, "pretrain_train", args.max_train_windows, args.seed, args.dataset
    )
    val_index = load_index(
        args.processed_root, "pretrain_val", args.max_val_windows, args.seed, args.dataset
    )
    print(
        f"  Train : {len(train_index):>7,} windows  |  {train_index['participant'].nunique()} participants"
    )
    print(
        f"  Val   : {len(val_index):>7,} windows  |  {val_index['participant'].nunique()} participants"
    )

    encoder = ResNet1D(n_channels=len(SENSOR_CHANNELS), base_width=args.resnet_base_width).to(
        device
    )
    proj = ProjectionHead(encoder.embedding_dim, output_dim=args.projection_dim).to(device)
    fusion_proj = FusionProjectionHead(fused_dim, output_dim=args.projection_dim).to(device)

    all_params = (
        list(encoder.parameters()) + list(proj.parameters()) + list(fusion_proj.parameters())
    )
    n_params = sum(p.numel() for p in all_params)
    print(f"  Params (encoder + both projection heads): {n_params:,}")

    optimizer = torch.optim.AdamW(all_params, lr=args.lr)

    val_loader = DataLoader(
        WindowDataset(
            val_index.sort_values("segment_path").reset_index(drop=True), with_label=False
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=True,  # same KernelPCA n_components constraint as the train loader
    )

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    history = []

    print()
    with start_run(run_name="pretrain_tffc_resnet1d"):
        log_params(
            {
                "pretext_task": "tffc",
                "architecture": "resnet1d",
                "resnet_base_width": args.resnet_base_width,
                "dataset": args.dataset or "all",
                "n_params": n_params,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "temperature": args.temperature,
                "projection_dim": args.projection_dim,
                "kernel_pca_n_components": augmentation_config["kernel_pca"]["n_components"],
                "kernel_pca_kernel": augmentation_config["kernel_pca"]["kernel"],
                "n_train_windows": len(train_index),
                "n_val_windows": len(val_index),
                "seed": args.seed,
                "device": str(device),
            }
        )

        for epoch in range(args.epochs):
            t0 = time.time()
            print(f"Epoch {epoch + 1}/{args.epochs}  [tffc / resnet1d]")

            shuffled = shuffle_by_segment(train_index, args.seed + epoch)
            train_loader = DataLoader(
                WindowDataset(shuffled, with_label=False),
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                drop_last=True,  # a smaller final batch shifts KernelPCA's n_components (see fuse_time_frequency)
            )

            epoch_label = f"epoch {epoch + 1}/{args.epochs}"
            train_m = run_epoch(
                encoder,
                proj,
                fusion_proj,
                train_loader,
                device,
                augmentation_config,
                args.temperature,
                optimizer,
                epoch_label,
            )
            val_m = run_epoch(
                encoder,
                proj,
                fusion_proj,
                val_loader,
                device,
                augmentation_config,
                args.temperature,
                None,
                epoch_label,
            )
            dt = time.time() - t0

            log_metrics({"train_loss": train_m["loss"], "val_loss": val_m["loss"]}, step=epoch)
            print(f"  train_loss={train_m['loss']:.4f}  val_loss={val_m['loss']:.4f}  ({dt:.1f}s)")

            history.append(
                {"epoch": epoch, "train_loss": train_m["loss"], "val_loss": val_m["loss"]}
            )

            if val_m["loss"] < best_val_loss:
                best_val_loss = val_m["loss"]
                ckpt_path = args.checkpoint_dir / "resnet1d_best.pt"
                torch.save(
                    {
                        "pretext_task": "tffc",
                        "architecture": "resnet1d",
                        "resnet_base_width": args.resnet_base_width,
                        "encoder_state_dict": encoder.state_dict(),
                        "proj_state_dict": proj.state_dict(),
                        "embedding_dim": encoder.embedding_dim,
                        "epoch": epoch,
                        "val_loss": best_val_loss,
                    },
                    ckpt_path,
                )
                print(f"  ✓ New best val_loss={best_val_loss:.4f}  →  {ckpt_path}")

        log_metrics({"best_val_loss": best_val_loss})
        history_path = args.checkpoint_dir / "resnet1d_history.json"
        history_path.write_text(json.dumps(history, indent=2))

    print(f"\n{'=' * 60}")
    print(f"  Done.  Best val_loss={best_val_loss:.4f}")
    print(f"  Checkpoint + history  →  {args.checkpoint_dir}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
