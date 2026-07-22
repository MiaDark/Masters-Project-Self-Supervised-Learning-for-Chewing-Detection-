"""SSL pretraining: TF-C (Time-Frequency Consistency) approach, per Zhang,
Zhao, Tsiligkaridis & Zitnik, "Self-Supervised Contrastive Pre-Training For
Time Series via Time-Frequency Consistency" (NeurIPS 2022).

A second pretext-task variant, distinct from pretrain.py (masked
reconstruction, decision 001). This is the literal TF-C architecture: two
SEPARATE encoders, three loss terms.

Architecture (paper's four components: "Contrastive time encoder, Contrastive
frequency encoder, and two cross-space projectors"):
  - Time encoder: one of the existing CNN/Bi-GRU architectures
    (--time-architecture), operating on the raw (B, C, T) window.
  - Frequency encoder: chewing_ssl.ssl.models.frequency_encoder.FrequencyEncoder,
    operating on the window's rFFT spectrum (B, C, F) -- a new architecture,
    since neither existing one accepts complex input.
  - Each encoder gets its own ProjectionHead (mean-pool + MLP) -- this
    doubles as both encoders' "cross-space projector": both heads map to the
    SAME output_dim, which is what makes the consistency loss meaningful.

Three loss terms per batch, matching the paper's total = weighted sum:
  1. L_time: NT-Xent between the original window's time-embedding and a
     time-domain-AUGMENTED view's time-embedding (both through the time
     encoder + its projection head).
  2. L_freq: NT-Xent between the original window's frequency-embedding and a
     frequency-domain-AUGMENTED view's frequency-embedding (both through the
     frequency encoder + its projection head).
  3. L_consistency: NT-Xent between the ORIGINAL window's time-embedding and
     its OWN frequency-embedding, treated as a positive pair (two views of
     the same underlying signal) with every other window in the batch as
     negatives -- this is what pulls the two encoders' outputs into a shared
     time-frequency space, per the paper's "consistency ... measured in the
     time-frequency space".
  loss = w_time * L_time + w_freq * L_freq + w_consistency * L_consistency

Fine-tuning granularity (important, see chewing_ssl.ssl.models.finetune_head):
the paper "concatenate[s] the time-domain and frequency-domain embeddings to
form the sample embedding" for downstream classification -- both source
embeddings are POOLED (mean over time-steps / frequency-bins respectively)
before concatenation, since a frequency encoder's output axis indexes
frequency bins, not time steps, so it cannot be aligned per-timestep with the
time encoder's output. This makes the resulting sample embedding
window-level, not per-timestep -- fine-tuning here targets a single
chewing/non-chewing label per window (majority vote over the window's dense
labels), not the per-timestep detection the masked-reconstruction path uses.
See chewing_ssl.ssl.tfc.finetune_window.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", message="An output with one or more elements was resized")


import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from chewing_ssl.common.data.preprocess import SENSOR_CHANNELS
from chewing_ssl.common.data.windowing import WindowDataset
from chewing_ssl.ssl.models import ARCHITECTURES
from chewing_ssl.ssl.models.augmentation import (
    apply_frequency_domain_augmentation,
    apply_time_domain_augmentation,
    to_frequency,
)
from chewing_ssl.ssl.models.contrastive import nt_xent_loss
from chewing_ssl.ssl.models.frequency_encoder import FrequencyEncoder
from chewing_ssl.ssl.models.projection_head import ProjectionHead
from chewing_ssl.common.tracking.mlflow_utils import log_metrics, log_params, start_run
from chewing_ssl.ssl.masked.pretrain import MODEL_KWARGS, load_index, pick_device, shuffle_by_segment
from chewing_ssl.common.utils.config import load_augmentation_config
from chewing_ssl.common.utils.seed import set_seed

PROJECTION_OUTPUT_DIM = 128


def run_epoch(
    time_encoder: torch.nn.Module,
    freq_encoder: torch.nn.Module,
    time_proj: torch.nn.Module,
    freq_proj: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    augmentation_config: dict,
    temperature: float,
    loss_weights: dict[str, float],
    optimizer: torch.optim.Optimizer | None,
    epoch_label: str = "",
) -> dict:
    is_train = optimizer is not None
    for m in (time_encoder, freq_encoder, time_proj, freq_proj):
        m.train(is_train)

    totals = {"loss": 0.0, "time": 0.0, "freq": 0.0, "consistency": 0.0}
    n_batches = 0
    desc = f"  [{'train' if is_train else 'val  '}] {epoch_label}"
    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        pbar = tqdm(loader, desc=desc, leave=False, unit="batch")
        for x in pbar:
            x = x.to(device)
            n_time = x.shape[-1]
            fs = augmentation_config["target_hz"]

            x_time_aug = apply_time_domain_augmentation(x, augmentation_config)
            x_freq = to_frequency(x)
            x_freq_aug = apply_frequency_domain_augmentation(x_freq, augmentation_config, n_time, fs)

            z_time_orig = time_proj(time_encoder.encode(x))
            z_time_aug = time_proj(time_encoder.encode(x_time_aug))
            z_freq_orig = freq_proj(freq_encoder.encode(x_freq))
            z_freq_aug = freq_proj(freq_encoder.encode(x_freq_aug))

            l_time = nt_xent_loss(z_time_orig, z_time_aug, temperature=temperature)
            l_freq = nt_xent_loss(z_freq_orig, z_freq_aug, temperature=temperature)
            l_consistency = nt_xent_loss(z_time_orig, z_freq_orig, temperature=temperature)

            loss = (
                loss_weights["time"] * l_time
                + loss_weights["freq"] * l_freq
                + loss_weights["consistency"] * l_consistency
            )

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            totals["loss"] += loss.item()
            totals["time"] += l_time.item()
            totals["freq"] += l_freq.item()
            totals["consistency"] += l_consistency.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{totals['loss'] / n_batches:.4f}")

    return {k: v / max(n_batches, 1) for k, v in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain via the TF-C (time-frequency consistency) approach.")
    parser.add_argument("--time-architecture", default="cnn", choices=list(ARCHITECTURES))
    parser.add_argument("--dataset", default=None, help="Restrict to one dataset. Default: all datasets.")
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--augmentation-config", type=Path, default=None, help="Defaults to configs/augmentation.yaml.")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/pretrain_tfc"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=0.5, help="NT-Xent temperature.")
    parser.add_argument("--projection-dim", type=int, default=PROJECTION_OUTPUT_DIM)
    parser.add_argument("--weight-time", type=float, default=1.0)
    parser.add_argument("--weight-freq", type=float, default=1.0)
    parser.add_argument("--weight-consistency", type=float, default=1.0)
    parser.add_argument("--max-train-windows", type=int, default=None)
    parser.add_argument("--max-val-windows", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    device = pick_device(args.device)
    augmentation_config = (
        load_augmentation_config(args.augmentation_config) if args.augmentation_config else load_augmentation_config()
    )
    loss_weights = {"time": args.weight_time, "freq": args.weight_freq, "consistency": args.weight_consistency}

    print(f"\n{'=' * 60}")
    print(f"  Pretext task : TF-C (time-frequency consistency)")
    print(f"  Time encoder : {args.time_architecture}  |  Freq encoder: frequency_encoder")
    print(f"  Dataset      : {args.dataset or 'all'}")
    print(f"  Device       : {device}")
    print(f"  Epochs       : {args.epochs}  |  Batch size: {args.batch_size}  |  LR: {args.lr}")
    print(f"  Loss weights : time={args.weight_time}  freq={args.weight_freq}  consistency={args.weight_consistency}")
    print(f"{'=' * 60}")

    train_index = load_index(args.processed_root, "pretrain_train", args.max_train_windows, args.seed, args.dataset)
    val_index = load_index(args.processed_root, "pretrain_val", args.max_val_windows, args.seed, args.dataset)
    print(f"  Train : {len(train_index):>7,} windows  |  {train_index['participant'].nunique()} participants")
    print(f"  Val   : {len(val_index):>7,} windows  |  {val_index['participant'].nunique()} participants")

    n_channels = len(SENSOR_CHANNELS)
    time_encoder = ARCHITECTURES[args.time_architecture](
        n_channels=n_channels, **MODEL_KWARGS[args.time_architecture]
    ).to(device)
    freq_encoder = FrequencyEncoder(n_channels=n_channels).to(device)
    time_proj = ProjectionHead(time_encoder.embedding_dim, output_dim=args.projection_dim).to(device)
    freq_proj = ProjectionHead(freq_encoder.embedding_dim, output_dim=args.projection_dim).to(device)

    all_params = (
        list(time_encoder.parameters())
        + list(freq_encoder.parameters())
        + list(time_proj.parameters())
        + list(freq_proj.parameters())
    )
    n_params = sum(p.numel() for p in all_params)
    print(f"  Params (both encoders + both projection heads): {n_params:,}")

    optimizer = torch.optim.AdamW(all_params, lr=args.lr)

    val_loader = DataLoader(
        WindowDataset(val_index.sort_values("segment_path").reset_index(drop=True), with_label=False),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
    )

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    history = []

    print()
    with start_run(run_name=f"pretrain_tfc_{args.time_architecture}"):
        log_params(
            {
                "pretext_task": "tfc",
                "time_architecture": args.time_architecture,
                "dataset": args.dataset or "all",
                "n_params": n_params,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "temperature": args.temperature,
                "projection_dim": args.projection_dim,
                "weight_time": args.weight_time,
                "weight_freq": args.weight_freq,
                "weight_consistency": args.weight_consistency,
                "n_train_windows": len(train_index),
                "n_val_windows": len(val_index),
                "seed": args.seed,
                "device": str(device),
            }
        )

        for epoch in range(args.epochs):
            t0 = time.time()
            print(f"Epoch {epoch + 1}/{args.epochs}  [tfc / {args.time_architecture}]")

            shuffled = shuffle_by_segment(train_index, args.seed + epoch)
            train_loader = DataLoader(
                WindowDataset(shuffled, with_label=False),
                batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
            )

            epoch_label = f"epoch {epoch + 1}/{args.epochs}"
            train_m = run_epoch(
                time_encoder, freq_encoder, time_proj, freq_proj, train_loader, device,
                augmentation_config, args.temperature, loss_weights, optimizer, epoch_label,
            )
            val_m = run_epoch(
                time_encoder, freq_encoder, time_proj, freq_proj, val_loader, device,
                augmentation_config, args.temperature, loss_weights, None, epoch_label,
            )
            dt = time.time() - t0

            log_metrics(
                {
                    "train_loss": train_m["loss"], "train_loss_time": train_m["time"],
                    "train_loss_freq": train_m["freq"], "train_loss_consistency": train_m["consistency"],
                    "val_loss": val_m["loss"], "val_loss_time": val_m["time"],
                    "val_loss_freq": val_m["freq"], "val_loss_consistency": val_m["consistency"],
                },
                step=epoch,
            )
            print(
                f"  train_loss={train_m['loss']:.4f} (t={train_m['time']:.3f} f={train_m['freq']:.3f} "
                f"c={train_m['consistency']:.3f})  val_loss={val_m['loss']:.4f}  ({dt:.1f}s)"
            )

            history.append({"epoch": epoch, "train": train_m, "val": val_m})

            if val_m["loss"] < best_val_loss:
                best_val_loss = val_m["loss"]
                ckpt_path = args.checkpoint_dir / f"{args.time_architecture}_best.pt"
                torch.save(
                    {
                        "pretext_task": "tfc",
                        "time_architecture": args.time_architecture,
                        "time_encoder_state_dict": time_encoder.state_dict(),
                        "freq_encoder_state_dict": freq_encoder.state_dict(),
                        "time_proj_state_dict": time_proj.state_dict(),
                        "freq_proj_state_dict": freq_proj.state_dict(),
                        "time_embedding_dim": time_encoder.embedding_dim,
                        "freq_embedding_dim": freq_encoder.embedding_dim,
                        "epoch": epoch,
                        "val_loss": best_val_loss,
                    },
                    ckpt_path,
                )
                print(f"  ✓ New best val_loss={best_val_loss:.4f}  →  {ckpt_path}")

        log_metrics({"best_val_loss": best_val_loss})
        history_path = args.checkpoint_dir / f"{args.time_architecture}_history.json"
        history_path.write_text(json.dumps(history, indent=2))

    print(f"\n{'=' * 60}")
    print(f"  Done.  Best val_loss={best_val_loss:.4f}")
    print(f"  Checkpoint + history  →  {args.checkpoint_dir}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
