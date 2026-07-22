"""SSL pretraining: masked reconstruction, comparing two encoder architectures.

Trains one architecture at a time (cnn / bigru) on the `pretrain_train`
window purpose, validates on `pretrain_val`, and logs everything to MLflow
(local file store -- see chewing_ssl.common.tracking.mlflow_utils).

Held fixed across both architectures, so the comparison measures
inductive bias, not confounds:
  - normalization (the single global variant -- see docs/preprocessing.md)
  - masking policy (chewing_ssl.ssl.models.masking)
  - parameter budget (~100-120K params each, tuned via each model's
    constructor kwargs below)
  - optimizer, learning rate, batch size, number of training windows seen
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from chewing_ssl.common.data.preprocess import SENSOR_CHANNELS
from chewing_ssl.common.data.windowing import WindowDataset
from chewing_ssl.ssl.models import ARCHITECTURES
from chewing_ssl.ssl.models.masking import (
    apply_mask_token,
    masked_mse,
    masked_mse_per_channel,
    sample_mask_batch,
)
from chewing_ssl.common.tracking.mlflow_utils import log_metrics, log_params, start_run
from chewing_ssl.common.utils.seed import set_seed

MODEL_KWARGS = {
    "cnn": {"base_width": 12},
    "bigru": {},
}


def shuffle_by_segment(index: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Shuffle windows at the segment level to keep LRU cache hit rate near 100%.

    A fully-randomised DataLoader reads a new parquet file for almost every
    window (cache miss on every item).  Shuffling at the segment level keeps
    windows from the same segment consecutive so each segment is read once
    per epoch rather than once per window.
    """
    rng = np.random.default_rng(seed)
    segs = index["segment_path"].unique().copy()
    rng.shuffle(segs)
    order = {s: i for i, s in enumerate(segs)}
    return (
        index.assign(_seg_order=index["segment_path"].map(order))
        .sort_values("_seg_order")
        .drop(columns="_seg_order")
        .reset_index(drop=True)
    )


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
    purpose: str,
    max_windows: int | None,
    seed: int,
    dataset: str | None = None,
) -> pd.DataFrame:
    index = pd.read_parquet(processed_root / "index" / f"{purpose}.parquet")
    if dataset:
        index = index[index["dataset"] == dataset].reset_index(drop=True)
    if max_windows is not None and len(index) > max_windows:
        index = index.sample(n=max_windows, random_state=seed).reset_index(drop=True)
    return index


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    mask_kwargs: dict,
    optimizer: torch.optim.Optimizer | None,
    epoch_label: str = "",
) -> dict:
    """One pass over loader.  If optimizer is None, runs in eval / no-grad mode."""
    is_train = optimizer is not None
    model.train(is_train)

    total_loss, n_batches = 0.0, 0
    per_channel_sum = torch.zeros(len(SENSOR_CHANNELS))
    per_channel_count = torch.zeros(len(SENSOR_CHANNELS))

    desc = f"  [{'train' if is_train else 'val  '}] {epoch_label}"
    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        pbar = tqdm(loader, desc=desc, leave=False, unit="batch")
        for x in pbar:
            x = x.to(device)
            b, c, t = x.shape
            mask = sample_mask_batch(b, c, t, device=device, **mask_kwargs)
            x_in = apply_mask_token(x, mask, model.mask_token)

            recon = model(x_in)
            loss = masked_mse(recon, x, mask)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            with torch.no_grad():
                per_ch = masked_mse_per_channel(recon, x, mask).cpu()
                valid = ~torch.isnan(per_ch)
                per_channel_sum[valid] += per_ch[valid]
                per_channel_count[valid] += 1

            pbar.set_postfix(loss=f"{total_loss / n_batches:.4f}")

    per_channel_avg = (per_channel_sum / per_channel_count.clamp(min=1)).tolist()
    return {
        "loss": total_loss / max(n_batches, 1),
        "per_channel": dict(zip(SENSOR_CHANNELS, per_channel_avg)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pretrain one encoder architecture via masked reconstruction."
    )
    parser.add_argument("--architecture", required=True, choices=list(ARCHITECTURES))
    parser.add_argument(
        "--dataset",
        default=None,
        help="Restrict to one dataset (e.g. SIPMEatingStudy). Default: all datasets.",
    )
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/pretrain"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--max-train-windows",
        type=int,
        default=None,
        help="Cap training windows (default: use all available)",
    )
    parser.add_argument("--max-val-windows", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--mask-span-ratio", type=float, default=0.4)
    parser.add_argument("--mask-channel-prob", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    device = pick_device(args.device)

    print(f"\n{'=' * 60}")
    print(f"  Architecture : {args.architecture}")
    print(f"  Dataset      : {args.dataset or 'all'}")
    print(f"  Device       : {device}")
    print(f"  Epochs       : {args.epochs}  |  Batch size: {args.batch_size}  |  LR: {args.lr}")
    print(f"{'=' * 60}")

    train_index = load_index(
        args.processed_root,
        "pretrain_train",
        args.max_train_windows,
        args.seed,
        args.dataset,
    )
    val_index = load_index(
        args.processed_root,
        "pretrain_val",
        args.max_val_windows,
        args.seed,
        args.dataset,
    )

    n_train_segs = train_index["segment_path"].nunique()
    n_val_segs = val_index["segment_path"].nunique()
    print(
        f"  Train : {len(train_index):>7,} windows  |  {n_train_segs} segments  |  "
        f"{train_index['participant'].nunique()} participants"
    )
    print(
        f"  Val   : {len(val_index):>7,} windows  |  {n_val_segs} segments  |  "
        f"{val_index['participant'].nunique()} participants"
    )

    model = ARCHITECTURES[args.architecture](
        n_channels=len(SENSOR_CHANNELS), **MODEL_KWARGS[args.architecture]
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    mask_kwargs = {
        "span_mask_ratio": args.mask_span_ratio,
        "channel_mask_prob": args.mask_channel_prob,
    }

    # Val loader: sorted by segment so the LRU cache never thrashes.
    val_loader = DataLoader(
        WindowDataset(
            val_index.sort_values("segment_path").reset_index(drop=True), with_label=False
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    history = []

    print()
    with start_run(run_name=f"pretrain_{args.architecture}"):
        log_params(
            {
                "architecture": args.architecture,
                "dataset": args.dataset or "all",
                "n_params": n_params,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "n_train_windows": len(train_index),
                "n_val_windows": len(val_index),
                "mask_span_ratio": args.mask_span_ratio,
                "mask_channel_prob": args.mask_channel_prob,
                "seed": args.seed,
                "device": str(device),
                **{f"model_kwarg_{k}": v for k, v in MODEL_KWARGS[args.architecture].items()},
            }
        )

        for epoch in range(args.epochs):
            t0 = time.time()
            print(f"Epoch {epoch + 1}/{args.epochs}  [{args.architecture}]")

            # Rebuild loader each epoch with segment-level shuffle (avoids I/O bottleneck).
            shuffled = shuffle_by_segment(train_index, args.seed + epoch)
            train_loader = DataLoader(
                WindowDataset(shuffled, with_label=False),
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
            )

            epoch_label = f"epoch {epoch + 1}/{args.epochs}"
            train_m = run_epoch(model, train_loader, device, mask_kwargs, optimizer, epoch_label)
            val_m = run_epoch(
                model, val_loader, device, mask_kwargs, optimizer=None, epoch_label=epoch_label
            )
            dt = time.time() - t0

            log_metrics({"train_loss": train_m["loss"], "val_loss": val_m["loss"]}, step=epoch)
            log_metrics({f"val_loss_{c}": v for c, v in val_m["per_channel"].items()}, step=epoch)

            # Compact per-channel summary (first 3 letters of each channel name)
            ch_summary = "  ".join(
                f"{c.split('_')[0][:3]}={v:.3f}" for c, v in val_m["per_channel"].items()
            )
            print(
                f"  train_loss={train_m['loss']:.4f}  val_loss={val_m['loss']:.4f}  ({dt:.1f}s)\n"
                f"  per-channel val: {ch_summary}"
            )

            history.append(
                {
                    "epoch": epoch,
                    **train_m,
                    "val_loss": val_m["loss"],
                    "val_per_channel": val_m["per_channel"],
                }
            )

            if val_m["loss"] < best_val_loss:
                best_val_loss = val_m["loss"]
                ckpt_path = args.checkpoint_dir / f"{args.architecture}_best.pt"
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "epoch": epoch,
                        "val_loss": best_val_loss,
                    },
                    ckpt_path,
                )
                print(f"  ✓ New best val_loss={best_val_loss:.4f}  →  {ckpt_path}")

        log_metrics({"best_val_loss": best_val_loss})
        history_path = args.checkpoint_dir / f"{args.architecture}_history.json"
        history_path.write_text(json.dumps(history, indent=2))

    print(f"\n{'=' * 60}")
    print(f"  Done.  Best val_loss={best_val_loss:.4f}")
    print(f"  Checkpoint + history  →  {args.checkpoint_dir}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
