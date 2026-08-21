"""Shared masking policy for the masked-reconstruction pretext task (decision 001).

Used identically by both encoder architectures (CNN, Bi-GRU) so an
architecture comparison measures inductive bias, not differences in
pretext-task difficulty: the exact same masked input tensor is fed to
whichever model is being trained.

Two mechanisms, selectable independently via `mode` -- run as two SEPARATE
pretext tasks (not combined), per the decision to ablate them rather than
always OR them together:
  - `mode="channel"` -- whole-channel masking: each channel independently
    has a small chance of being masked for its ENTIRE duration. There is no
    real value left anywhere in that channel for this window, so the only
    way to reconstruct it is from the *other channels at the same time* --
    a pure cross-sensor/cross-modality inference task. Forces nothing about
    temporal dynamics (every unmasked channel is given perfect ground truth
    at every timestep).
  - `mode="span"` -- temporal span masking: 1-3 contiguous spans covering
    ~40% of timesteps, masked across ALL channels at once. Every sensor is
    blank during the gap, so the only way to reconstruct it is from the
    *same channels' own past/future* -- a pure cross-time inference task.
    Forces nothing about cross-channel redundancy.
  - `mode="both"` (legacy) -- the two masks OR'd together, as this module
    originally worked. Kept for backward compatibility / as a third,
    harder-combined option, but not the focus of the channel-vs-span
    ablation this module now primarily supports.

Masked positions are filled with a learnable mask token, not zero -- on
z-scored data, zero is the channel mean, i.e. a perfectly plausible real
value, so it carries no "this is missing" signal.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch

MaskMode = Literal["channel", "span", "both"]

# Span-length bounds (seconds), grounded in src/EDA/tables/summary_stats.csv --
# real within-meal inter-bite pauses run p5=0.43s to p75=5.04s (median 2.47s),
# real bites p25=6.4s to p75=15.1s. Default span bounds cover the pause scale
# (the shorter, sub-window-length phenomenon fixed-ratio masking couldn't
# represent at all) rather than the bite scale, since a span longer than the
# window itself is meaningless. See docs/preprocessing.md and src/EDA/.
DEFAULT_MIN_SPAN_S = 0.4
DEFAULT_MAX_SPAN_S = 5.0


def _sample_channel_mask(
    n_channels: int, channel_mask_prob: float, generator: torch.Generator | None
) -> torch.Tensor:
    """(C,) bool -- each channel independently masked w.p. channel_mask_prob."""
    return torch.rand(n_channels, generator=generator) < channel_mask_prob


def _sample_span_lengths(
    n_timesteps: int,
    span_mask_ratio: float,
    n_spans: int,
    fs: float,
    min_span_s: float,
    max_span_s: float,
    generator: torch.Generator | None,
) -> list[int]:
    """Sample `n_spans` VARIABLE span lengths (in samples): each drawn
    independently, log-uniformly, from [min_span_s, max_span_s] seconds --
    then rescaled (preserving relative proportions) so the total still
    matches span_mask_ratio * n_timesteps, the same overall masking BUDGET
    the old fixed-ratio scheme used, just distributed unevenly instead of
    split equally across spans.

    Log-uniform (not uniform) so short spans (sub-1s, chew-cycle scale) are
    sampled about as often as long ones (multi-second, pause scale) rather
    than the distribution being dominated by long spans -- real inter-bite
    pauses are themselves heavy-tailed (see src/EDA/plots/gap_duration_hist.png).
    """
    target_total = int(span_mask_ratio * n_timesteps)
    if target_total <= 0:
        return [0] * n_spans
    lo = max(min_span_s, 1.0 / fs)
    hi = max(max_span_s, lo * 1.01)
    log_lo, log_hi = float(np.log(lo)), float(np.log(hi))
    u = torch.rand(n_spans, generator=generator)
    raw_s = torch.exp(log_lo + u * (log_hi - log_lo))
    raw_samples = (raw_s * fs).clamp(min=1.0)
    scale = target_total / float(raw_samples.sum())
    lengths = (raw_samples * scale).round().clamp(min=1, max=n_timesteps).long()
    return lengths.tolist()


def _sample_span_mask(
    n_timesteps: int,
    span_mask_ratio: float,
    min_spans: int,
    max_spans: int,
    generator: torch.Generator | None,
    fs: float = 50.0,
    min_span_s: float = DEFAULT_MIN_SPAN_S,
    max_span_s: float = DEFAULT_MAX_SPAN_S,
    variable_length: bool = True,
) -> torch.Tensor:
    """(T,) bool -- 1-3 contiguous spans totaling ~span_mask_ratio of timesteps.

    variable_length=True (default): each span's own length is sampled from
    the real inter-bite-pause timescale this project measured (src/EDA/),
    not split evenly -- see `_sample_span_lengths`.
    variable_length=False reproduces the original deterministic equal-split
    behavior (kept only to exactly reproduce pre-existing ablation
    checkpoints; not recommended for new pretraining runs).
    """
    n_spans = int(torch.randint(min_spans, max_spans + 1, (1,), generator=generator).item())
    time_mask = torch.zeros(n_timesteps, dtype=torch.bool)

    if variable_length:
        span_lengths = _sample_span_lengths(
            n_timesteps, span_mask_ratio, n_spans, fs, min_span_s, max_span_s, generator
        )
    else:
        target_masked = int(span_mask_ratio * n_timesteps)
        remaining = target_masked
        span_lengths = []
        for s in range(n_spans):
            span_len = remaining // (n_spans - s) if s < n_spans - 1 else remaining
            span_lengths.append(max(0, min(span_len, n_timesteps)))
            remaining -= span_lengths[-1]

    for span_len in span_lengths:
        if span_len <= 0:
            continue
        hi = max(1, n_timesteps - span_len + 1)
        start = int(torch.randint(0, hi, (1,), generator=generator).item())
        time_mask[start : start + span_len] = True
    return time_mask


def sample_mask_batch(
    batch_size: int,
    n_channels: int,
    n_timesteps: int,
    mode: MaskMode = "both",
    span_mask_ratio: float = 0.4,
    channel_mask_prob: float = 0.15,
    min_spans: int = 1,
    max_spans: int = 3,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
    fs: float = 50.0,
    min_span_s: float = DEFAULT_MIN_SPAN_S,
    max_span_s: float = DEFAULT_MAX_SPAN_S,
    variable_length: bool = True,
) -> torch.Tensor:
    """Sample a boolean mask, shape (B, C, T), True = masked.

    mode="channel": whole-channel masking only.
    mode="span": temporal span masking only.
    mode="both": both, OR'd together (legacy combined behavior).

    variable_length/fs/min_span_s/max_span_s: see `_sample_span_mask` --
    only relevant when mode is "span" or "both".
    """
    if mode not in ("channel", "span", "both"):
        raise ValueError(f"Unknown mask mode {mode!r} -- expected 'channel', 'span', or 'both'.")

    masks = torch.zeros(batch_size, n_channels, n_timesteps, dtype=torch.bool, device=device)

    for b in range(batch_size):
        channel_mask = (
            _sample_channel_mask(n_channels, channel_mask_prob, generator)
            if mode in ("channel", "both")
            else torch.zeros(n_channels, dtype=torch.bool)
        )
        time_mask = (
            _sample_span_mask(
                n_timesteps,
                span_mask_ratio,
                min_spans,
                max_spans,
                generator,
                fs=fs,
                min_span_s=min_span_s,
                max_span_s=max_span_s,
                variable_length=variable_length,
            )
            if mode in ("span", "both")
            else torch.zeros(n_timesteps, dtype=torch.bool)
        )
        masks[b] = channel_mask.unsqueeze(1).to(device) | time_mask.unsqueeze(0).to(device)

    return masks


def apply_mask_token(x: torch.Tensor, mask: torch.Tensor, mask_token: torch.Tensor) -> torch.Tensor:
    """Replace masked (B, C, T) positions in x with the learnable per-channel mask_token."""
    token = mask_token.view(1, -1, 1).expand_as(x)
    return torch.where(mask, token, x)


def masked_mse(
    reconstruction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """MSE computed only over masked positions -- otherwise the model could trivially
    copy unmasked input through and the loss would mostly reflect that, not how well
    it fills in the masked spans."""
    diff2 = (reconstruction - target) ** 2
    return diff2[mask].mean()


def masked_mse_per_channel(
    reconstruction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Per-channel masked MSE, shape (C,). NaN for a channel with no masked positions in this batch."""
    diff2 = (reconstruction - target) ** 2
    n_channels = reconstruction.shape[1]
    out = torch.full((n_channels,), float("nan"), device=reconstruction.device)
    for c in range(n_channels):
        m = mask[:, c, :]
        if m.any():
            out[c] = diff2[:, c, :][m].mean()
    return out
