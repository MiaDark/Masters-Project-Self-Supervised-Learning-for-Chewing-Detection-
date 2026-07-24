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

import torch

MaskMode = Literal["channel", "span", "both"]


def _sample_channel_mask(
    n_channels: int, channel_mask_prob: float, generator: torch.Generator | None
) -> torch.Tensor:
    """(C,) bool -- each channel independently masked w.p. channel_mask_prob."""
    return torch.rand(n_channels, generator=generator) < channel_mask_prob


def _sample_span_mask(
    n_timesteps: int,
    span_mask_ratio: float,
    min_spans: int,
    max_spans: int,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """(T,) bool -- 1-3 contiguous spans totaling ~span_mask_ratio of timesteps."""
    target_masked = int(span_mask_ratio * n_timesteps)
    n_spans = int(torch.randint(min_spans, max_spans + 1, (1,), generator=generator).item())
    time_mask = torch.zeros(n_timesteps, dtype=torch.bool)
    remaining = target_masked
    for s in range(n_spans):
        span_len = remaining // (n_spans - s) if s < n_spans - 1 else remaining
        span_len = max(0, min(span_len, n_timesteps))
        if span_len == 0:
            continue
        hi = max(1, n_timesteps - span_len + 1)
        start = int(torch.randint(0, hi, (1,), generator=generator).item())
        time_mask[start : start + span_len] = True
        remaining -= span_len
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
) -> torch.Tensor:
    """Sample a boolean mask, shape (B, C, T), True = masked.

    mode="channel": whole-channel masking only.
    mode="span": temporal span masking only.
    mode="both": both, OR'd together (legacy combined behavior).
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
            _sample_span_mask(n_timesteps, span_mask_ratio, min_spans, max_spans, generator)
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
