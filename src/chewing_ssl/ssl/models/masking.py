"""Shared masking policy for the masked-reconstruction pretext task (decision 001).

Used identically by both encoder architectures (CNN, Bi-GRU) so an
architecture comparison measures inductive bias, not differences in
pretext-task difficulty: the exact same masked input tensor is fed to
whichever model is being trained.

Two masking modes, combined per window (decision 001: "hide random temporal
segments or entire sensor modalities"):
  - Whole-channel masking: each channel independently has a small chance of
    being masked for its entire duration -- forces the model to reconstruct
    it from the *other* channels (a warm-up toward cross-sensor prediction).
  - Temporal span masking: 1-3 contiguous spans covering ~40% of timesteps,
    masked across all channels at once -- forces the model to use *temporal*
    context (what came before/after) to fill the gap.

Masked positions are filled with a learnable mask token, not zero -- on
z-scored data, zero is the channel mean, i.e. a perfectly plausible real
value, so it carries no "this is missing" signal.
"""

from __future__ import annotations

import torch


def sample_mask_batch(
    batch_size: int,
    n_channels: int,
    n_timesteps: int,
    span_mask_ratio: float = 0.4,
    channel_mask_prob: float = 0.15,
    min_spans: int = 1,
    max_spans: int = 3,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample a boolean mask, shape (B, C, T), True = masked."""
    masks = torch.zeros(batch_size, n_channels, n_timesteps, dtype=torch.bool, device=device)
    target_masked = int(span_mask_ratio * n_timesteps)

    for b in range(batch_size):
        channel_mask = torch.rand(n_channels, generator=generator) < channel_mask_prob

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
