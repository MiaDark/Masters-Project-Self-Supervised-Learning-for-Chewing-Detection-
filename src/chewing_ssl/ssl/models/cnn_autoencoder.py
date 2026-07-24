"""1D CNN U-Net-style masked autoencoder.

Convolutional inductive bias: locality and translation-invariance, a good
match for the ~1.5-2 Hz chewing rhythm found in the PSD analysis (a local,
repeating pattern). Downsample/upsample with skip connections so the decoder
gets both a compressed bottleneck and full-resolution detail to reconstruct
from -- output stays at the input's full temporal resolution, required for
dense per-timestep fine-tuning later.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _num_groups(channels: int, max_groups: int = 8) -> int:
    """Largest divisor of channels that is <= max_groups (GroupNorm requires an exact divisor)."""
    for g in range(min(max_groups, channels), 0, -1):
        if channels % g == 0:
            return g
    return 1


class ConvBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int = 7, dropout: float = 0.0
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(num_groups=_num_groups(out_channels), num_channels=out_channels),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CNNAutoencoder(nn.Module):
    """Input/output: (B, n_channels, T), any T -- not just multiples of 2**depth.

    T=200 (4s @ 50Hz) divides evenly by 2**3=8, but T=100 (the 2s/25%-overlap
    variant) does not (100/8=12.5) -- naively downsampling would misalign the
    decoder's upsampled length against the original input. Fixed by
    replicate-padding up to the next multiple of 2**depth before the down
    path, and cropping the decoder output back to the original T afterward --
    a standard U-Net trick for arbitrary input lengths, needed so this
    architecture can train on both window sizes without a depth-per-window-
    size special case.
    """

    def __init__(
        self,
        n_channels: int = 7,
        base_width: int = 32,
        depth: int = 3,
        kernel_size: int = 7,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.mask_token = nn.Parameter(torch.zeros(n_channels))
        self.depth = depth

        widths = [base_width * (2**i) for i in range(depth)]
        self.embedding_dim = widths[-1] * 2

        self.down_blocks = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        in_ch = n_channels
        for w in widths:
            self.down_blocks.append(ConvBlock(in_ch, w, kernel_size, dropout))
            self.downsamplers.append(nn.Conv1d(w, w, kernel_size=2, stride=2))
            in_ch = w

        self.bottleneck = ConvBlock(widths[-1], widths[-1] * 2, kernel_size, dropout)

        self.upsamplers = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        up_in = widths[-1] * 2
        for w in reversed(widths):
            self.upsamplers.append(nn.ConvTranspose1d(up_in, w, kernel_size=2, stride=2))
            self.up_blocks.append(ConvBlock(w * 2, w, kernel_size, dropout))  # *2 for concatenated skip
            up_in = w

        self.head = nn.Conv1d(widths[0], n_channels, kernel_size=1)

    def _pad_to_multiple(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        multiple = 2**self.depth
        t = x.shape[-1]
        pad = (multiple - t % multiple) % multiple
        if pad:
            x = F.pad(x, (0, pad), mode="replicate")
        return x, pad

    def _down(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        skips = []
        h = x
        for block, down in zip(self.down_blocks, self.downsamplers):
            h = block(h)
            skips.append(h)
            h = down(h)
        return self.bottleneck(h), skips

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """(B, n_channels, T) -> (B, T', embedding_dim), the pre-decoder bottleneck
        (T' = ceil(T / 2**depth)), transposed to channels-last so it matches
        the other two architectures' encode() output convention. Callers that
        need per-timestep resolution (e.g. a detection head) must upsample T'
        back to T themselves -- see ssl/masked/finetune.py's ChewingDetector."""
        x, _ = self._pad_to_multiple(x)
        h, _ = self._down(x)
        return h.transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t_orig = x.shape[-1]
        x, pad = self._pad_to_multiple(x)
        h, skips = self._down(x)

        for up, block, skip in zip(self.upsamplers, self.up_blocks, reversed(skips)):
            h = up(h)
            h = torch.cat([h, skip], dim=1)
            h = block(h)

        out = self.head(h)
        return out[..., :t_orig] if pad else out
