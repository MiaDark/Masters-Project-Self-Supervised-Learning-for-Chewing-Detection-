"""3-layer 1D ResNet encoder, for the TF-FC-style (Kernel-PCA fusion)
contrastive pretext task (decision 004 follow-up) -- per "use a 3 layer
resnet" in the Frontiers TF-FC paper (Time-series representation learning via
Time-Frequency Fusion Contrasting, 2024).

Encoder only, no decoder: like FrequencyEncoder, this only ever trains via a
contrastive loss against the Kernel-PCA-fused augmented view
(chewing_ssl.ssl.models.augmentation.fuse_time_frequency), so there's no
reconstruction target requiring a symmetric decoder.
"""

from __future__ import annotations

import torch
from torch import nn


class ResidualBlock1D(nn.Module):
    """Conv -> BN -> ReLU -> Conv -> BN, + a (possibly strided/projected)
    skip connection, then a final ReLU -- the standard ResNet basic block,
    adapted to 1D."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, kernel_size: int = 7):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=pad)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=pad)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.shortcut is None else self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class ResNet1D(nn.Module):
    """Input: (B, n_channels, T). Three residual stages, downsampling (stride 2)
    between stages -- the "3 layer resnet" from the TF-FC paper. encode()
    output: (B, 1, embedding_dim) -- global-average-pooled; see
    FrequencyEncoder's docstring for why output_size=1 (not some larger T')
    and why that specifically also avoids a PyTorch MPS-backend limitation
    (AdaptiveAvgPool1d requires the input size to evenly divide the output
    size on MPS only -- 50 timesteps into 8 bins doesn't divide evenly and
    crashes there, output_size=1 always does).
    """

    def __init__(self, n_channels: int = 5, base_width: int = 32, kernel_size: int = 7):
        super().__init__()
        widths = [base_width, base_width * 2, base_width * 4]
        self.embedding_dim = widths[-1]

        self.stem = nn.Sequential(
            nn.Conv1d(n_channels, widths[0], kernel_size=7, padding=3),
            nn.BatchNorm1d(widths[0]),
            nn.ReLU(inplace=True),
        )
        self.stage1 = ResidualBlock1D(widths[0], widths[0], stride=1, kernel_size=kernel_size)
        self.stage2 = ResidualBlock1D(widths[0], widths[1], stride=2, kernel_size=kernel_size)
        self.stage3 = ResidualBlock1D(widths[1], widths[2], stride=2, kernel_size=kernel_size)
        self.pool = nn.AdaptiveAvgPool1d(output_size=1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """(B, n_channels, T) -> (B, 1, embedding_dim)."""
        h = self.stem(x)
        h = self.stage1(h)
        h = self.stage2(h)
        h = self.stage3(h)
        h = self.pool(h)  # (B, embedding_dim, 1)
        return h.transpose(1, 2)  # (B, 1, embedding_dim)
