"""Frequency-domain encoder for the TF-C-style dual-encoder pretext task
(decision 004 follow-up). Neither existing encoder (CNN, Bi-GRU) accepts
complex-valued input, so this is a new, dedicated architecture rather than
a reuse of CNNAutoencoder.

Input is the rFFT spectrum of a window (chewing_ssl.ssl.models.augmentation.
to_frequency), complex-valued, shape (B, C, F) with F = window_len // 2 + 1.
Real and imaginary parts are stacked along the channel axis (2C real-valued
channels in) rather than using magnitude/phase, matching the convention
chewing_ssl.ssl.models.augmentation.fuse_time_frequency already uses when it
flattens a frequency view for Kernel PCA.

"""

from __future__ import annotations

import torch
import torch.nn as nn

from chewing_ssl.ssl.models.cnn_autoencoder import ConvBlock


class FrequencyEncoder(nn.Module):
    """Input: (B, n_channels, F) complex. encode() output: (B, 1, embedding_dim)
    -- global-average-pooled, since ProjectionHead only ever mean-pools this
    output anyway (there's no genuine per-bin resolution used downstream, unlike
    the other encoders' real per-timestep (B, T', D) outputs). Also sidesteps a
    PyTorch MPS-backend limitation: AdaptiveAvgPool1d requires the input size to
    evenly divide the output size on MPS (not CPU/CUDA) -- output_size=1 always
    divides evenly, so this is robust regardless of window length / F.
    """

    def __init__(
        self, n_channels: int = 5, base_width: int = 32, depth: int = 3, kernel_size: int = 7
    ):
        super().__init__()
        widths = [base_width * (2**i) for i in range(depth)]
        self.embedding_dim = widths[-1]

        blocks = []
        in_ch = n_channels * 2  # real + imag stacked
        for w in widths:
            blocks.append(ConvBlock(in_ch, w, kernel_size))
            in_ch = w
        self.blocks = nn.ModuleList(blocks)
        self.pool = nn.AdaptiveAvgPool1d(output_size=1)

    def encode(self, x_freq: torch.Tensor) -> torch.Tensor:
        """(B, n_channels, F) complex -> (B, 1, embedding_dim)."""
        h = torch.cat([x_freq.real, x_freq.imag], dim=1)  # (B, 2*n_channels, F)
        for block in self.blocks:
            h = block(h)
        h = self.pool(h)  # (B, embedding_dim, 1)
        return h.transpose(1, 2)  # (B, 1, embedding_dim)
