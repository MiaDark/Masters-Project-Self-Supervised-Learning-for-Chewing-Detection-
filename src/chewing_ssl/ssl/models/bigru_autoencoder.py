"""Bidirectional GRU masked autoencoder.

Recurrent inductive bias: explicit sequential order, one hidden state per
timestep (no resolution loss to manage, unlike pooling-based CNNs). No
forced single-vector bottleneck -- this is a contextual encoder with a
per-timestep linear reconstruction head, matching the masked-token-prediction
framing (BERT/MAE-style) rather than classic compress-then-decompress
autoencoding, so it's directly comparable to the other two architectures.
"""

from __future__ import annotations

import torch
from torch import nn


class BiGRUAutoencoder(nn.Module):
    """Input/output: (B, n_channels, T)."""

    def __init__(
        self, n_channels: int = 7, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.1
    ):
        super().__init__()
        self.embedding_dim = hidden_size * 2
        self.num_layers = num_layers
        self.mask_token = nn.Parameter(torch.zeros(n_channels))
        self.encoder = nn.GRU(
            input_size=n_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size * 2, n_channels)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """(B, n_channels, T) -> (B, T, embedding_dim), pre-reconstruction-head."""
        h, _ = self.encoder(x.transpose(1, 2))
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encode(x)  # (B, T, 2*hidden)
        recon = self.head(h)  # (B, T, n_channels)
        return recon.transpose(1, 2)  # (B, n_channels, T)

    def layer_parameters(self, layer_idx: int) -> list[nn.Parameter]:
        """Parameters belonging to one GRU layer (0 = closest to the input,
        num_layers-1 = closest to the output), for gradual unfreezing during
        fine-tuning -- nn.GRU names its per-layer weights/biases with an
        '_l{layer_idx}' suffix (and a further '_reverse' suffix for the
        backward direction, since this GRU is bidirectional; both directions
        of a layer are unfrozen together, since they're the same conceptual
        depth)."""
        suffix = f"_l{layer_idx}"
        return [
            p
            for name, p in self.encoder.named_parameters()
            if name.rsplit("_reverse", 1)[0].endswith(suffix)
        ]
