"""Projection head for the contrastive pretext task (decision 004).

Standard SimCLR-style setup: mean-pool an encoder's (B, T', D) sequence
output to a single vector per window, then a small MLP into the space the
contrastive loss actually operates in. Discarded after pretraining -- only
the encoder (model.encode()) is kept for fine-tuning, same convention as
SimCLR/TF-C.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 128, hidden_dim: int | None = None):
        super().__init__()
        hidden_dim = hidden_dim or input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (B, T', input_dim) encoder output -> (B, output_dim)."""
        pooled = h.mean(dim=1)
        return self.net(pooled)
