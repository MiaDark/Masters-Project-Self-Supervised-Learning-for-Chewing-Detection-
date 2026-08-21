"""Fine-tuning classification head for the TF-C / TF-FC contrastive pretext
tasks (decision 004 follow-up).

Deliberately shallow and narrow: the pretrained encoder should already carry
most of the representational load (that's the point of pretraining), so this
head only needs to be an easy-to-fit classifier on top of a good embedding,
not a second model doing its own heavy lifting.

Operates on a single pooled *window*-level embedding (B, input_dim), not a
per-timestep sequence -- see chewing_ssl.ssl.tfc.pretrain_tfc / pretrain_tffc
module docstrings for why: both source embeddings (TF-C's concatenated
time+frequency vectors, TF-FC's Kernel-PCA-fused/ResNet vector) are pooled by
construction, so there is no per-timestep resolution left to classify at this
stage (unlike the masked-reconstruction encoders' finetune.py path, which
does stay per-timestep).
"""

from __future__ import annotations

import torch
from torch import nn


class WindowClassifierHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, input_dim) pooled embedding -> (B,) chewing logits."""
        return self.net(z).squeeze(-1)
