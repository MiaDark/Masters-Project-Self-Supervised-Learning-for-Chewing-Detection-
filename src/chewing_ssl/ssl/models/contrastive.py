"""NT-Xent (normalized temperature-scaled cross-entropy) contrastive loss,
for the TF-C-style pretext task (decision 004). Standard SimCLR formulation:
given two augmented views' projected embeddings, each item's positive is its
own other-view counterpart; every other item in the batch (both views) is a
negative.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    """z1, z2: (B, D) projected embeddings of the two augmented views of the
    same B windows (z1[i] and z2[i] are a positive pair). Returns a scalar."""
    b = z1.shape[0]
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    z = torch.cat([z1, z2], dim=0)  # (2B, D)

    sim = (z @ z.T) / temperature  # (2B, 2B) cosine similarity
    sim.fill_diagonal_(float("-inf"))  # exclude self-similarity

    # row i's positive is at i+B (i<B) or i-B (i>=B)
    targets = torch.cat([torch.arange(b, 2 * b), torch.arange(0, b)]).to(z.device)
    return F.cross_entropy(sim, targets)
