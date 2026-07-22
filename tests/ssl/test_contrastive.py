from __future__ import annotations

import torch

from chewing_ssl.ssl.models.contrastive import nt_xent_loss
from chewing_ssl.ssl.models.projection_head import ProjectionHead


def test_projection_head_output_shape():
    head = ProjectionHead(input_dim=32, output_dim=16)
    h = torch.randn(4, 10, 32)  # (B, T', input_dim)
    out = head(h)
    assert out.shape == (4, 16)


def test_projection_head_is_differentiable():
    head = ProjectionHead(input_dim=32, output_dim=16)
    h = torch.randn(4, 10, 32, requires_grad=True)
    out = head(h)
    out.sum().backward()
    assert h.grad is not None
    assert all(p.grad is not None for p in head.parameters())


def test_nt_xent_loss_is_scalar_and_finite():
    z1 = torch.randn(8, 16)
    z2 = torch.randn(8, 16)
    loss = nt_xent_loss(z1, z2)
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_nt_xent_loss_lower_when_positives_match_better():
    torch.manual_seed(0)
    z1 = torch.randn(16, 8)
    z_matching = z1.clone()  # perfect positive alignment
    z_random = torch.randn(16, 8)  # unrelated "positives"

    loss_matching = nt_xent_loss(z1, z_matching, temperature=0.5)
    loss_random = nt_xent_loss(z1, z_random, temperature=0.5)
    assert loss_matching < loss_random


def test_nt_xent_loss_is_differentiable():
    z1 = torch.randn(8, 16, requires_grad=True)
    z2 = torch.randn(8, 16, requires_grad=True)
    loss = nt_xent_loss(z1, z2)
    loss.backward()
    assert z1.grad is not None
    assert z2.grad is not None


def test_nt_xent_loss_batch_size_one_is_degenerate_but_runs():
    # No negatives available (2B=2, only the positive pair off-diagonal exists),
    # but the function shouldn't crash.
    z1 = torch.randn(1, 8)
    z2 = torch.randn(1, 8)
    loss = nt_xent_loss(z1, z2)
    assert torch.isfinite(loss)
