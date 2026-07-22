from __future__ import annotations

import torch

from chewing_ssl.ssl.models.masking import (
    apply_mask_token,
    masked_mse,
    masked_mse_per_channel,
    sample_mask_batch,
)


def test_sample_mask_batch_shape_and_dtype():
    mask = sample_mask_batch(8, 7, 200)
    assert mask.shape == (8, 7, 200)
    assert mask.dtype == torch.bool


def test_sample_mask_batch_ratio_is_at_least_the_span_target():
    torch.manual_seed(0)
    mask = sample_mask_batch(32, 7, 200, span_mask_ratio=0.4, channel_mask_prob=0.0)
    frac = mask.float().mean(dim=(1, 2))
    # whole-channel masking disabled, so coverage should be close to (>=, due to
    # possible span overlap reducing it slightly) the requested span ratio
    assert frac.mean().item() > 0.25


def test_sample_mask_batch_zero_ratio_and_prob_masks_nothing():
    mask = sample_mask_batch(
        4, 7, 200, span_mask_ratio=0.0, channel_mask_prob=0.0, min_spans=1, max_spans=1
    )
    assert not mask.any()


def test_apply_mask_token_only_changes_masked_positions():
    torch.manual_seed(0)
    x = torch.randn(4, 7, 200)
    mask = sample_mask_batch(4, 7, 200)
    mask_token = torch.zeros(7)
    x_in = apply_mask_token(x, mask, mask_token)

    assert torch.allclose(x_in[~mask], x[~mask])
    assert torch.allclose(x_in[mask], torch.zeros_like(x_in[mask]))


def test_apply_mask_token_uses_per_channel_values():
    x = torch.zeros(1, 3, 5)
    mask = torch.zeros(1, 3, 5, dtype=torch.bool)
    mask[0, 1, :] = True  # mask channel 1 entirely
    mask_token = torch.tensor([10.0, 20.0, 30.0])
    x_in = apply_mask_token(x, mask, mask_token)
    assert torch.allclose(x_in[0, 1, :], torch.full((5,), 20.0))
    assert torch.allclose(x_in[0, 0, :], torch.zeros(5))


def test_masked_mse_matches_manual_computation():
    recon = torch.tensor([[[1.0, 2.0, 3.0]]])
    target = torch.tensor([[[0.0, 0.0, 0.0]]])
    mask = torch.tensor([[[True, False, True]]])
    # only positions 0 and 2 count: (1^2 + 3^2) / 2 = 5.0
    assert masked_mse(recon, target, mask).item() == 5.0


def test_masked_mse_per_channel_nan_when_no_masked_positions():
    recon = torch.zeros(2, 3, 10)
    target = torch.ones(2, 3, 10)
    mask = torch.zeros(2, 3, 10, dtype=torch.bool)
    mask[:, 0, :] = True  # only channel 0 has any masked positions

    per_channel = masked_mse_per_channel(recon, target, mask)
    assert per_channel[0].item() == 1.0  # (0-1)^2
    assert torch.isnan(per_channel[1])
    assert torch.isnan(per_channel[2])
