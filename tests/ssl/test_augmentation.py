from __future__ import annotations

import numpy as np
import torch

from chewing_ssl.ssl.models.augmentation import (
    add_frequency,
    apply_frequency_domain_augmentation,
    apply_time_domain_augmentation,
    from_frequency,
    fuse_time_frequency,
    jitter,
    low_pass_freq,
    make_augmented_views,
    masking,
    permutation,
    phase_shift,
    remove_frequency,
    scaling,
    to_frequency,
)

B, C, T = 4, 5, 200


def _config(**overrides) -> dict:
    cfg = {
        "window_len": T,
        "target_hz": 50.0,
        "apply_probability": 0.25,
        "time_domain": {
            "jitter": {"enabled": True, "sigma": 0.01},
            "scaling": {"enabled": True, "mean": 1.0, "sigma": 0.1},
            "permutation": {"enabled": True, "n_segments": 8},
            "masking": {"enabled": True, "mask_ratio": 0.1, "mask_value": 0.0},
        },
        "frequency_domain": {
            "low_pass": {"enabled": True, "cutoff_hz": 4.0},
            "phase_shift": {"enabled": True, "max_shift_rad": 3.14159265},
            "remove_frequency": {"enabled": True, "ratio": 0.1},
            "add_frequency": {"enabled": True, "ratio": 0.1, "add_scale": 0.1},
        },
        "kernel_pca": {"kernel": "rbf", "n_components": 3, "gamma": None},
    }
    for k, v in overrides.items():
        cfg[k] = v
    return cfg


# ---- time domain ------------------------------------------------------------


def test_jitter_zero_sigma_is_identity():
    x = torch.randn(B, C, T)
    assert torch.allclose(jitter(x, sigma=0.0), x)


def test_jitter_nonzero_sigma_changes_values():
    x = torch.zeros(B, C, T)
    out = jitter(x, sigma=1.0, generator=torch.Generator().manual_seed(0))
    assert not torch.allclose(out, x)
    assert out.shape == x.shape


def test_scaling_factor_one_is_identity():
    x = torch.randn(B, C, T)
    out = scaling(x, mean=1.0, sigma=0.0)
    assert torch.allclose(out, x)


def test_scaling_preserves_shape_and_changes_values():
    x = torch.randn(B, C, T) + 5.0  # nonzero so scaling has visible effect
    out = scaling(x, mean=1.0, sigma=0.5, generator=torch.Generator().manual_seed(0))
    assert out.shape == x.shape
    assert not torch.allclose(out, x)


def test_permutation_preserves_per_channel_value_set():
    x = torch.arange(B * C * T, dtype=torch.float32).reshape(B, C, T)
    out = permutation(x, n_segments=8, generator=torch.Generator().manual_seed(0))
    assert out.shape == x.shape
    for b in range(B):
        for c in range(C):
            assert torch.equal(torch.sort(out[b, c]).values, torch.sort(x[b, c]).values)


def test_permutation_single_segment_is_identity():
    x = torch.randn(B, C, T)
    out = permutation(x, n_segments=1)
    assert torch.equal(out, x)


def test_masking_zeroes_exactly_mask_ratio_span():
    x = torch.ones(B, C, T)
    out = masking(x, mask_ratio=0.1, mask_value=0.0, generator=torch.Generator().manual_seed(0))
    expected_len = round(0.1 * T)
    for b in range(B):
        n_masked = (out[b, 0] == 0.0).sum().item()
        assert n_masked == expected_len
        # masked across every channel identically
        assert torch.equal(out[b, 0] == 0.0, out[b, -1] == 0.0)


def test_masking_zero_ratio_is_identity():
    x = torch.randn(B, C, T)
    out = masking(x, mask_ratio=0.0, mask_value=0.0)
    assert torch.equal(out, x)


def test_apply_time_domain_augmentation_zero_probability_is_identity():
    x = torch.randn(B, C, T)
    out = apply_time_domain_augmentation(x, _config(apply_probability=0.0))
    assert torch.equal(out, x)


def test_apply_time_domain_augmentation_shape_preserved():
    x = torch.randn(B, C, T)
    out = apply_time_domain_augmentation(
        x, _config(apply_probability=1.0), generator=torch.Generator().manual_seed(1)
    )
    assert out.shape == x.shape


# ---- frequency domain --------------------------------------------------------


def test_to_from_frequency_roundtrips():
    x = torch.randn(B, C, T)
    x_freq = to_frequency(x)
    recovered = from_frequency(x_freq, n=T)
    assert torch.allclose(recovered, x, atol=1e-5)


def test_low_pass_freq_zeroes_bins_above_cutoff():
    x = torch.randn(B, C, T)
    x_freq = to_frequency(x)
    out = low_pass_freq(x_freq, cutoff_hz=4.0, fs=50.0, n_time=T)
    freqs = torch.fft.rfftfreq(T, d=1.0 / 50.0)
    above = freqs > 4.0
    assert torch.all(out[:, :, above] == 0)
    below = freqs <= 4.0
    assert torch.equal(out[:, :, below], x_freq[:, :, below])


def test_phase_shift_zero_max_is_identity():
    x_freq = to_frequency(torch.randn(B, C, T))
    out = phase_shift(x_freq, max_shift_rad=0.0)
    assert torch.allclose(out, x_freq, atol=1e-6)


def test_phase_shift_preserves_magnitude():
    x_freq = to_frequency(torch.randn(B, C, T))
    out = phase_shift(x_freq, max_shift_rad=3.14159265, generator=torch.Generator().manual_seed(0))
    assert torch.allclose(out.abs(), x_freq.abs(), atol=1e-4)


def test_remove_frequency_zero_ratio_is_identity():
    x_freq = to_frequency(torch.randn(B, C, T))
    out = remove_frequency(x_freq, ratio=0.0)
    assert torch.equal(out, x_freq)


def test_remove_frequency_full_ratio_zeroes_everything():
    x_freq = to_frequency(torch.randn(B, C, T))
    out = remove_frequency(x_freq, ratio=1.0)
    assert torch.all(out == 0)


def test_add_frequency_zero_ratio_is_identity():
    x_freq = to_frequency(torch.randn(B, C, T))
    out = add_frequency(x_freq, ratio=0.0, add_scale=0.5)
    assert torch.equal(out, x_freq)


def test_apply_frequency_domain_augmentation_zero_probability_is_identity():
    x_freq = to_frequency(torch.randn(B, C, T))
    out = apply_frequency_domain_augmentation(
        x_freq, _config(apply_probability=0.0), n_time=T, fs=50.0
    )
    assert torch.equal(out, x_freq)


def test_apply_frequency_domain_augmentation_shape_preserved():
    x_freq = to_frequency(torch.randn(B, C, T))
    out = apply_frequency_domain_augmentation(
        x_freq,
        _config(apply_probability=1.0),
        n_time=T,
        fs=50.0,
        generator=torch.Generator().manual_seed(2),
    )
    assert out.shape == x_freq.shape


# ---- fusion + orchestration ---------------------------------------------------


def test_fuse_time_frequency_output_shape():
    time_aug = np.random.randn(B, C, T).astype(np.float32)
    freq_aug = (np.random.randn(B, C, T // 2 + 1) + 1j * np.random.randn(B, C, T // 2 + 1)).astype(
        np.complex64
    )
    out = fuse_time_frequency(
        time_aug, freq_aug, {"kernel": "rbf", "n_components": 3, "gamma": None}
    )
    assert out.shape == (B, 6)  # 2 * n_components


def test_make_augmented_views_shapes():
    x = torch.randn(B, C, T)
    views = make_augmented_views(x, _config(), generator=torch.Generator().manual_seed(3))
    assert views.time_domain.shape == (B, C, T)
    assert views.frequency_domain.shape == (B, C, T)
    assert views.fused is None


def test_make_augmented_views_with_fusion():
    x = torch.randn(B, C, T)
    views = make_augmented_views(
        x, _config(), generator=torch.Generator().manual_seed(4), include_fusion=True
    )
    assert views.fused is not None
    assert views.fused.shape == (B, 6)  # 2 * n_components
