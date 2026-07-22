from __future__ import annotations

import torch

from chewing_ssl.ssl.masked.pretrain import MODEL_KWARGS
from chewing_ssl.ssl.models import ARCHITECTURES
from chewing_ssl.ssl.models.masking import apply_mask_token, masked_mse, sample_mask_batch

N_CHANNELS, WINDOW_LEN, BATCH = 7, 200, 4


def _build(name):
    return ARCHITECTURES[name](n_channels=N_CHANNELS, **MODEL_KWARGS[name])


def test_all_architectures_registered():
    assert set(ARCHITECTURES) == {"cnn", "bigru"}


def test_all_architectures_preserve_shape():
    x = torch.randn(BATCH, N_CHANNELS, WINDOW_LEN)
    for name in ARCHITECTURES:
        model = _build(name)
        out = model(x)
        assert out.shape == x.shape, f"{name} changed shape: {out.shape}"


def test_all_architectures_have_a_mask_token_per_channel():
    for name in ARCHITECTURES:
        model = _build(name)
        assert model.mask_token.shape == (N_CHANNELS,)


def test_all_architectures_backprop_reaches_every_parameter():
    x = torch.randn(BATCH, N_CHANNELS, WINDOW_LEN)
    for name in ARCHITECTURES:
        model = _build(name)
        mask = sample_mask_batch(BATCH, N_CHANNELS, WINDOW_LEN)
        x_in = apply_mask_token(x, mask, model.mask_token)
        recon = model(x_in)
        loss = masked_mse(recon, x, mask)
        loss.backward()
        missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
        assert not missing, f"{name}: parameters with no gradient: {missing}"


def test_architecture_parameter_counts_are_comparable():
    """Regression guard: the two architectures were deliberately tuned to a
    comparable ~100-120K parameter budget so an architecture comparison isn't
    confounded by one model simply having much more capacity (see
    docs/architecture.md / ssl.masked.pretrain.MODEL_KWARGS)."""
    counts = {name: sum(p.numel() for p in _build(name).parameters()) for name in ARCHITECTURES}
    assert max(counts.values()) / min(counts.values()) < 2.5, counts


def test_cnn_autoencoder_rejects_non_power_of_two_friendly_length():
    from chewing_ssl.ssl.models.cnn_autoencoder import CNNAutoencoder

    model = CNNAutoencoder(n_channels=N_CHANNELS, depth=3)
    # 200 is divisible by 2**3=8, should work without error
    x = torch.randn(2, N_CHANNELS, 200)
    out = model(x)
    assert out.shape == x.shape


def test_all_architectures_encode_matches_embedding_dim():
    x = torch.randn(BATCH, N_CHANNELS, WINDOW_LEN)
    for name in ARCHITECTURES:
        model = _build(name)
        h = model.encode(x)
        assert h.shape[0] == BATCH, f"{name}: batch dim wrong"
        assert h.shape[-1] == model.embedding_dim, f"{name}: encode() last dim != embedding_dim"
        assert h.dim() == 3, f"{name}: encode() should be (B, T', D)"


def test_all_architectures_encode_is_differentiable():
    # decoder-only params (reconstruction head, CNN up-path, mask_token) get no
    # gradient from encode() alone -- that's expected, not a bug. Just check
    # every remaining (encoder) parameter does.
    decoder_only = {"head", "mask_token", "upsamplers", "up_blocks"}
    x = torch.randn(BATCH, N_CHANNELS, WINDOW_LEN)
    for name in ARCHITECTURES:
        model = _build(name)
        h = model.encode(x)
        h.sum().backward()
        missing = [
            n for n, p in model.named_parameters()
            if p.requires_grad and p.grad is None and n.split(".")[0] not in decoder_only
        ]
        assert not missing, f"{name}: encoder parameters with no gradient: {missing}"
