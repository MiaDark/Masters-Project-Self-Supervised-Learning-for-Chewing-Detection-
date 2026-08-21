"""TF-C-style contrastive augmentation (decision 004): time-domain and
frequency-domain perturbations of a window batch, for generating positive
pairs/views for a contrastive pretext task -- combined with, not replacing,
the masked-reconstruction task in chewing_ssl.ssl.models.masking.

Applied per-batch, at training time, to already-preprocessed and normalized
windows (mean 0, std 1 per channel). Parameters live in configs/augmentation.yaml.

Two views, both directly usable through the existing CNN/Bi-GRU
encoders (same (B, C, T) shape as the input):
  - `apply_time_domain_augmentation`: jitter, scaling, permutation, masking.
  - `apply_frequency_domain_augmentation` + `from_frequency`: low-pass,
    phase-shift, remove/add-frequency, computed via FFT then inverse-FFT
    back to a (B, C, T) time-domain signal.
`make_augmented_views` returns both together.

Open question (decision 004): the paper's kernel-PCA FUSION of these two
views (`fuse_time_frequency`, provided below for experimentation/fidelity)
produces a flattened (batch, 2 * n_components) embedding, not a (C, T)
sequence -- it does not fit the existing sequence encoders as-is, and it's a
*batch-level* operation (KernelPCA fits on the whole batch's kernel matrix,
so one item's fused representation depends on which other items share its
batch -- unusual for an augmentation). Until the training loop resolves how
(or whether) to use that fused embedding, `make_augmented_views`'s two plain
(B, C, T) views are the more directly usable path for a standard two-view
contrastive setup.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.decomposition import KernelPCA

# ---- time-domain augmentations, x: (B, C, T) -------------------------------


def jitter(x: torch.Tensor, sigma: float, generator: torch.Generator | None = None) -> torch.Tensor:
    """Additive Gaussian noise, independent per element."""
    noise = torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype) * sigma
    return x + noise


def scaling(
    x: torch.Tensor, mean: float, sigma: float, generator: torch.Generator | None = None
) -> torch.Tensor:
    """One scale factor ~ N(mean, sigma) per (batch item, channel), broadcast across time."""
    b, c, _ = x.shape
    factor = (
        torch.randn(b, c, 1, generator=generator, device=x.device, dtype=x.dtype) * sigma + mean
    )
    return x * factor


def permutation(
    x: torch.Tensor, n_segments: int, generator: torch.Generator | None = None
) -> torch.Tensor:
    """Split the time axis into n_segments chunks and shuffle their order,
    independently per batch item. The same permutation is applied to every
    channel at once, so cross-channel alignment within a chunk survives --
    only the between-chunk ordering is disrupted."""
    b, _, t = x.shape
    bounds = np.linspace(0, t, n_segments + 1, dtype=int)
    spans = [
        (int(bounds[i]), int(bounds[i + 1])) for i in range(n_segments) if bounds[i + 1] > bounds[i]
    ]
    out = torch.empty_like(x)
    for i in range(b):
        order = torch.randperm(len(spans), generator=generator).tolist()
        out[i] = torch.cat([x[i, :, spans[j][0] : spans[j][1]] for j in order], dim=-1)
    return out


def masking(
    x: torch.Tensor, mask_ratio: float, mask_value: float, generator: torch.Generator | None = None
) -> torch.Tensor:
    """One contiguous span covering mask_ratio of timesteps, independently per
    batch item, replaced with mask_value across all channels. A much milder
    perturbation than models.masking's span_mask_ratio=0.4 reconstruction
    target -- this is meant to be a survivable view for contrastive learning,
    not a task that requires reconstructing most of the window."""
    b, _, t = x.shape
    span_len = max(round(mask_ratio * t), 0)
    out = x.clone()
    if span_len == 0:
        return out
    for i in range(b):
        hi = max(1, t - span_len + 1)
        start = int(torch.randint(0, hi, (1,), generator=generator).item())
        out[i, :, start : start + span_len] = mask_value
    return out


TIME_DOMAIN_FNS = {
    "jitter": lambda x, cfg, gen: jitter(x, cfg["sigma"], gen),
    "scaling": lambda x, cfg, gen: scaling(x, cfg["mean"], cfg["sigma"], gen),
    "permutation": lambda x, cfg, gen: permutation(x, cfg["n_segments"], gen),
    "masking": lambda x, cfg, gen: masking(x, cfg["mask_ratio"], cfg["mask_value"], gen),
}


def apply_time_domain_augmentation(
    x: torch.Tensor, config: dict, generator: torch.Generator | None = None
) -> torch.Tensor:
    """Each enabled config['time_domain'] augmentation, applied independently
    with probability config['apply_probability'] (TF-C paper: 0.25), composed
    sequentially in the order listed if more than one fires."""
    p = config["apply_probability"]
    out = x
    for name, fn in TIME_DOMAIN_FNS.items():
        sub_cfg = config["time_domain"].get(name)
        if not sub_cfg or not sub_cfg.get("enabled", True):
            continue
        if torch.rand(1, generator=generator).item() < p:
            out = fn(out, sub_cfg, generator)
    return out


# ---- frequency-domain augmentations, x_freq: (B, C, F) complex -------------


def to_frequency(x: torch.Tensor) -> torch.Tensor:
    """rFFT along the time axis. (B, C, T) real -> (B, C, T // 2 + 1) complex."""
    return torch.fft.rfft(x, dim=-1)


def from_frequency(x_freq: torch.Tensor, n: int) -> torch.Tensor:
    """Inverse of to_frequency. n = the original T (needed since rfft/irfft
    can't infer odd-vs-even length from the frequency-domain shape alone)."""
    return torch.fft.irfft(x_freq, n=n, dim=-1)


def low_pass_freq(x_freq: torch.Tensor, cutoff_hz: float, fs: float, n_time: int) -> torch.Tensor:
    """Brick-wall low-pass: zero every bin above cutoff_hz. Deliberately a
    different mechanism from preprocessing's smooth Butterworth (a hard
    frequency-domain cutoff, not a zero-phase IIR filter) -- see
    configs/augmentation.yaml for why the cutoff itself also differs."""
    freqs = torch.fft.rfftfreq(n_time, d=1.0 / fs).to(x_freq.device)
    mask = (freqs <= cutoff_hz).view(1, 1, -1)
    return x_freq * mask


def phase_shift(
    x_freq: torch.Tensor, max_shift_rad: float, generator: torch.Generator | None = None
) -> torch.Tensor:
    """Random phase offset, uniform in [-max_shift_rad, max_shift_rad], one
    value per (batch item, channel) applied to every frequency bin -- per the
    paper: "introduces a random value ranging from -pi to pi to the existing
    phase values"."""
    b, c, _ = x_freq.shape
    shift = (torch.rand(b, c, 1, generator=generator, device=x_freq.device) * 2 - 1) * max_shift_rad
    return x_freq * torch.exp(1j * shift)


def remove_frequency(
    x_freq: torch.Tensor, ratio: float, generator: torch.Generator | None = None
) -> torch.Tensor:
    """Zero `ratio` fraction of frequency bins, independent random mask per
    (batch item, channel, bin)."""
    keep = torch.rand(x_freq.shape, generator=generator, device=x_freq.device) >= ratio
    return x_freq * keep


def add_frequency(
    x_freq: torch.Tensor, ratio: float, add_scale: float, generator: torch.Generator | None = None
) -> torch.Tensor:
    """Add small random-magnitude, random-phase components at `ratio` fraction
    of bins, magnitude scaled relative to that bin's existing amplitude."""
    mask = torch.rand(x_freq.shape, generator=generator, device=x_freq.device) < ratio
    amplitude = x_freq.abs() * add_scale
    random_phase = torch.rand(x_freq.shape, generator=generator, device=x_freq.device) * 2 * np.pi
    perturbation = amplitude * torch.exp(1j * random_phase)
    return x_freq + perturbation * mask


FREQUENCY_DOMAIN_FNS = {
    "low_pass": lambda xf, cfg, n_time, fs, gen: low_pass_freq(xf, cfg["cutoff_hz"], fs, n_time),
    "phase_shift": lambda xf, cfg, n_time, fs, gen: phase_shift(xf, cfg["max_shift_rad"], gen),
    "remove_frequency": lambda xf, cfg, n_time, fs, gen: remove_frequency(xf, cfg["ratio"], gen),
    "add_frequency": lambda xf, cfg, n_time, fs, gen: add_frequency(
        xf, cfg["ratio"], cfg["add_scale"], gen
    ),
}


def apply_frequency_domain_augmentation(
    x_freq: torch.Tensor,
    config: dict,
    n_time: int,
    fs: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Each enabled config['frequency_domain'] augmentation, applied
    independently with probability config['apply_probability'], composed
    sequentially. Returns the perturbed FREQUENCY-domain tensor (same shape
    as x_freq) -- inverse-transform (from_frequency) for a time-domain view,
    or pass directly to fuse_time_frequency."""
    p = config["apply_probability"]
    out = x_freq
    for name, fn in FREQUENCY_DOMAIN_FNS.items():
        sub_cfg = config["frequency_domain"].get(name)
        if not sub_cfg or not sub_cfg.get("enabled", True):
            continue
        if torch.rand(1, generator=generator).item() < p:
            out = fn(out, sub_cfg, n_time, fs, generator)
    return out


# ---- kernel-PCA fusion (literal paper method; see module docstring) -------


def fuse_time_frequency(
    time_aug: np.ndarray, freq_aug: np.ndarray, kernel_pca_config: dict
) -> np.ndarray:
    """datafinal = KernelPCA(time_aug_flat) concat KernelPCA(freq_aug_flat),
    exactly as described in the TF-C paper. time_aug: (B, C, T) real,
    freq_aug: (B, C, F) complex (real/imag concatenated when flattened, since
    KernelPCA needs real-valued input). Returns (B, 2 * n_components) --
    see the module docstring for why this doesn't feed the sequence encoders
    directly, and why it's a batch-level (not per-sample) transform.
    """
    b = time_aug.shape[0]
    time_flat = time_aug.reshape(b, -1)
    freq_flat = np.concatenate([freq_aug.real.reshape(b, -1), freq_aug.imag.reshape(b, -1)], axis=1)

    kwargs = {
        "n_components": kernel_pca_config["n_components"],
        "kernel": kernel_pca_config["kernel"],
        "gamma": kernel_pca_config.get("gamma"),
    }
    time_embedded = KernelPCA(**kwargs).fit_transform(time_flat)
    freq_embedded = KernelPCA(**kwargs).fit_transform(freq_flat)
    return np.concatenate([time_embedded, freq_embedded], axis=1)


# ---- orchestration ----------------------------------------------------------


@dataclass
class AugmentedViews:
    time_domain: torch.Tensor  # (B, C, T)
    frequency_domain: torch.Tensor  # (B, C, T) -- frequency-augmented, inverse-transformed back
    fused: np.ndarray | None = (
        None  # (B, 2 * n_components), only if requested -- see fuse_time_frequency
    )


def make_augmented_views(
    x: torch.Tensor,
    config: dict,
    generator: torch.Generator | None = None,
    include_fusion: bool = False,
) -> AugmentedViews:
    """The two augmented views for contrastive pretraining. x: (B, C, T),
    already preprocessed and normalized."""
    n_time = x.shape[-1]
    fs = config["target_hz"]

    time_view = apply_time_domain_augmentation(x, config, generator)

    x_freq = to_frequency(x)
    x_freq_aug = apply_frequency_domain_augmentation(x_freq, config, n_time, fs, generator)
    freq_view = from_frequency(x_freq_aug, n=n_time)

    fused = None
    if include_fusion:
        fused = fuse_time_frequency(
            time_view.detach().cpu().numpy(),
            x_freq_aug.detach().cpu().numpy(),
            config["kernel_pca"],
        )

    return AugmentedViews(time_domain=time_view, frequency_domain=freq_view, fused=fused)
