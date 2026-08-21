from __future__ import annotations

import numpy as np
import torch
from torch import nn

from chewing_ssl.common.data.windowing import WINDOW_LEN
from chewing_ssl.uncertainty.decompose import decompose_uncertainty


def enable_mc_dropout(model: nn.Module) -> None:
    """model.eval() everywhere, then re-enable train() on nn.Dropout
    submodules only -- so every other layer (Linear, GRU, GroupNorm, ...)
    stays in its deterministic eval-mode behaviour and only dropout
    becomes stochastic."""
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


@torch.no_grad()
def infer_recording_mc(
    model: nn.Module,
    x_full: np.ndarray,
    window_len: int = WINDOW_LEN,
    n_passes: int = 30,
    device: torch.device | None = None,
) -> dict:
    """Slide non-overlapping window_len windows over (C, N), exactly like
    ssl.masked.evaluate_bouts.infer_recording, but run each window through
    the model `n_passes` times with dropout active instead of once, batched
    as one (n_passes, C, window_len) forward call per window (each batch
    row gets an independent dropout mask, so this is equivalent to -- just
    faster than -- n_passes separate calls).

    Returns per-timestep arrays, each length N (NaN in the uncovered tail):
      mean        - MC mean probability (the point prediction to feed
                    downstream postprocessing/bout-matching, same role as
                    evaluate_bouts.infer_recording's single-pass output)
      std         - MC standard deviation (raw spread across passes)
      entropy     - predictive entropy of the MEAN (TOTAL uncertainty)
      exp_entropy - mean of each pass's own entropy (ALEATORIC only)
      mutual_info - entropy - exp_entropy (EPISTEMIC / BALD)
    """
    _C, N = x_full.shape
    device = device or next(model.parameters()).device
    mean = np.full(N, np.nan, dtype=np.float32)
    std = np.full(N, np.nan, dtype=np.float32)
    entropy = np.full(N, np.nan, dtype=np.float32)
    exp_entropy = np.full(N, np.nan, dtype=np.float32)
    mutual_info = np.full(N, np.nan, dtype=np.float32)

    for start in range(0, N - window_len + 1, window_len):
        xw = torch.from_numpy(x_full[:, start : start + window_len]).to(device)
        xw_batch = xw.unsqueeze(0).repeat(n_passes, 1, 1)  # (T, C, window_len)
        logits = model(xw_batch)  # (T, window_len)
        passes = torch.sigmoid(logits).cpu().numpy()
        m, s, h_tot, h_exp, mi = decompose_uncertainty(passes)
        sl = slice(start, start + window_len)
        mean[sl], std[sl] = m, s
        entropy[sl], exp_entropy[sl], mutual_info[sl] = h_tot, h_exp, mi

    return {
        "mean": mean,
        "std": std,
        "entropy": entropy,
        "exp_entropy": exp_entropy,
        "mutual_info": mutual_info,
    }
