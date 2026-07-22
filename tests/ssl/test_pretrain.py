from __future__ import annotations

import copy

import pandas as pd
import torch
from torch.utils.data import DataLoader

from chewing_ssl.common.data.preprocess import SENSOR_CHANNELS
from chewing_ssl.common.data.windowing import WINDOW_LEN, WindowDataset
from chewing_ssl.ssl.models.bigru_autoencoder import BiGRUAutoencoder
from chewing_ssl.ssl.masked.pretrain import load_index, pick_device, run_epoch
from tests.conftest import make_interim_segment


def test_pick_device_respects_explicit_override():
    assert pick_device("cpu") == torch.device("cpu")


def test_pick_device_default_returns_a_valid_device():
    device = pick_device(None)
    assert device.type in {"cuda", "mps", "cpu"}


def test_load_index_subsamples_deterministically(tmp_path):
    index_dir = tmp_path / "index"
    index_dir.mkdir(parents=True)
    full = pd.DataFrame({"start": range(100), "end": range(WINDOW_LEN, 100 + WINDOW_LEN)})
    full.to_parquet(index_dir / "pretrain_train.parquet", index=False)

    a = load_index(tmp_path, "pretrain_train", max_windows=10, seed=42)
    b = load_index(tmp_path, "pretrain_train", max_windows=10, seed=42)
    assert len(a) == 10
    assert a["start"].tolist() == b["start"].tolist()


def test_load_index_no_subsample_when_smaller_than_cap(tmp_path):
    index_dir = tmp_path / "index"
    index_dir.mkdir(parents=True)
    full = pd.DataFrame({"start": range(5), "end": range(WINDOW_LEN, 5 + WINDOW_LEN)})
    full.to_parquet(index_dir / "pretrain_train.parquet", index=False)

    out = load_index(tmp_path, "pretrain_train", max_windows=1000, seed=42)
    assert len(out) == 5


def _tiny_loader(tmp_path, n_windows=12):
    seg_dir = make_interim_segment(tmp_path / "interim", "SIPMEatingStudy", "p01", n_samples=300)
    seg_path = seg_dir / "segment_00.parquet"
    rows = [
        {"dataset": "SIPMEatingStudy", "participant": "p01", "tag": "p01", "purpose": "pretrain_train",
         "segment_path": str(seg_path), "start": i, "end": i + WINDOW_LEN}
        for i in range(0, n_windows)
    ]
    index = pd.DataFrame(rows)
    return DataLoader(WindowDataset(index, with_label=False), batch_size=4, shuffle=False)


def test_run_epoch_train_updates_parameters(tmp_path):
    torch.manual_seed(0)
    loader = _tiny_loader(tmp_path)
    model = BiGRUAutoencoder(n_channels=len(SENSOR_CHANNELS), hidden_size=8, num_layers=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    before = copy.deepcopy(model.state_dict())

    metrics = run_epoch(model, loader, torch.device("cpu"), {"span_mask_ratio": 0.4, "channel_mask_prob": 0.15}, optimizer)

    assert metrics["loss"] >= 0
    assert set(metrics["per_channel"]) == set(SENSOR_CHANNELS)
    after = model.state_dict()
    changed = any(not torch.equal(before[k], after[k]) for k in before)
    assert changed, "optimizer step should have changed at least one parameter"


def test_run_epoch_eval_does_not_update_parameters(tmp_path):
    torch.manual_seed(0)
    loader = _tiny_loader(tmp_path)
    model = BiGRUAutoencoder(n_channels=len(SENSOR_CHANNELS), hidden_size=8, num_layers=1)
    before = copy.deepcopy(model.state_dict())

    run_epoch(model, loader, torch.device("cpu"), {"span_mask_ratio": 0.4, "channel_mask_prob": 0.15}, optimizer=None)

    after = model.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)


def test_run_epoch_eval_sets_model_to_eval_mode(tmp_path):
    loader = _tiny_loader(tmp_path)
    model = BiGRUAutoencoder(n_channels=len(SENSOR_CHANNELS), hidden_size=8, num_layers=1)
    model.train()
    run_epoch(model, loader, torch.device("cpu"), {"span_mask_ratio": 0.4, "channel_mask_prob": 0.15}, optimizer=None)
    assert model.training is False
