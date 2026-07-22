from __future__ import annotations

import json

import pandas as pd

from chewing_ssl.common.data.windowing import (
    DATASETS,
    EVAL_STRIDE,
    TRAIN_STRIDE,
    WINDOW_LEN,
    WindowDataset,
    _split_list,
    assign_pretrain_val_holdout,
    assign_splits,
    build_window_index,
    extract_participant,
    list_tags,
    purposes_for,
    run,
)
from tests.conftest import make_interim_segment


def test_datasets_config_has_all_five_expected():
    assert set(DATASETS) == {
        "SIPMEatingStudy",
        "SussexEatingUK",
        "FEIT_eating_S05",
        "Office_eating_s05",
        "Spectacle_NonMeal",
    }


def test_extract_participant_identity_when_no_underscore():
    assert extract_participant("SIPMEatingStudy", "p01") == "p01"
    assert extract_participant("SussexEatingUK", "EMTEQRAP2") == "EMTEQRAP2"
    assert extract_participant("FEIT_eating_S05", "p03") == "p03"
    assert extract_participant("Office_eating_s05", "p03") == "p03"


def test_extract_participant_spectacle_nonmeal_splits_on_first_underscore():
    assert (
        extract_participant("Spectacle_NonMeal", "p01_week2_p01_year_2024-month_9-day_3_snippet004")
        == "p01"
    )


def test_split_list_assigns_every_item_exactly_once():
    items = [f"p{i:02d}" for i in range(31)]
    assignment = _split_list(items, {"train": 0.6, "val": 0.2, "test": 0.2}, seed=42)
    assert set(assignment.keys()) == set(items)
    assert set(assignment.values()) <= {"train", "val", "test"}


def test_split_list_is_deterministic():
    items = [f"p{i:02d}" for i in range(31)]
    a = _split_list(items, {"train": 0.6, "val": 0.2, "test": 0.2}, seed=42)
    b = _split_list(items, {"train": 0.6, "val": 0.2, "test": 0.2}, seed=42)
    assert a == b


def test_split_list_different_seed_can_differ():
    items = [f"p{i:02d}" for i in range(31)]
    a = _split_list(items, {"train": 0.6, "val": 0.2, "test": 0.2}, seed=42)
    b = _split_list(items, {"train": 0.6, "val": 0.2, "test": 0.2}, seed=1)
    assert a != b


def test_split_list_ratios_approximately_correct():
    items = [f"p{i:02d}" for i in range(100)]
    assignment = _split_list(items, {"train": 0.6, "val": 0.2, "test": 0.2}, seed=42)
    counts = pd.Series(assignment.values()).value_counts()
    assert counts["train"] == 60
    assert counts["val"] == 20
    assert counts["test"] == 20


def test_purposes_for_is_uniform_across_datasets():
    # Same train/val/test -> purpose mapping regardless of which dataset --
    # there's no more dataset-specific branching (Spectacle used to be special).
    for dataset in ("SIPMEatingStudy", "FEIT_eating_S05", "Spectacle_NonMeal"):
        splits = {dataset: {"p01": "train", "p02": "val", "p03": "test"}}
        assert purposes_for(dataset, "p01", splits) == ["pretrain", "finetune_train"]
        assert purposes_for(dataset, "p02", splits) == ["finetune_val"]
        assert purposes_for(dataset, "p03", splits) == ["finetune_test"]


def test_assign_pretrain_val_holdout_is_subset_of_pretrain_eligible():
    datasets = ["SIPMEatingStudy", "SussexEatingUK", "FEIT_eating_S05"]
    splits = {
        "SIPMEatingStudy": {f"p{i:02d}": ("train" if i < 8 else "test") for i in range(10)},
        "SussexEatingUK": {f"e{i:02d}": ("train" if i < 8 else "test") for i in range(10)},
        "FEIT_eating_S05": {f"f{i:02d}": ("train" if i < 8 else "test") for i in range(10)},
    }
    holdout = assign_pretrain_val_holdout(splits, datasets, seed=42, val_fraction=0.25)
    for dataset in datasets:
        eligible = {p for p, label in splits[dataset].items() if label == "train"}
        assert set(holdout[dataset]) <= eligible


def test_list_tags_on_synthetic_interim_tree(synthetic_interim_tree):
    tags = list_tags(synthetic_interim_tree, "SIPMEatingStudy")
    assert tags == ["p01", "p02", "p03", "p04", "p05"]


def test_list_tags_missing_dataset_returns_empty(tmp_path):
    assert list_tags(tmp_path, "SIPMEatingStudy") == []


def test_assign_splits_on_synthetic_tree(synthetic_interim_tree):
    splits = assign_splits(synthetic_interim_tree, seed=42)
    assert set(splits["SIPMEatingStudy"]) == {"p01", "p02", "p03", "p04", "p05"}
    assert set(splits["SussexEatingUK"]) == {f"EMTEQ{i}" for i in range(1, 6)}
    assert set(splits["FEIT_eating_S05"]) == {"p01", "p02", "p03"}
    assert set(splits["Office_eating_s05"]) == {"p01", "p02", "p03"}
    # Spectacle_NonMeal tags are "p0N_snippet001" -- must resolve to just "p0N".
    assert set(splits["Spectacle_NonMeal"]) == {"p01", "p02", "p03"}
    for dataset_splits in splits.values():
        assert set(dataset_splits.values()) <= {"train", "val", "test"}


def test_build_window_index_respects_purpose_and_stride(synthetic_interim_tree):
    splits = assign_splits(synthetic_interim_tree, seed=42)
    holdout = assign_pretrain_val_holdout(splits, seed=42)
    index = build_window_index(synthetic_interim_tree, splits, holdout)

    assert len(index) > 0
    assert set(index["purpose"]) <= {
        "pretrain_train",
        "pretrain_val",
        "finetune_train",
        "finetune_val",
        "finetune_test",
        "meal_test",
    }

    # every (dataset, participant) belongs to exactly one of the mutually-exclusive
    # purpose families (a participant can't simultaneously be e.g. finetune_val
    # and finetune_test, but train participants legitimately feed both pretrain_*
    # and finetune_train).
    train_rows = index[index["purpose"].isin(["finetune_train", "pretrain_train", "pretrain_val"])]
    val_rows = index[index["purpose"] == "finetune_val"]
    test_rows = index[index["purpose"] == "finetune_test"]
    assert set(zip(val_rows["dataset"], val_rows["participant"])).isdisjoint(
        set(zip(test_rows["dataset"], test_rows["participant"]))
    )
    assert set(zip(val_rows["dataset"], val_rows["participant"])).isdisjoint(
        set(zip(train_rows["dataset"], train_rows["participant"]))
    )

    # stride sanity: consecutive windows for the same segment should be spaced
    # by exactly the purpose's stride.
    one_purpose = index[index["purpose"] == "finetune_train"]
    if len(one_purpose) > 1:
        one_segment = one_purpose[
            one_purpose["segment_path"] == one_purpose["segment_path"].iloc[0]
        ]
        starts = sorted(one_segment["start"].tolist())
        if len(starts) > 1:
            assert starts[1] - starts[0] == TRAIN_STRIDE

    eval_rows = index[index["purpose"] == "finetune_val"]
    if len(eval_rows) > 1:
        one_segment = eval_rows[eval_rows["segment_path"] == eval_rows["segment_path"].iloc[0]]
        starts = sorted(one_segment["start"].tolist())
        if len(starts) > 1:
            assert starts[1] - starts[0] == EVAL_STRIDE


def test_build_window_index_excludes_unlabelled_segments_from_labelled_purposes(tmp_path):
    root = tmp_path / "interim"
    # an unlabelled SIPM "train" participant shouldn't be possible in practice,
    # but build_window_index must still never emit a labelled-purpose row for it.
    make_interim_segment(root, "SIPMEatingStudy", "p01", n_samples=300, has_label=False)
    splits = {"SIPMEatingStudy": {"p01": "train"}}
    holdout = {"SIPMEatingStudy": []}
    index = build_window_index(root, splits, holdout, datasets=["SIPMEatingStudy"])
    assert "finetune_train" not in set(index["purpose"])
    assert set(index["purpose"]) <= {"pretrain_train", "pretrain_val"}


def test_build_window_index_custom_window_len(tmp_path):
    root = tmp_path / "interim"
    make_interim_segment(root, "SIPMEatingStudy", "p01", n_samples=300, has_label=False)
    splits = {"SIPMEatingStudy": {"p01": "train"}}
    holdout = {"SIPMEatingStudy": []}
    index = build_window_index(root, splits, holdout, datasets=["SIPMEatingStudy"], window_len=100)
    assert (index["end"] - index["start"] == 100).all()


def test_window_dataset_shapes_and_values(tmp_path):
    from chewing_ssl.common.data.preprocess import SENSOR_CHANNELS

    seg_dir = make_interim_segment(tmp_path / "interim", "SIPMEatingStudy", "p01", n_samples=300)
    seg_path = seg_dir / "segment_00.parquet"
    segment = pd.read_parquet(seg_path)

    index = pd.DataFrame(
        [
            {
                "dataset": "SIPMEatingStudy",
                "participant": "p01",
                "tag": "p01",
                "purpose": "finetune_train",
                "segment_path": str(seg_path),
                "start": 0,
                "end": WINDOW_LEN,
            },
            {
                "dataset": "SIPMEatingStudy",
                "participant": "p01",
                "tag": "p01",
                "purpose": "finetune_train",
                "segment_path": str(seg_path),
                "start": 50,
                "end": 50 + WINDOW_LEN,
            },
        ]
    )
    ds = WindowDataset(index, with_label=True)
    assert len(ds) == 2

    x, y = ds[0]
    assert x.shape == (len(SENSOR_CHANNELS), WINDOW_LEN)
    assert y.shape == (WINDOW_LEN,)
    assert (x[0] == segment[SENSOR_CHANNELS[0]].to_numpy(dtype="float32")[0:WINDOW_LEN]).all()

    # cache hit: same segment_path accessed twice should give consistent values
    x2, _ = ds[1]
    assert x2.shape == (len(SENSOR_CHANNELS), WINDOW_LEN)


def test_window_dataset_evicts_least_recently_used_segment(tmp_path):
    paths = []
    for i in range(3):
        seg_dir = make_interim_segment(
            tmp_path / "interim", "SIPMEatingStudy", f"p0{i}", n_samples=300
        )
        paths.append(str(seg_dir / "segment_00.parquet"))

    index = pd.DataFrame(
        [
            {
                "dataset": "SIPMEatingStudy",
                "participant": f"p0{i}",
                "tag": f"p0{i}",
                "purpose": "finetune_train",
                "segment_path": p,
                "start": 0,
                "end": WINDOW_LEN,
            }
            for i, p in enumerate(paths)
        ]
    )
    ds = WindowDataset(index, with_label=True, cache_size=2)
    for i in range(3):
        ds[i]
    # cache_size=2: after loading all 3 segments in order, the first should be evicted
    assert len(ds._cache) == 2
    assert paths[0] not in ds._cache
    assert paths[1] in ds._cache
    assert paths[2] in ds._cache


def test_run_writes_splits_and_index(synthetic_interim_tree, tmp_path):
    out_root = tmp_path / "processed"
    run(synthetic_interim_tree, out_root, seed=42)

    splits_path = out_root / "splits.json"
    assert splits_path.exists()
    splits = json.loads(splits_path.read_text())
    assert "pretrain_val_holdout" in splits
    assert set(splits["SIPMEatingStudy"].values()) <= {"train", "val", "test"}

    index_dir = out_root / "index"
    assert index_dir.exists()
    written_purposes = {p.stem for p in index_dir.glob("*.parquet")}
    assert written_purposes  # at least one purpose got windows
    assert written_purposes <= {
        "pretrain_train",
        "pretrain_val",
        "finetune_train",
        "finetune_val",
        "finetune_test",
        "meal_test",
    }


def test_run_reuses_existing_splits_on_second_call(synthetic_interim_tree, tmp_path):
    out_root = tmp_path / "processed"
    run(synthetic_interim_tree, out_root, seed=42)
    first = json.loads((out_root / "splits.json").read_text())

    run(synthetic_interim_tree, out_root, seed=999)  # different seed, should be ignored
    second = json.loads((out_root / "splits.json").read_text())

    assert first == second


def test_run_splits_only_skips_index(synthetic_interim_tree, tmp_path):
    out_root = tmp_path / "processed"
    run(synthetic_interim_tree, out_root, seed=42, splits_only=True)

    assert (out_root / "splits.json").exists()
    assert not (out_root / "index").exists()


def test_run_custom_window_len(synthetic_interim_tree, tmp_path):
    out_root = tmp_path / "processed"
    run(synthetic_interim_tree, out_root, seed=42, window_len=100)

    index_dir = out_root / "index"
    one = next(index_dir.glob("*.parquet"))
    df = pd.read_parquet(one)
    assert (df["end"] - df["start"] == 100).all()


def test_window_dataset_without_label(tmp_path):
    seg_dir = make_interim_segment(
        tmp_path / "interim", "Spectacle_NonMeal", "p01_snippet001", n_samples=300, has_label=False
    )
    seg_path = seg_dir / "segment_00.parquet"
    index = pd.DataFrame(
        [
            {
                "dataset": "Spectacle_NonMeal",
                "participant": "p01",
                "tag": "p01_snippet001",
                "purpose": "pretrain_train",
                "segment_path": str(seg_path),
                "start": 0,
                "end": WINDOW_LEN,
            }
        ]
    )
    ds = WindowDataset(index, with_label=False)
    x = ds[0]
    assert x.shape[1] == WINDOW_LEN
