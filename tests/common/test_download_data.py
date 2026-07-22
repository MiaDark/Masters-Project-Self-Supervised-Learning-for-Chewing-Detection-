from pathlib import Path

from chewing_ssl.common.data.download import (
    parse_int_ranges,
    participant_number,
    wanted_extension,
)


def test_package_imports():
    import chewing_ssl  # noqa: F401
    from chewing_ssl.common.data import download  # noqa: F401
    from chewing_ssl.common.utils import seed  # noqa: F401


def test_set_seed_runs():
    from chewing_ssl.common.utils.seed import set_seed

    set_seed(123)  # should not raise even without torch installed


def test_data_config_loads():
    from chewing_ssl.common.utils.config import load_yaml

    cfg = load_yaml(Path("configs/data.yaml"))

    assert "s3" in cfg
    assert "paths" in cfg


def test_data_config_loads_from_any_cwd(monkeypatch, tmp_path):
    from chewing_ssl.common.utils.config import load_data_config

    monkeypatch.chdir(tmp_path)

    cfg = load_data_config()

    assert "s3" in cfg
    assert "paths" in cfg


def test_parse_int_ranges():
    assert parse_int_ranges(None) is None
    assert parse_int_ranges("") is None
    assert parse_int_ranges("all") is None
    assert parse_int_ranges("*") is None
    assert parse_int_ranges("1") == {1}
    assert parse_int_ranges("1-3") == {1, 2, 3}
    assert parse_int_ranges("1-3,5,7-8") == {1, 2, 3, 5, 7, 8}


def test_participant_number():
    assert participant_number("p001") == 1
    assert participant_number("p42") == 42
    assert participant_number("participant42") is None
    assert participant_number("abc") is None


def test_wanted_extension():
    assert wanted_extension("file.csv", [".csv"])
    assert wanted_extension("file.CSV", [".csv"])
    assert wanted_extension("file.txt", [])
    assert not wanted_extension("file.txt", [".csv"])