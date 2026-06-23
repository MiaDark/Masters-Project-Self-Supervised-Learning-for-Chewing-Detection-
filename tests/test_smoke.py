"""Smoke tests — confirm the package wiring is intact."""

from pathlib import Path


def test_package_imports():
    import chewing_ssl  # noqa: F401
    from chewing_ssl.data import download  # noqa: F401
    from chewing_ssl.utils import seed  # noqa: F401


def test_set_seed_runs():
    from chewing_ssl.utils.seed import set_seed

    set_seed(123)  # should not raise even without torch installed


def test_data_config_loads():
    from chewing_ssl.utils.config import load_yaml

    cfg = load_yaml(Path("configs/data.yaml"))
    assert "s3" in cfg
    assert "paths" in cfg
