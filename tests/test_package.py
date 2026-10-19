from importlib.metadata import version

import kv_cache_lab


def test_version_matches_package_metadata() -> None:
    assert kv_cache_lab.__version__ == version("kv-cache-lab")
