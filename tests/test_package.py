from importlib.metadata import version

import kvsim


def test_version_matches_package_metadata() -> None:
    assert kvsim.__version__ == version("kvsim")
