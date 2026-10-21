"""Shared pytest configuration.

Tests marked ``gpu`` are skipped automatically when CuPy or a CUDA device is
missing, so the same suite runs on a laptop GPU and on a CPU-only CI runner.
"""

import pytest

from lbm.backends import gpu_available


def pytest_collection_modifyitems(config, items):
    if gpu_available():
        return
    skip_gpu = pytest.mark.skip(reason="CuPy or a CUDA device is not available")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)
