"""Lid-driven cavity against Ghia, Ghia & Shin (1982).

Errors are the largest centreline deviation from Ghia's table, in units of the
lid velocity. Measured (TRT, U = 0.1):
Re = 100,  n = 64:  0.0054 (float64 and float32 agree to 4 digits)
Re = 1000, n = 128: 0.0148
Re = 1000, n = 256: 0.0086
Ghia's own solution used a 129 x 129 grid, so agreement at the ~1 % level is
the meaningful target.
"""

import numpy as np
import pytest

from lbm.cases.ghia1982 import REYNOLDS_NUMBERS, ghia_centerlines
from lbm.cases.lid_driven_cavity import run_cavity
from tests.helpers import all_backends


@pytest.mark.parametrize("backend", all_backends())
def test_re100_matches_ghia(backend):
    dtype = "float64" if backend == "numpy" else "float32"
    assert run_cavity(reynolds=100, n=64, backend=backend, dtype=dtype).max_error < 0.01


@pytest.mark.gpu
def test_re1000_matches_ghia():
    assert run_cavity(reynolds=1000, n=128, backend="cuda-aa", dtype="float32").max_error < 0.025


@pytest.mark.gpu
@pytest.mark.slow
def test_re1000_error_shrinks_with_resolution():
    coarse = run_cavity(reynolds=1000, n=128, backend="cuda-aa", dtype="float32").max_error
    fine = run_cavity(reynolds=1000, n=256, backend="cuda-aa", dtype="float32").max_error
    assert fine < 0.012
    assert fine < coarse


def test_ghia_tables_are_consistent():
    for re in REYNOLDS_NUMBERS:
        ref = ghia_centerlines(re)
        assert ref.y.shape == ref.u.shape == ref.x.shape == ref.v.shape == (17,)
        assert ref.u[0] == 1.0 and ref.u[-1] == 0.0  # lid and bottom wall
        assert ref.v[0] == 0.0 and ref.v[-1] == 0.0  # side walls
        assert np.all(np.diff(ref.y) < 0) and np.all(np.diff(ref.x) < 0)


def test_unknown_reynolds_number_is_rejected():
    with pytest.raises(ValueError, match="Re"):
        ghia_centerlines(5000)
    with pytest.raises(ValueError, match="reynolds"):
        run_cavity(reynolds=5000)
