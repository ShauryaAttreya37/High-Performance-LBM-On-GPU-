"""D2Q9 lattice constants must satisfy the moment identities that make LBM recover Navier-Stokes."""

import itertools

import numpy as np
import pytest

from lbm import lattice

C = np.stack([lattice.CX, lattice.CY])  # shape (2, Q)


def test_weights_sum_to_one():
    assert lattice.W.sum() == pytest.approx(1.0, abs=1e-15)


def test_opposite_reverses_each_velocity():
    opp = lattice.OPPOSITE
    np.testing.assert_array_equal(lattice.CX[opp], -lattice.CX)
    np.testing.assert_array_equal(lattice.CY[opp], -lattice.CY)


def test_opposite_is_an_involution():
    np.testing.assert_array_equal(lattice.OPPOSITE[lattice.OPPOSITE], np.arange(lattice.Q))


def test_odd_moments_of_weights_vanish():
    for a in range(2):
        assert np.sum(lattice.W * C[a]) == pytest.approx(0.0, abs=1e-15)
    for a, b, c in itertools.product(range(2), repeat=3):
        assert np.sum(lattice.W * C[a] * C[b] * C[c]) == pytest.approx(0.0, abs=1e-15)


def test_second_moment_is_isotropic():
    for a, b in itertools.product(range(2), repeat=2):
        expected = lattice.CS2 if a == b else 0.0
        assert np.sum(lattice.W * C[a] * C[b]) == pytest.approx(expected, abs=1e-15)


def test_fourth_moment_is_isotropic():
    delta = np.eye(2)
    for a, b, c, d in itertools.product(range(2), repeat=4):
        expected = lattice.CS2**2 * (
            delta[a, b] * delta[c, d] + delta[a, c] * delta[b, d] + delta[a, d] * delta[b, c]
        )
        moment = np.sum(lattice.W * C[a] * C[b] * C[c] * C[d])
        assert moment == pytest.approx(expected, abs=1e-15)


def test_constant_arrays_are_read_only():
    with pytest.raises(ValueError):
        lattice.W[0] = 0.0
    with pytest.raises(ValueError):
        lattice.CX[1] = 2


def test_python_tuples_match_arrays():
    assert tuple(int(c) for c in lattice.CX) == lattice.CX_TUPLE
    assert tuple(int(c) for c in lattice.CY) == lattice.CY_TUPLE
    assert tuple(float(w) for w in lattice.W) == lattice.W_TUPLE
    assert tuple(int(o) for o in lattice.OPPOSITE) == lattice.OPPOSITE_TUPLE


@pytest.mark.parametrize("c_type", ["float", "double"])
def test_cuda_preamble_declares_types_and_every_constant(c_type):
    source = lattice.cuda_preamble(c_type)
    assert f"typedef {c_type} real_t;" in source
    assert "#define Q 9" in source
    for name in ("cx", "cy", "opp", "weight"):
        assert f"constexpr {'real_t' if name == 'weight' else 'int'} {name}(int i)" in source
    for w in lattice.W_TUPLE:
        assert repr(w) in source


def test_cuda_preamble_rejects_unknown_type():
    with pytest.raises(ValueError, match="real type"):
        lattice.cuda_preamble("half")
