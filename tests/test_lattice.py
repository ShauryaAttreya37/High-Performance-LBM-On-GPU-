"""Velocity sets must satisfy the moment identities that make LBM recover Navier-Stokes."""

import itertools

import numpy as np
import pytest

from lbm import lattice
from lbm.lattice import D2Q9, D3Q19, LATTICES, VelocitySet, cuda_preamble, default_lattice

ALL = pytest.mark.parametrize("vs", list(LATTICES.values()), ids=list(LATTICES))


def moment(vs, *axes):
    """sum_i w_i c_ia c_ib ... for the given axes."""
    c = vs.c_array.astype(float)
    product = np.ones(vs.q)
    for a in axes:
        product = product * c[:, a]
    return float(np.sum(vs.w_array * product))


@ALL
def test_weights_sum_to_one(vs):
    assert vs.w_array.sum() == pytest.approx(1.0, abs=1e-15)


@ALL
def test_opposite_reverses_each_velocity(vs):
    np.testing.assert_array_equal(vs.c_array[list(vs.opposite)], -vs.c_array)


@ALL
def test_opposite_is_an_involution(vs):
    opp = np.array(vs.opposite)
    np.testing.assert_array_equal(opp[opp], np.arange(vs.q))


@ALL
def test_odd_moments_vanish(vs):
    for axes in itertools.chain(
        itertools.product(range(vs.d), repeat=1), itertools.product(range(vs.d), repeat=3)
    ):
        assert moment(vs, *axes) == pytest.approx(0.0, abs=1e-15)


@ALL
def test_second_moment_is_isotropic(vs):
    for a, b in itertools.product(range(vs.d), repeat=2):
        assert moment(vs, a, b) == pytest.approx(vs.cs2 * (a == b), abs=1e-15)


@ALL
def test_fourth_moment_is_isotropic(vs):
    delta = np.eye(vs.d)
    for a, b, c, d in itertools.product(range(vs.d), repeat=4):
        expected = vs.cs2**2 * (
            delta[a, b] * delta[c, d] + delta[a, c] * delta[b, d] + delta[a, d] * delta[b, c]
        )
        assert moment(vs, a, b, c, d) == pytest.approx(expected, abs=1e-15)


@ALL
def test_array_views_are_read_only(vs):
    with pytest.raises(ValueError):
        vs.w_array[0] = 0.0
    with pytest.raises(ValueError):
        vs.c_array[1, 0] = 2


@ALL
def test_component_tuples_match_vectors(vs):
    for axis in range(vs.d):
        assert vs.component(axis) == tuple(int(v[axis]) for v in vs.c)
    if vs.d == 2:
        assert vs.component(2) == (0,) * vs.q, "2D lattices have no z component"


@ALL
def test_velocity_sets_are_hashable_for_kernel_caching(vs):
    assert {vs: 1}[vs] == 1


def test_d2q9_direction_order_is_unchanged():
    # Kernels, tests and docs depend on this numbering.
    assert D2Q9.c == ((0, 0), (1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, 1), (-1, -1), (1, -1))
    assert D2Q9.opposite == (0, 3, 4, 1, 2, 7, 8, 5, 6)
    assert D2Q9.w == (4 / 9, *(1 / 9,) * 4, *(1 / 36,) * 4)


def test_d3q19_has_rest_axis_and_edge_links():
    lengths = [sum(map(abs, v)) for v in D3Q19.c]
    assert lengths.count(0) == 1 and lengths.count(1) == 6 and lengths.count(2) == 12
    for v, w in zip(D3Q19.c, D3Q19.w, strict=True):
        assert w == {0: 1 / 3, 1: 1 / 18, 2: 1 / 36}[sum(map(abs, v))]


def test_registry_and_defaults():
    assert LATTICES["D3Q19"] is D3Q19
    assert default_lattice(2) is D2Q9
    assert default_lattice(3) is D3Q19
    with pytest.raises(ValueError, match="dimension"):
        default_lattice(4)


def test_velocity_set_rejects_inconsistent_definitions():
    with pytest.raises(ValueError, match="weights"):
        VelocitySet("bad", 1, ((0,), (1,), (-1,)), (2 / 3, 1 / 6))
    with pytest.raises(ValueError, match="length"):
        VelocitySet("bad", 2, ((0,), (1,), (-1,)), (2 / 3, 1 / 6, 1 / 6))
    with pytest.raises(ValueError, match="opposite"):
        VelocitySet("bad", 1, ((0,), (1,), (2,)), (2 / 3, 1 / 6, 1 / 6))
    with pytest.raises(ValueError, match="sum"):
        VelocitySet("bad", 1, ((0,), (1,), (-1,)), (0.5, 0.1, 0.1))


def test_transitional_d2q9_aliases_match():
    # Kept until every physics module takes a VelocitySet (build step 3).
    assert D2Q9.component(0) == lattice.CX_TUPLE
    assert D2Q9.component(1) == lattice.CY_TUPLE
    assert D2Q9.w == lattice.W_TUPLE
    assert D2Q9.opposite == lattice.OPPOSITE_TUPLE


@ALL
@pytest.mark.parametrize("c_type", ["float", "double"])
def test_cuda_preamble_declares_types_dimension_and_constants(vs, c_type):
    source = cuda_preamble(vs, c_type)
    assert f"typedef {c_type} real_t;" in source
    assert f"#define DIM {vs.d}" in source
    assert f"#define Q {vs.q}" in source
    for name in ("cx", "cy", "cz", "opp"):
        assert f"constexpr int {name}(int i)" in source
    assert "constexpr real_t weight(int i)" in source
    for w in set(vs.w):
        assert repr(w) in source


def test_cuda_preamble_rejects_unknown_type():
    with pytest.raises(ValueError, match="real type"):
        cuda_preamble(D2Q9, "half")


@ALL
@pytest.mark.parametrize("c_type", ["float", "double"])
def test_cuda_preamble_compiles(vs, c_type):
    from tests.helpers import compile_to_ptx

    kernel = r"""
    extern "C" __global__ void probe(int* out, real_t* w) {
        int s = 0;
    #pragma unroll
        for (int i = 0; i < Q; ++i) s += cx(i) + 2 * cy(i) + 4 * cz(i) + opp(opp(i)) - i;
        out[0] = s + DIM;
        w[0] = weight(0);
    }"""
    assert "probe" in compile_to_ptx(cuda_preamble(vs, c_type) + kernel)
