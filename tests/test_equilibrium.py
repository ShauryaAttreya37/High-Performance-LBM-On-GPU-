"""The discrete equilibrium must reproduce the hydrodynamic moments of a Maxwellian."""

import itertools

import numpy as np
import pytest

from lbm.equilibrium import equilibrium
from tests.helpers import ALL_LATTICES, SHAPES, first_moments, random_macroscopic


@ALL_LATTICES
def test_shape_and_dtype_follow_input(vs):
    drho, u = random_macroscopic(vs, dtype=np.float32)
    feq = equilibrium(vs, drho, u)
    assert feq.shape == (vs.q, *SHAPES[vs.d])
    assert feq.dtype == np.float32


@ALL_LATTICES
def test_rest_state_is_zero_in_shifted_storage(vs):
    zeros = np.zeros(SHAPES[vs.d])
    np.testing.assert_array_equal(equilibrium(vs, zeros, (zeros,) * vs.d), 0.0)


@ALL_LATTICES
def test_zeroth_moment_is_density_deviation(vs):
    drho, u = random_macroscopic(vs)
    np.testing.assert_allclose(equilibrium(vs, drho, u).sum(axis=0), drho, atol=1e-15)


@ALL_LATTICES
def test_first_moment_is_momentum(vs):
    drho, u = random_macroscopic(vs)
    rho = 1.0 + drho
    for j_a, u_a in zip(first_moments(vs, equilibrium(vs, drho, u)), u, strict=True):
        np.testing.assert_allclose(j_a, rho * u_a, atol=1e-15)


@ALL_LATTICES
def test_second_moment_is_momentum_flux(vs):
    # Shifted storage: sum_i c_a c_b (f_eq - w) = (rho - 1) cs^2 delta_ab + rho u_a u_b
    drho, u = random_macroscopic(vs)
    rho = 1.0 + drho
    feq = equilibrium(vs, drho, u)
    c = vs.c_array.astype(float)
    for a, b in itertools.product(range(vs.d), repeat=2):
        flux = np.tensordot(c[:, a] * c[:, b], feq, axes=1)
        expected = drho * vs.cs2 * (a == b) + rho * u[a] * u[b]
        np.testing.assert_allclose(flux, expected, atol=1e-15)


@ALL_LATTICES
def test_works_on_scalars_broadcast_to_arrays(vs):
    u = (np.full((), 0.1), *(np.zeros(()),) * (vs.d - 1))
    feq = equilibrium(vs, np.zeros(()), u)
    assert feq.shape == (vs.q,)
    assert feq.sum() == pytest.approx(0.0, abs=1e-15)
