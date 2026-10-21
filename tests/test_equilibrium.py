"""The discrete equilibrium must reproduce the hydrodynamic moments of a Maxwellian."""

import itertools

import numpy as np
import pytest

from lbm import lattice
from lbm.equilibrium import equilibrium
from tests.helpers import random_macroscopic

C = (lattice.CX, lattice.CY)


def test_shape_and_dtype_follow_input():
    drho, ux, uy = random_macroscopic((4, 6), dtype=np.float32)
    feq = equilibrium(drho, ux, uy)
    assert feq.shape == (9, 4, 6)
    assert feq.dtype == np.float32


def test_rest_state_is_zero_in_shifted_storage():
    zeros = np.zeros((3, 3))
    np.testing.assert_array_equal(equilibrium(zeros, zeros, zeros), 0.0)


def test_zeroth_moment_is_density_deviation():
    drho, ux, uy = random_macroscopic()
    feq = equilibrium(drho, ux, uy)
    np.testing.assert_allclose(feq.sum(axis=0), drho, atol=1e-15)


def test_first_moment_is_momentum():
    drho, ux, uy = random_macroscopic()
    rho = 1.0 + drho
    feq = equilibrium(drho, ux, uy)
    for a, u in enumerate((ux, uy)):
        momentum = np.tensordot(C[a], feq, axes=1)
        np.testing.assert_allclose(momentum, rho * u, atol=1e-15)


def test_second_moment_is_momentum_flux():
    # Shifted storage: sum_i c_a c_b (f_eq - w) = (rho - 1) cs^2 delta_ab + rho u_a u_b
    drho, ux, uy = random_macroscopic()
    rho = 1.0 + drho
    u = (ux, uy)
    feq = equilibrium(drho, ux, uy)
    for a, b in itertools.product(range(2), repeat=2):
        flux = np.tensordot(C[a] * C[b], feq, axes=1)
        expected = drho * lattice.CS2 * (a == b) + rho * u[a] * u[b]
        np.testing.assert_allclose(flux, expected, atol=1e-15)


def test_works_on_scalars_broadcast_to_arrays():
    feq = equilibrium(np.zeros(()), np.full((), 0.1), np.zeros(()))
    assert feq.shape == (9,)
    assert feq.sum() == pytest.approx(0.0, abs=1e-15)
