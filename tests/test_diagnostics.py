"""Post-processing helpers."""

import numpy as np
import pytest

from lbm.diagnostics import kinetic_energy, vorticity


def test_vorticity_of_solid_body_rotation_is_twice_the_angular_velocity():
    omega = 0.01
    y, x = np.mgrid[0:16, 0:16].astype(float)
    ux, uy = -omega * (y - 7.5), omega * (x - 7.5)
    np.testing.assert_allclose(vorticity(ux, uy), 2 * omega, atol=1e-15)


def test_vorticity_of_uniform_flow_is_zero():
    ux = np.full((6, 6), 0.1)
    np.testing.assert_allclose(vorticity(ux, np.zeros_like(ux)), 0.0, atol=1e-15)


def test_kinetic_energy_sums_half_rho_u_squared():
    rho = np.full((2, 3), 2.0)
    ux = np.full((2, 3), 0.1)
    uy = np.zeros((2, 3))
    assert kinetic_energy(rho, ux, uy) == pytest.approx(6 * 0.5 * 2.0 * 0.01)
