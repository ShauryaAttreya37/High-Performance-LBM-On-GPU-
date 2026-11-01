"""Post-processing helpers, in 2D and 3D."""

import numpy as np
import pytest

from lbm.diagnostics import MacroscopicFields, kinetic_energy, vorticity


def test_vorticity_of_2d_solid_body_rotation_is_twice_the_angular_velocity():
    omega = 0.01
    y, x = np.mgrid[0:16, 0:16].astype(float)
    np.testing.assert_allclose(vorticity(-omega * (y - 7.5), omega * (x - 7.5)), 2 * omega)


def test_vorticity_of_uniform_flow_is_zero():
    ux = np.full((6, 6), 0.1)
    np.testing.assert_allclose(vorticity(ux, np.zeros_like(ux)), 0.0, atol=1e-15)


@pytest.mark.parametrize("axis", [0, 1, 2], ids=["about-x", "about-y", "about-z"])
def test_vorticity_of_3d_solid_body_rotation_points_along_the_axis(axis):
    omega = 0.01
    z, y, x = np.mgrid[0:8, 0:9, 0:10].astype(float)
    r = np.stack([x - 4.5, y - 4.0, z - 3.5])  # position components (x, y, z)
    rotation_axis = np.eye(3)[axis] * omega
    u = np.cross(rotation_axis, r, axis=0)  # u = Omega x r
    expected = np.zeros(3)
    expected[axis] = 2 * omega
    w = vorticity(*u)
    assert w.shape == (3, 8, 9, 10)
    for k in range(3):
        np.testing.assert_allclose(w[k], expected[k], atol=1e-15)


def test_vorticity_needs_two_or_three_components():
    with pytest.raises(ValueError, match="components"):
        vorticity(np.zeros((4, 4)))


def test_kinetic_energy_sums_half_rho_u_squared_2d_and_3d():
    rho = np.full((2, 3), 2.0)
    ux = np.full((2, 3), 0.1)
    assert kinetic_energy(rho, ux, np.zeros((2, 3))) == pytest.approx(6 * 0.5 * 2.0 * 0.01)
    rho3 = np.ones((2, 2, 2))
    u3 = [np.full((2, 2, 2), v) for v in (0.1, 0.2, 0.2)]
    assert kinetic_energy(rho3, *u3) == pytest.approx(8 * 0.5 * 0.09)


def test_macroscopic_fields_expose_components_by_name():
    shape = (3, 4)
    fields = MacroscopicFields(
        rho=np.ones(shape),
        u=(np.full(shape, 0.3), np.full(shape, 0.4)),
        solid=np.zeros(shape, bool),
    )
    assert fields.d == 2
    np.testing.assert_allclose(fields.speed, 0.5)
    with pytest.raises(AttributeError, match="2D"):
        _ = fields.uz
    fields3 = MacroscopicFields(
        rho=np.ones((2, *shape)),
        u=tuple(np.full((2, *shape), v) for v in (0.1, 0.2, 0.3)),
        solid=np.zeros((2, *shape), bool),
    )
    np.testing.assert_allclose(fields3.uz, 0.3)
    with pytest.raises(ValueError):
        fields3.uz[0, 0, 0] = 1.0
