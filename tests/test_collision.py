"""Collision (BGK/TRT + Guo forcing) must conserve mass, inject exactly F of momentum per step."""

import numpy as np
import pytest

from lbm import lattice
from lbm.collision import collide, pre_collision_moments
from lbm.equilibrium import equilibrium
from lbm.forcing import guo_source
from lbm.params import SimulationParams
from tests.helpers import random_populations

BGK = SimulationParams(nx=1, ny=1, tau=0.8, collision="bgk").relaxation_rates
TRT = SimulationParams(nx=1, ny=1, tau=0.6, collision="trt").relaxation_rates
RATES = pytest.mark.parametrize("rates", [BGK, TRT], ids=["bgk", "trt"])
NO_FORCE = (0.0, 0.0)


def momentum(f):
    return np.tensordot(lattice.CX, f, axes=1), np.tensordot(lattice.CY, f, axes=1)


def split(f):
    """Symmetric and antisymmetric parts: f+_i = (f_i + f_opp)/2, f-_i = (f_i - f_opp)/2."""
    f_opp = f[lattice.OPPOSITE]
    return 0.5 * (f + f_opp), 0.5 * (f - f_opp)


@RATES
def test_conserves_mass_and_momentum_without_force(rates):
    f = random_populations()
    out = collide(f, *rates, NO_FORCE)
    np.testing.assert_allclose(out.sum(axis=0), f.sum(axis=0), atol=1e-15)
    for before, after in zip(momentum(f), momentum(out), strict=True):
        np.testing.assert_allclose(after, before, atol=1e-15)


@RATES
def test_equilibrium_is_a_fixed_point(rates):
    f = random_populations(noise=0.0)
    np.testing.assert_allclose(collide(f, *rates, NO_FORCE), f, atol=1e-15)


@RATES
def test_force_adds_exactly_one_force_of_momentum(rates):
    force = (2e-4, -1e-4)
    f = random_populations()
    out = collide(f, *rates, force)
    np.testing.assert_allclose(out.sum(axis=0), f.sum(axis=0), atol=1e-15)
    for a, (before, after) in enumerate(zip(momentum(f), momentum(out), strict=True)):
        np.testing.assert_allclose(after - before, force[a], atol=1e-15)


@RATES
def test_symmetric_and_antisymmetric_parts_relax_at_their_own_rates(rates):
    omega_plus, omega_minus = rates
    f = random_populations()
    drho, ux, uy = pre_collision_moments(f, NO_FORCE)
    feq_plus, feq_minus = split(equilibrium(drho, ux, uy))
    f_plus, f_minus = split(f)
    out_plus, out_minus = split(collide(f, omega_plus, omega_minus, NO_FORCE))
    np.testing.assert_allclose(out_plus - feq_plus, (1 - omega_plus) * (f_plus - feq_plus))
    np.testing.assert_allclose(out_minus - feq_minus, (1 - omega_minus) * (f_minus - feq_minus))


def test_pre_collision_velocity_includes_half_the_force():
    force = (1e-3, 2e-3)
    f = random_populations()
    drho, ux, uy = pre_collision_moments(f, force)
    rho = 1.0 + drho
    jx, jy = momentum(f)
    np.testing.assert_allclose(ux, (jx + 0.5 * force[0]) / rho)
    np.testing.assert_allclose(uy, (jy + 0.5 * force[1]) / rho)


@RATES
def test_guo_source_has_zero_mass_and_scaled_force_momentum(rates):
    omega_plus, omega_minus = rates
    force = (3e-4, 1e-4)
    rng = np.random.default_rng(3)
    ux, uy = 0.05 * rng.standard_normal((2, 4, 5))
    source = guo_source(ux, uy, force, omega_plus, omega_minus)
    np.testing.assert_allclose(source.sum(axis=0), 0.0, atol=1e-18)
    for a, s_momentum in enumerate(momentum(source)):
        np.testing.assert_allclose(s_momentum, (1 - 0.5 * omega_minus) * force[a], atol=1e-18)


def test_float32_input_stays_float32():
    f = random_populations(dtype=np.float32)
    out = collide(f, *TRT, (1e-5, 0.0))
    assert out.dtype == np.float32
