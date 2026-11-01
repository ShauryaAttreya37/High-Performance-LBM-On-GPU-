"""Collision (BGK/TRT + Guo forcing) must conserve mass, inject exactly F of momentum per step."""

import numpy as np
import pytest

from lbm.collision import collide, pre_collision_moments
from lbm.equilibrium import equilibrium
from lbm.forcing import guo_source
from lbm.params import SimulationParams
from tests.helpers import ALL_LATTICES, SHAPES, first_moments, random_populations

BGK = SimulationParams(nx=1, ny=1, tau=0.8, collision="bgk").relaxation_rates
TRT = SimulationParams(nx=1, ny=1, tau=0.6, collision="trt").relaxation_rates
RATES = pytest.mark.parametrize("rates", [BGK, TRT], ids=["bgk", "trt"])


def no_force(vs):
    return (0.0,) * vs.d


def some_force(vs):
    return (2e-4, -1e-4, 5e-5)[: vs.d]


def split(vs, f):
    """Symmetric and antisymmetric parts: f+_i = (f_i + f_opp)/2, f-_i = (f_i - f_opp)/2."""
    f_opp = f[list(vs.opposite)]
    return 0.5 * (f + f_opp), 0.5 * (f - f_opp)


@ALL_LATTICES
@RATES
def test_conserves_mass_and_momentum_without_force(vs, rates):
    f = random_populations(vs)
    out = collide(vs, f, *rates, no_force(vs))
    np.testing.assert_allclose(out.sum(axis=0), f.sum(axis=0), atol=1e-15)
    for before, after in zip(first_moments(vs, f), first_moments(vs, out), strict=True):
        np.testing.assert_allclose(after, before, atol=1e-15)


@ALL_LATTICES
@RATES
def test_equilibrium_is_a_fixed_point(vs, rates):
    f = random_populations(vs, noise=0.0)
    np.testing.assert_allclose(collide(vs, f, *rates, no_force(vs)), f, atol=1e-15)


@ALL_LATTICES
@RATES
def test_force_adds_exactly_one_force_of_momentum(vs, rates):
    force = some_force(vs)
    f = random_populations(vs)
    out = collide(vs, f, *rates, force)
    np.testing.assert_allclose(out.sum(axis=0), f.sum(axis=0), atol=1e-15)
    moments = zip(first_moments(vs, f), first_moments(vs, out), force, strict=True)
    for before, after, f_a in moments:
        np.testing.assert_allclose(after - before, f_a, atol=1e-15)


@ALL_LATTICES
@RATES
def test_symmetric_and_antisymmetric_parts_relax_at_their_own_rates(vs, rates):
    omega_plus, omega_minus = rates
    f = random_populations(vs)
    drho, u = pre_collision_moments(vs, f, no_force(vs))
    feq_plus, feq_minus = split(vs, equilibrium(vs, drho, u))
    f_plus, f_minus = split(vs, f)
    out_plus, out_minus = split(vs, collide(vs, f, omega_plus, omega_minus, no_force(vs)))
    np.testing.assert_allclose(out_plus - feq_plus, (1 - omega_plus) * (f_plus - feq_plus))
    np.testing.assert_allclose(out_minus - feq_minus, (1 - omega_minus) * (f_minus - feq_minus))


@ALL_LATTICES
def test_pre_collision_velocity_includes_half_the_force(vs):
    force = some_force(vs)
    f = random_populations(vs)
    drho, u = pre_collision_moments(vs, f, force)
    rho = 1.0 + drho
    for u_a, j_a, f_a in zip(u, first_moments(vs, f), force, strict=True):
        np.testing.assert_allclose(u_a, (j_a + 0.5 * f_a) / rho)


@ALL_LATTICES
@RATES
def test_guo_source_has_zero_mass_and_scaled_force_momentum(vs, rates):
    omega_plus, omega_minus = rates
    force = some_force(vs)
    rng = np.random.default_rng(3)
    u = tuple(0.05 * rng.standard_normal(SHAPES[vs.d]) for _ in range(vs.d))
    source = guo_source(vs, u, force, omega_plus, omega_minus)
    np.testing.assert_allclose(source.sum(axis=0), 0.0, atol=1e-18)
    for s_a, f_a in zip(first_moments(vs, source), force, strict=True):
        np.testing.assert_allclose(s_a, (1 - 0.5 * omega_minus) * f_a, atol=1e-18)


@ALL_LATTICES
def test_float32_input_stays_float32(vs):
    f = random_populations(vs, dtype=np.float32)
    assert collide(vs, f, *TRT, some_force(vs)).dtype == np.float32
