"""SimulationParams validates input once, so solvers can trust it."""

import numpy as np
import pytest

from lbm.params import SimulationParams, StabilityWarning


def make(**overrides):
    base = {"nx": 8, "ny": 6, "tau": 0.8}
    base.update(overrides)
    return SimulationParams(**base)


def test_defaults_are_fully_periodic_fluid():
    p = make()
    assert p.solid.shape == (6, 8)
    assert not p.solid.any()
    assert p.wall_velocity.shape == (2, 6, 8)
    assert not p.wall_velocity.any()
    assert p.force == (0.0, 0.0)
    assert p.dtype == "float32"


def test_viscosity_follows_from_tau():
    assert make(tau=0.8).viscosity == pytest.approx((0.8 - 0.5) / 3.0)


def test_bgk_uses_one_relaxation_rate():
    omega_plus, omega_minus = make(tau=0.8, collision="bgk").relaxation_rates
    assert omega_plus == pytest.approx(1 / 0.8)
    assert omega_minus == omega_plus


@pytest.mark.parametrize("magic", [3 / 16, 1 / 4, 1 / 12])
def test_trt_rates_reproduce_magic_parameter(magic):
    omega_plus, omega_minus = make(tau=0.7, collision="trt", magic=magic).relaxation_rates
    assert omega_plus == pytest.approx(1 / 0.7)
    assert (1 / omega_plus - 0.5) * (1 / omega_minus - 0.5) == pytest.approx(magic)


@pytest.mark.parametrize("tau", [0.5, 0.3, -1.0])
def test_tau_at_or_below_half_is_rejected(tau):
    with pytest.raises(ValueError, match="tau"):
        make(tau=tau)


def test_tau_close_to_half_warns():
    with pytest.warns(StabilityWarning, match="tau"):
        make(tau=0.505)


def test_unknown_collision_is_rejected():
    with pytest.raises(ValueError, match="collision"):
        make(collision="mrt")


@pytest.mark.parametrize("magic", [0.0, -0.1])
def test_non_positive_magic_is_rejected(magic):
    with pytest.raises(ValueError, match="magic"):
        make(magic=magic)


@pytest.mark.parametrize(("nx", "ny"), [(0, 6), (8, 0), (-3, 6)])
def test_non_positive_dimensions_are_rejected(nx, ny):
    with pytest.raises(ValueError, match="nx and ny"):
        make(nx=nx, ny=ny)


def test_solid_shape_must_match_grid():
    with pytest.raises(ValueError, match="solid"):
        make(solid=np.zeros((8, 6), dtype=bool))


def test_wall_velocity_shape_must_match_grid():
    with pytest.raises(ValueError, match="wall_velocity"):
        make(wall_velocity=np.zeros((2, 8, 6)))


def test_unsupported_dtype_is_rejected():
    with pytest.raises(ValueError, match="dtype"):
        make(dtype="float16")


def test_wall_velocity_is_zeroed_on_fluid_nodes():
    solid = np.zeros((6, 8), dtype=bool)
    solid[-1, :] = True
    wall = np.ones((2, 6, 8))
    p = make(solid=solid, wall_velocity=wall)
    assert np.all(p.wall_velocity[:, -1, :] == 1.0)
    assert np.all(p.wall_velocity[:, :-1, :] == 0.0)


def test_arrays_are_read_only_copies():
    solid = np.zeros((6, 8), dtype=bool)
    p = make(solid=solid)
    solid[0, 0] = True
    assert not p.solid[0, 0], "params must not alias the caller's array"
    with pytest.raises(ValueError):
        p.solid[0, 0] = True
    with pytest.raises(ValueError):
        p.wall_velocity[0, 0, 0] = 1.0


def test_force_is_normalised_to_float_tuple():
    p = make(force=np.array([1, 2]))
    assert p.force == (1.0, 2.0)
    assert all(type(c) is float for c in p.force)


def test_fluid_mask_is_complement_of_solid():
    solid = np.zeros((6, 8), dtype=bool)
    solid[0] = True
    p = make(solid=solid)
    np.testing.assert_array_equal(p.fluid, ~solid)
