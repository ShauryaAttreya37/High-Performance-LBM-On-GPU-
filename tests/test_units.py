"""Physical <-> lattice unit conversion."""

import math

import pytest

from lbm.params import StabilityWarning
from lbm.units import UnitConverter


def test_from_reynolds_follows_definitions():
    conv = UnitConverter.from_reynolds(100.0, resolution=64, lattice_velocity=0.1)
    assert conv.lattice_viscosity == pytest.approx(0.1 * 64 / 100.0)
    assert conv.tau == pytest.approx(3.0 * conv.lattice_viscosity + 0.5)
    assert conv.reynolds == pytest.approx(100.0)


def test_physical_case_round_trips():
    # 10 cm body in a 2 m/s air stream (Re ~ 13 000). At 200 nodes per length
    # tau = 0.50225: realistic air flows push LBM straight to its stability limit.
    with pytest.warns(StabilityWarning, match="tau"):
        conv = UnitConverter(
            length=0.1, velocity=2.0, viscosity=1.5e-5, resolution=200, lattice_velocity=0.05
        )
    assert conv.reynolds == pytest.approx(0.1 * 2.0 / 1.5e-5)
    assert conv.dx == pytest.approx(0.1 / 200)
    assert conv.dt == pytest.approx(0.05 * conv.dx / 2.0)
    assert conv.velocity_to_lattice(2.0) == pytest.approx(0.05)
    assert conv.velocity_to_physical(0.05) == pytest.approx(2.0)
    assert conv.time_to_physical(1000) == pytest.approx(1000 * conv.dt)
    assert conv.steps_for_time(conv.time_to_physical(1000)) == 1000
    # Reynolds number is the same in both unit systems.
    lattice_re = conv.lattice_velocity * conv.resolution / conv.lattice_viscosity
    assert lattice_re == pytest.approx(conv.reynolds)


def test_mach_number_uses_lattice_sound_speed():
    conv = UnitConverter.from_reynolds(100.0, resolution=64, lattice_velocity=0.1)
    assert conv.mach == pytest.approx(0.1 * math.sqrt(3.0))


def test_tiny_lattice_viscosity_warns_about_stability():
    with pytest.warns(StabilityWarning, match="tau"):
        UnitConverter.from_reynolds(1e5, resolution=100, lattice_velocity=0.05)


def test_high_lattice_velocity_warns_about_compressibility():
    with pytest.warns(StabilityWarning, match="lattice velocity"):
        UnitConverter.from_reynolds(100.0, resolution=64, lattice_velocity=0.2)


@pytest.mark.parametrize(
    "field", ["length", "velocity", "viscosity", "resolution", "lattice_velocity"]
)
def test_non_positive_inputs_are_rejected(field):
    kwargs = {
        "length": 1.0,
        "velocity": 1.0,
        "viscosity": 0.01,
        "resolution": 32,
        "lattice_velocity": 0.05,
    }
    kwargs[field] = 0
    with pytest.raises(ValueError, match=field):
        UnitConverter(**kwargs)
