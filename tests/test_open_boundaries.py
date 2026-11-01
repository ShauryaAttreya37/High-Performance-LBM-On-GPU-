"""Zou-He velocity inlet and pressure outlet as link rules."""

import numpy as np
import pytest

from lbm import SimulationParams, make_solver
from lbm.boundaries import apply_bounce_back, channel_walls
from lbm.lattice import default_lattice
from lbm.link_rules import evaluate_affine
from lbm.open_boundaries import open_channel
from lbm.streaming import pull_stream


def small_open_channel(d, rho_out=1.003, dtype="float64"):
    lattice = default_lattice(d)
    nx, ny, nz = 12, 9, 5
    solid = channel_walls(nx, ny) if d == 2 else channel_walls(nx, ny, nz)
    inflow = np.zeros((d, *solid.shape[:-1]))
    inflow[0], inflow[1] = 0.04, 0.01
    if d == 3:
        inflow[2] = -0.02
    solid, rules = open_channel(lattice, solid, inflow, rho_out=rho_out)
    sizes = {"nx": nx, "ny": ny} | ({"nz": nz} if d == 3 else {})
    params = SimulationParams(
        **sizes, tau=0.7, collision="trt", solid=solid, dtype=dtype, link_rules=rules
    )
    return params, inflow


def random_solver(params, backend="numpy"):
    rng = np.random.default_rng(1)
    solver = make_solver(params, backend=backend)
    solver.initialize(
        1 + 0.01 * rng.standard_normal(params.shape),
        *(0.03 * rng.standard_normal((params.d, *params.shape))),
    )
    return solver


@pytest.mark.parametrize("d", [2, 3])
def test_zou_he_rules_impose_inlet_velocity_and_outlet_density_exactly(d):
    params, inflow = small_open_channel(d)
    lattice, shape = params.lattice, params.shape
    solver = random_solver(params)
    f = solver._f
    streamed = apply_bounce_back(lattice, pull_stream(lattice, f), f, solver._links)
    streamed.reshape(-1)[solver._rules.target] = evaluate_affine(
        solver._rules.values, f.reshape(-1)
    )
    rho = (1 + streamed.sum(0)).reshape(-1)
    u = np.tensordot(np.asarray(lattice.c, float).T, streamed, 1).reshape(d, -1) / rho
    nodes = np.unique(params.link_rules.node)
    column = np.unravel_index(nodes, shape)[-1]
    inlet, outlet = nodes[column == 1], nodes[column == shape[-1] - 2]
    assert len(inlet) and len(outlet)
    np.testing.assert_allclose(rho[outlet], 1.003, atol=1e-15)
    for a in range(1, d):
        np.testing.assert_allclose(u[a][outlet], 0.0, atol=1e-15)
    for a in range(d):
        np.testing.assert_allclose(u[a][inlet], inflow[a].reshape(-1)[0], atol=1e-15)


@pytest.mark.gpu
@pytest.mark.parametrize("d", [2, 3])
@pytest.mark.parametrize("backend", ["cupy", "cuda-naive", "cuda-aa"])
def test_open_boundaries_on_the_gpu_match_the_reference(backend, d):
    params, _ = small_open_channel(d)
    ref, gpu = random_solver(params), random_solver(params, backend)
    for s in (ref, gpu):
        s.run(40)
    fluid = params.fluid
    assert np.abs(ref.populations()[:, fluid] - gpu.populations()[:, fluid]).max() < 1e-15


def test_parabolic_inflow_stays_poiseuille_through_the_channel():
    """Measured (U = 0.01, H = 16, L = 64): 1.1e-3 U at the inlet, 2.7e-3 U mid-channel.

    The interior error is weak compressibility, not the boundaries: the pressure
    drop makes rho fall by 0.6 % along the channel, and u = flux / rho follows.
    """
    lattice = default_lattice(2)
    height, length, u_max = 16, 64, 0.01
    y = np.arange(height) + 0.5
    profile = 4 * u_max * y * (height - y) / height**2
    inflow = np.zeros((2, height + 2))
    inflow[0, 1:-1] = profile
    solid, rules = open_channel(lattice, channel_walls(length, height + 2), inflow)
    params = SimulationParams(
        nx=length,
        ny=height + 2,
        tau=0.8,
        collision="trt",
        solid=solid,
        dtype="float64",
        link_rules=rules,
    )
    solver = make_solver(params)
    solver.run(16000)
    mass = solver.macroscopic().total_mass
    solver.run(2000)
    fields = solver.macroscopic()
    assert fields.total_mass == pytest.approx(mass, rel=1e-10)
    assert np.abs(fields.ux[1:-1, 1] - profile).max() < 2e-3 * u_max
    assert np.abs(fields.ux[1:-1, length // 2] - profile).max() < 5e-3 * u_max
    np.testing.assert_allclose(fields.rho[2:-2, -2], 1.0, atol=1e-12)


def test_open_channel_needs_room_for_both_layers():
    with pytest.raises(ValueError, match="at least 4"):
        open_channel(default_lattice(2), np.zeros((5, 3), bool), 0.01)
