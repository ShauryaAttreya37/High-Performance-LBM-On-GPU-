"""ReferenceSolver in 2D and 3D: initialisation, run loop, conservation, divergence watchdog."""

import numpy as np
import pytest

from lbm import BackendUnavailable, SimulationDiverged, SimulationParams, make_solver
from lbm.backends import gpu_available
from lbm.boundaries import cavity_walls

DIMS = pytest.mark.parametrize("d", [2, 3], ids=["2d", "3d"])
N = 8


def grid(d, n=N):
    return (n,) * d


def periodic(d=2, n=N, **kwargs):
    kwargs.setdefault("dtype", "float64")
    if d == 3:
        kwargs["nz"] = n
    return SimulationParams(nx=n, ny=n, tau=0.8, **kwargs)


def cavity_params(d=2, n=6, lid=0.05, **kwargs):
    solid, wall = cavity_walls(n=n, lid_velocity=lid, dimensions=d)
    size = n + 2
    if d == 3:
        kwargs["nz"] = size
    return SimulationParams(
        nx=size, ny=size, tau=0.7, solid=solid, wall_velocity=wall, dtype="float64", **kwargs
    )


@DIMS
def test_initialize_reproduces_requested_fields(d):
    solver = make_solver(periodic(d), backend="numpy")
    rng = np.random.default_rng(0)
    rho = 1 + 0.01 * rng.standard_normal(grid(d))
    u = 0.02 * rng.standard_normal((d, *grid(d)))
    solver.initialize(rho, *u)
    fields = solver.macroscopic()
    np.testing.assert_allclose(fields.rho, rho, atol=1e-14)
    for got, want in zip(fields.u, u, strict=True):
        np.testing.assert_allclose(got, want, atol=1e-14)


@DIMS
def test_initialize_with_force_still_reports_requested_velocity(d):
    force = (1e-4,) + (0.0,) * (d - 1)
    solver = make_solver(periodic(d, force=force), backend="numpy")
    solver.initialize(1.0, 0.03)
    np.testing.assert_allclose(solver.macroscopic().ux, 0.03, atol=1e-15)


def test_initialize_rejects_z_velocity_in_2d():
    solver = make_solver(periodic(2), backend="numpy")
    with pytest.raises(ValueError, match="2D"):
        solver.initialize(1.0, 0.0, 0.0, 0.01)


def test_run_counts_steps_and_fires_callback_every_interval():
    solver = make_solver(periodic(), backend="numpy")
    seen = []
    solver.run(10, every=3, callback=lambda s: seen.append(s.time_step))
    assert solver.time_step == 10
    assert seen == [3, 6, 9, 10]


@pytest.mark.parametrize("kwargs", [{"steps": -1}, {"steps": 5, "every": -2}])
def test_run_rejects_negative_counts(kwargs):
    solver = make_solver(periodic(), backend="numpy")
    with pytest.raises(ValueError):
        solver.run(**kwargs)


@DIMS
def test_uniform_flow_is_an_exact_steady_state(d):
    velocity = (0.05, -0.02, 0.01)[:d]
    solver = make_solver(periodic(d), backend="numpy")
    solver.initialize(1.0, *velocity)
    solver.run(50)
    fields = solver.macroscopic()
    for got, want in zip(fields.u, velocity, strict=True):
        np.testing.assert_allclose(got, want, atol=1e-14)
    np.testing.assert_allclose(fields.rho, 1.0, atol=1e-14)


@DIMS
def test_body_force_accelerates_periodic_fluid_by_f_per_step(d):
    force = (1e-5, -2e-5, 3e-5)[:d]
    solver = make_solver(periodic(d, force=force), backend="numpy")
    solver.run(40)
    for got, f_a in zip(solver.macroscopic().u, force, strict=True):
        np.testing.assert_allclose(got, 40 * f_a, rtol=1e-10)


@DIMS
@pytest.mark.parametrize("collision", ["bgk", "trt"])
def test_closed_cavity_with_moving_lid_conserves_mass(d, collision):
    params = cavity_params(d, collision=collision)
    solver = make_solver(params, backend="numpy")
    mass0 = solver.macroscopic().total_mass
    solver.run(300)
    fields = solver.macroscopic()
    assert fields.total_mass == pytest.approx(mass0, rel=1e-13)
    assert np.abs(fields.ux).max() > 1e-3, "the lid must drive a flow"
    assert np.all(fields.ux[params.solid] == 0.0) and np.all(fields.rho[params.solid] == 1.0)


@DIMS
def test_non_finite_state_raises_with_location(d):
    solver = make_solver(periodic(d), backend="numpy")
    rho = np.ones(grid(d))
    rho[(3, 5, 2)[:d]] = np.nan
    solver.initialize(rho)
    with pytest.raises(SimulationDiverged) as info:
        solver.run(10, check_every=5)
    assert info.value.step == 5
    assert len(info.value.location) == d


@DIMS
def test_vanishing_density_raises(d):
    solver = make_solver(periodic(d), backend="numpy")
    rho = np.ones(grid(d))
    # A 3^d block: after one pull step the centre node only receives populations
    # from inside the block, so its density stays negative.
    rho[(slice(1, 4),) * d] = -0.5
    solver.initialize(rho)
    with pytest.raises(SimulationDiverged, match="density"):
        solver.run(2, check_every=1)


def test_populations_are_a_host_copy():
    solver = make_solver(periodic(), backend="numpy")
    f = solver.populations()
    f[:] = 123.0
    assert solver.populations().max() < 1.0


@DIMS
def test_macroscopic_fields_are_read_only(d):
    fields = make_solver(periodic(d), backend="numpy").macroscopic()
    with pytest.raises(ValueError):
        fields.ux[(0,) * d] = 1.0
    with pytest.raises(ValueError):
        fields.rho[(0,) * d] = 1.0


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError, match="backend"):
        make_solver(periodic(), backend="opencl")


@pytest.mark.skipif(gpu_available(), reason="checks the error raised on machines without a GPU")
def test_cuda_backend_without_gpu_raises_clear_error():
    with pytest.raises(BackendUnavailable):
        make_solver(periodic(), backend="cuda-naive")
