"""ReferenceSolver behaviour: initialisation, run loop, conservation, divergence watchdog."""

import numpy as np
import pytest

from lbm import BackendUnavailable, SimulationDiverged, SimulationParams, make_solver
from lbm.backends import gpu_available
from lbm.boundaries import cavity_walls


def periodic(n=8, **kwargs):
    kwargs.setdefault("dtype", "float64")
    return SimulationParams(nx=n, ny=n, tau=0.8, **kwargs)


def cavity_params(n=8, lid=0.05, **kwargs):
    solid, wall = cavity_walls(n=n, lid_velocity=lid)
    return SimulationParams(
        nx=n + 2, ny=n + 2, tau=0.7, solid=solid, wall_velocity=wall, dtype="float64", **kwargs
    )


def test_initialize_reproduces_requested_fields():
    solver = make_solver(periodic(), backend="numpy")
    rng = np.random.default_rng(0)
    rho = 1 + 0.01 * rng.standard_normal((8, 8))
    ux, uy = 0.02 * rng.standard_normal((2, 8, 8))
    solver.initialize(rho, ux, uy)
    fields = solver.macroscopic()
    np.testing.assert_allclose(fields.rho, rho, atol=1e-14)
    np.testing.assert_allclose(fields.ux, ux, atol=1e-14)
    np.testing.assert_allclose(fields.uy, uy, atol=1e-14)


def test_initialize_with_force_still_reports_requested_velocity():
    solver = make_solver(periodic(force=(1e-4, 0.0)), backend="numpy")
    solver.initialize(1.0, 0.03, 0.0)
    np.testing.assert_allclose(solver.macroscopic().ux, 0.03, atol=1e-15)


def test_run_counts_steps_and_fires_callback_every_interval():
    solver = make_solver(periodic(), backend="numpy")
    solver.initialize()
    seen = []
    solver.run(10, every=3, callback=lambda s: seen.append(s.time_step))
    assert solver.time_step == 10
    assert seen == [3, 6, 9, 10]


@pytest.mark.parametrize("kwargs", [{"steps": -1}, {"steps": 5, "every": -2}])
def test_run_rejects_negative_counts(kwargs):
    solver = make_solver(periodic(), backend="numpy")
    with pytest.raises(ValueError):
        solver.run(**kwargs)


def test_uniform_flow_is_an_exact_steady_state():
    solver = make_solver(periodic(), backend="numpy")
    solver.initialize(1.0, 0.05, -0.02)
    solver.run(50)
    fields = solver.macroscopic()
    np.testing.assert_allclose(fields.ux, 0.05, atol=1e-14)
    np.testing.assert_allclose(fields.uy, -0.02, atol=1e-14)
    np.testing.assert_allclose(fields.rho, 1.0, atol=1e-14)


def test_body_force_accelerates_periodic_fluid_by_f_per_step():
    force = (1e-5, -2e-5)
    solver = make_solver(periodic(force=force), backend="numpy")
    solver.initialize()
    solver.run(40)
    fields = solver.macroscopic()
    np.testing.assert_allclose(fields.ux, 40 * force[0], rtol=1e-10)
    np.testing.assert_allclose(fields.uy, 40 * force[1], rtol=1e-10)


@pytest.mark.parametrize("collision", ["bgk", "trt"])
def test_closed_cavity_with_moving_lid_conserves_mass(collision):
    params = cavity_params(collision=collision)
    solver = make_solver(params, backend="numpy")
    solver.initialize()
    mass0 = solver.macroscopic().total_mass
    solver.run(300)
    fields = solver.macroscopic()
    assert fields.total_mass == pytest.approx(mass0, rel=1e-13)
    assert np.abs(fields.ux).max() > 1e-3, "the lid must drive a flow"
    solid = params.solid
    assert np.all(fields.ux[solid] == 0.0) and np.all(fields.rho[solid] == 1.0)


def test_non_finite_state_raises_with_location():
    solver = make_solver(periodic(), backend="numpy")
    rho = np.ones((8, 8))
    rho[3, 5] = np.nan
    solver.initialize(rho, 0.0, 0.0)
    with pytest.raises(SimulationDiverged) as info:
        solver.run(10, check_every=5)
    assert info.value.step == 5
    assert info.value.location is not None


def test_vanishing_density_raises():
    solver = make_solver(periodic(), backend="numpy")
    rho = np.ones((8, 8))
    # A 3x3 block: after one pull step the centre node only receives populations
    # from inside the block, so its density stays negative.
    rho[1:4, 1:4] = -0.5
    solver.initialize(rho, 0.0, 0.0)
    with pytest.raises(SimulationDiverged, match="density"):
        solver.run(2, check_every=1)


def test_populations_are_a_host_copy():
    solver = make_solver(periodic(), backend="numpy")
    solver.initialize()
    f = solver.populations()
    f[:] = 123.0
    assert solver.populations().max() < 1.0


def test_macroscopic_fields_are_read_only():
    solver = make_solver(periodic(), backend="numpy")
    solver.initialize()
    fields = solver.macroscopic()
    with pytest.raises(ValueError):
        fields.ux[0, 0] = 1.0


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError, match="backend"):
        make_solver(periodic(), backend="opencl")


@pytest.mark.skipif(gpu_available(), reason="checks the error raised on machines without a GPU")
def test_cuda_backend_without_gpu_raises_clear_error():
    with pytest.raises(BackendUnavailable):
        make_solver(periodic(), backend="cuda-naive")
