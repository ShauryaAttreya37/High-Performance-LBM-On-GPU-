"""Immersed boundary method: kernel, markers, stencils, GPU equivalence."""

import math

import numpy as np
import pytest

from lbm import SimulationParams, make_solver
from lbm.boundaries import channel_walls
from lbm.geometry import circle, sphere
from lbm.ibm import ImmersedBoundary, roma


def test_roma_kernel_is_a_partition_of_unity_with_zero_first_moment():
    for x in np.linspace(0.0, 1.0, 11):
        nodes = np.arange(-3, 4)
        w = roma(nodes - x)
        assert w.sum() == pytest.approx(1.0, abs=1e-14)
        assert ((nodes - x) * w).sum() == pytest.approx(0.0, abs=1e-14)
    assert roma(np.array([1.5, 2.0])).tolist() == [0.0, 0.0]


@pytest.mark.parametrize("body", [circle(10.2, 9.7, 4.0), sphere(9.1, 8.8, 9.4, 4.0)])
def test_markers_cover_the_surface(body):
    ibm = ImmersedBoundary.from_ball(body, spacing=0.8)
    distance = np.linalg.norm(ibm.markers - body.center, axis=1)
    np.testing.assert_allclose(distance, body.radius)
    surface = 2 * math.pi * body.radius if body.d == 2 else 4 * math.pi * body.radius**2
    assert ibm.areas.sum() == pytest.approx(surface)


@pytest.mark.parametrize("body", [circle(10.2, 9.7, 4.0), sphere(9.1, 8.8, 9.4, 4.0)])
def test_stencil_weights_sum_to_one_per_marker(body):
    shape = (20, 21) if body.d == 2 else (19, 18, 20)
    nodes, weights = ImmersedBoundary.from_ball(body).stencil(shape)
    assert nodes.shape == weights.shape == (len(nodes), 3**body.d)
    np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1e-14)


def test_stencil_outside_the_domain_is_rejected():
    with pytest.raises(ValueError, match="inside the domain"):
        ImmersedBoundary.from_ball(circle(1.0, 5.0, 2.0)).stencil((11, 11))


def test_stencil_touching_a_wall_is_rejected():
    body = ImmersedBoundary.from_ball(circle(10.0, 3.5, 2.0))
    params = SimulationParams(nx=21, ny=12, tau=0.8, solid=channel_walls(21, 12), immersed=body)
    with pytest.raises(ValueError, match="solid"):
        make_solver(params)


def test_dimension_mismatch_is_rejected():
    with pytest.raises(ValueError, match="immersed"):
        SimulationParams(
            nx=8, ny=8, nz=8, tau=0.8, immersed=ImmersedBoundary.from_ball(circle(4, 4, 1))
        )


def ibm_params(d, dtype):
    if d == 2:
        sizes, solid, body = {"nx": 45, "ny": 29}, channel_walls(45, 29), circle(20.3, 13.6, 5.2)
        wall, force = np.zeros((2, 29, 45)), (1e-5, -2e-6)
        wall[0, -1] = 0.05
    else:
        sizes = {"nx": 24, "ny": 20, "nz": 16}
        solid, body = channel_walls(24, 20, 16), sphere(11.3, 9.6, 7.8, 3.6)
        wall, force = np.zeros((3, 16, 20, 24)), (1e-5, -2e-6, 3e-6)
        wall[0, :, -1] = 0.05
    return SimulationParams(
        **sizes,
        tau=0.7,
        collision="trt",
        force=force,
        solid=solid,
        wall_velocity=wall,
        dtype=dtype,
        immersed=ImmersedBoundary.from_ball(body),
    )


def run_random(params, backend):
    rng = np.random.default_rng(0)
    solver = make_solver(params, backend=backend)
    solver.initialize(
        1 + 0.01 * rng.standard_normal(params.shape),
        *(0.02 * rng.standard_normal((params.d, *params.shape))),
    )
    solver.run(40)
    return solver


@pytest.mark.gpu
@pytest.mark.parametrize("d", [2, 3])
@pytest.mark.parametrize("backend", ["cupy", "cuda-naive", "cuda-aa"])
def test_ibm_on_the_gpu_matches_the_reference(backend, d):
    for dtype, tol in (("float64", 1e-15), ("float32", 1e-7)):
        params = ibm_params(d, dtype)
        ref, gpu = run_random(params, "numpy"), run_random(params, backend)
        fluid = params.fluid
        assert np.abs(ref.populations()[:, fluid] - gpu.populations()[:, fluid]).max() < tol
        np.testing.assert_allclose(gpu.immersed_force, ref.immersed_force, rtol=10 * tol)


def test_body_at_rest_in_fluid_at_rest_feels_no_force():
    params = SimulationParams(
        nx=24,
        ny=24,
        tau=0.8,
        dtype="float64",
        immersed=ImmersedBoundary.from_ball(circle(11.5, 11.5, 4.0)),
    )
    solver = make_solver(params)
    solver.run(10)
    np.testing.assert_allclose(solver.immersed_force, 0.0, atol=1e-16)


def test_immersed_force_needs_an_immersed_body():
    with pytest.raises(AttributeError, match="immersed"):
        _ = make_solver(SimulationParams(nx=8, ny=8, tau=0.8)).immersed_force
