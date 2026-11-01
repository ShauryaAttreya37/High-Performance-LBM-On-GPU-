"""Geometry (shapes, link fractions), link rules and momentum-exchange forces."""

import math

import numpy as np
import pytest

from lbm import SimulationParams, make_solver
from lbm.boundaries import channel_walls
from lbm.forces import ForceProbe
from lbm.geometry import Pipe, WallLinks, circle, sphere, voxelize, wall_links
from lbm.lattice import D2Q9, D3Q19, default_lattice
from lbm.link_rules import LAYOUTS, AffineSet, LinkRules, bouzidi_rules, compile_affine

GPU = [pytest.param(b, marks=pytest.mark.gpu) for b in ("cupy", "cuda-naive", "cuda-aa")]

# ---------------------------------------------------------------- geometry


def test_ball_ray_fraction_hits_the_surface():
    body = circle(0.0, 0.0, 1.5)
    t = body.ray_fraction(np.array([[2.0, 0.0], [0.0, -2.0]]), np.array([[-1.0, 0.0], [0.0, 1.0]]))
    np.testing.assert_allclose(t, [0.5, 0.5])
    diagonal = body.ray_fraction(np.array([[2.0, 2.0]]), np.array([[-1.0, -1.0]]))
    np.testing.assert_allclose(diagonal, [2.0 - 1.5 / math.sqrt(2.0)])


def test_pipe_ray_fraction_uses_the_exit_point():
    pipe = Pipe((0.0, 0.0), 2.5)
    t = pipe.ray_fraction(np.array([[0.0, 2.0, 0.0]]), np.array([[1.0, 1.0, 0.0]]))
    np.testing.assert_allclose(t, [0.5])
    assert pipe.contains(np.array([0.0, 3.0, 0.0])) and not pipe.contains(np.array([5.0, 0, 0]))


def test_voxelize_marks_nodes_inside():
    mask = voxelize(circle(5.0, 4.0, 1.0), (9, 11))
    assert mask.sum() == 5 and mask[4, 5] and mask[3, 5] and mask[4, 6] and not mask[3, 4]


@pytest.mark.parametrize("body", [circle(10.3, 8.6, 4.2), sphere(8.4, 7.1, 6.6, 3.7)])
def test_wall_links_start_in_fluid_end_in_the_body_and_measure_q(body):
    lattice = default_lattice(body.d)
    shape = (17, 21) if body.d == 2 else (14, 15, 17)
    solid = voxelize(body, shape)
    links = wall_links(lattice, body, solid)
    assert len(links) > 0
    assert not solid.ravel()[links.node].any()
    assert np.all((links.q > 0) & (links.q <= 1))
    positions = np.stack(np.unravel_index(links.node, shape)[::-1], axis=-1).astype(float)
    c = np.asarray(lattice.c)[links.direction]
    wall = positions - links.q[:, None] * c
    np.testing.assert_allclose(np.linalg.norm(wall - body.center, axis=1), body.radius)
    assert body.contains(positions - c).all()


def test_wall_links_require_the_body_to_be_solid():
    body = circle(5.0, 5.0, 2.0)
    with pytest.raises(ValueError, match="solid"):
        wall_links(D2Q9, body, np.zeros((11, 11), dtype=bool))


# ---------------------------------------------------------------- rules


def disk_case(nx=33, ny=21):
    body = circle(15.3, 10.4, 4.6)
    solid = channel_walls(nx, ny) | voxelize(body, (ny, nx))
    return body, solid, wall_links(D2Q9, body, solid)


def test_bouzidi_coefficients_interpolate_and_sum_to_one():
    _, solid, links = disk_case()
    rules = bouzidi_rules(D2Q9, links, solid)
    coeff = rules.values.coeff
    np.testing.assert_allclose(coeff.sum(axis=1), 1.0)
    for r, (x, j) in enumerate(zip(rules.node, rules.direction, strict=True)):
        q = links.q[(links.node == x) & (links.direction == j)][0]
        expected = 2 * q if q < 0.5 else 0.5 / q
        assert coeff[r, 0] == pytest.approx(expected)


@pytest.mark.parametrize("backend", ["numpy", *GPU])
def test_bouzidi_with_q_one_half_equals_plain_bounce_back(backend):
    _, solid, links = disk_case()
    half = WallLinks(links.node, links.direction, np.full(len(links), 0.5))
    common = {"nx": 33, "ny": 21, "tau": 0.7, "collision": "trt", "force": (1e-5, 0.0)}
    common |= {"solid": solid, "dtype": "float64"}
    plain = make_solver(SimulationParams(**common), backend=backend)
    rules = bouzidi_rules(D2Q9, half, solid)
    ruled = make_solver(SimulationParams(**common, link_rules=rules), backend=backend)
    for s in (plain, ruled):
        s.run(30)
    np.testing.assert_allclose(ruled.populations(), plain.populations(), rtol=0, atol=1e-16)


def test_rules_are_rejected_off_fluid_to_resting_solid_links():
    _, solid, links = disk_case()
    rules = bouzidi_rules(D2Q9, links, solid)
    with pytest.raises(ValueError, match="solid"):
        SimulationParams(nx=33, ny=21, tau=0.7, link_rules=rules)
    moving = np.zeros((2, 21, 33))
    moving[0] = 0.01
    with pytest.raises(ValueError, match="at rest"):
        SimulationParams(nx=33, ny=21, tau=0.7, solid=solid, wall_velocity=moving, link_rules=rules)


def test_duplicate_rules_on_one_link_are_rejected():
    values = AffineSet.from_lists([[(1, 0, 1.0)], [(1, 0, 1.0)]])
    with pytest.raises(ValueError, match="more than one"):
        LinkRules(np.array([5, 5]), np.array([3, 3]), values)


def test_unknown_layout_is_rejected():
    _, solid, links = disk_case()
    rules = bouzidi_rules(D2Q9, links, solid)
    with pytest.raises(ValueError, match="layout"):
        compile_affine(D2Q9, rules.values, "zigzag", solid, np.zeros((2, *solid.shape)))
    assert "aa_even" in LAYOUTS


def bouzidi_params(d, dtype):
    """Walls, a moving wall, a Bouzidi body and a force, 2D or 3D (not block-aligned)."""
    if d == 2:
        nx, ny = 45, 29
        solid, body = channel_walls(nx, ny), circle(20.3, 13.6, 5.2)
        wall = np.zeros((2, ny, nx))
        wall[0, -1, :] = 0.05
        sizes, force = {"nx": nx, "ny": ny}, (1e-5, -2e-6)
    else:
        nx, ny, nz = 24, 20, 16
        solid, body = channel_walls(nx, ny, nz), sphere(11.3, 9.6, 7.8, 4.1)
        solid[0] = solid[-1] = True
        wall = np.zeros((3, nz, ny, nx))
        wall[0, :, -1, :] = 0.05
        wall[2, :, -1, :] = -0.02
        sizes, force = {"nx": nx, "ny": ny, "nz": nz}, (1e-5, -2e-6, 3e-6)
    lattice = default_lattice(d)
    solid = solid | voxelize(body, solid.shape)
    links = wall_links(lattice, body, solid)
    params = SimulationParams(
        **sizes,
        tau=0.7,
        collision="trt",
        force=force,
        solid=solid,
        wall_velocity=wall,
        dtype=dtype,
        link_rules=bouzidi_rules(lattice, links, solid),
    )
    return params, links


def run_from_random_state(params, backend, steps=40):
    rng = np.random.default_rng(0)
    state = (1 + 0.01 * rng.standard_normal(params.shape),)
    state += tuple(0.02 * rng.standard_normal((params.d, *params.shape)))
    solver = make_solver(params, backend=backend)
    solver.initialize(*state)
    solver.run(steps)
    return solver


@pytest.mark.gpu
@pytest.mark.parametrize("d", [2, 3])
@pytest.mark.parametrize("backend", ["cupy", "cuda-naive", "cuda-aa"])
def test_bouzidi_on_the_gpu_matches_the_reference(backend, d):
    """Exercises every layout of the rule compiler (pull, AA odd and even) and the kernels."""
    for dtype, tol in (("float64", 1e-15), ("float32", 1e-7)):
        params, _ = bouzidi_params(d, dtype)
        ref = run_from_random_state(params, "numpy")
        gpu = run_from_random_state(params, backend)
        fluid = params.fluid
        diff = np.abs(ref.populations()[:, fluid] - gpu.populations()[:, fluid]).max()
        assert diff < tol, (dtype, diff)


def test_bouzidi_changes_the_flow():
    params, _ = bouzidi_params(2, "float64")
    plain = SimulationParams(
        nx=45,
        ny=29,
        tau=0.7,
        collision="trt",
        force=params.force,
        solid=params.solid,
        wall_velocity=params.wall_velocity,
        dtype="float64",
    )
    a, b = run_from_random_state(params, "numpy"), run_from_random_state(plain, "numpy")
    assert np.abs(a.populations() - b.populations()).max() > 1e-4


# ---------------------------------------------------------------- forces


def periodic_disk_params(rules: bool):
    size, body = 24, circle(11.5, 11.5, 3.3)
    solid = voxelize(body, (size, size))
    links = wall_links(D2Q9, body, solid)
    params = SimulationParams(
        nx=size,
        ny=size,
        tau=1.0,
        collision="trt",
        force=(1e-6, 0.0),
        solid=solid,
        dtype="float64",
        link_rules=bouzidi_rules(D2Q9, links, solid) if rules else None,
    )
    return params, links


@pytest.mark.parametrize("rules", [False, True], ids=["plain", "bouzidi"])
def test_steady_momentum_exchange_balances_the_body_force(rules):
    params, links = periodic_disk_params(rules)
    solver = make_solver(params)
    solver.run(6000)
    force = ForceProbe.build(params, links).evaluate(solver)
    total = params.force[0] * params.fluid.sum()
    assert force[0] == pytest.approx(total, rel=1e-9)
    assert abs(force[1]) < 1e-12 * total


def test_force_on_a_body_at_rest_in_fluid_at_rest_is_zero():
    params, links = periodic_disk_params(True)
    quiet = SimulationParams(
        nx=params.nx,
        ny=params.ny,
        tau=1.0,
        solid=params.solid,
        dtype="float64",
        link_rules=params.link_rules,
    )
    solver = make_solver(quiet)
    np.testing.assert_allclose(ForceProbe.build(quiet, links).evaluate(solver), 0.0, atol=1e-15)


@pytest.mark.gpu
@pytest.mark.parametrize("d", [2, 3])
@pytest.mark.parametrize("backend", ["cupy", "cuda-naive", "cuda-aa"])
def test_force_probe_reads_every_layout_the_same(backend, d):
    params, links = bouzidi_params(d, "float64")
    probe = ForceProbe.build(params, links)
    ref = probe.evaluate(run_from_random_state(params, "numpy"))
    gpu = probe.evaluate(run_from_random_state(params, backend))
    np.testing.assert_allclose(gpu, ref, rtol=1e-12, atol=1e-16)


def test_probe_refuses_a_solver_built_from_other_params():
    params, links = periodic_disk_params(False)
    other, _ = periodic_disk_params(False)
    with pytest.raises(ValueError, match="different"):
        ForceProbe.build(params, links).evaluate(make_solver(other))


def test_d3q19_links_into_a_sphere_cover_all_neighbour_directions():
    body = sphere(7.2, 7.1, 6.9, 3.4)
    solid = voxelize(body, (14, 15, 15))
    links = wall_links(D3Q19, body, solid)
    assert set(links.direction.tolist()) == set(range(1, 19))
