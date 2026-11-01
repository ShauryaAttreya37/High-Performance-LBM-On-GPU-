"""Force-driven flow in a circular pipe: a curved wall with an exact solution (3D).

Hagen-Poiseuille flow: a pipe of radius R along x (periodic), uniform body
force F along x. The exact steady profile is

    u(r) = F (R^2 - r^2) / (4 nu)

The wall is round but the grid is square, so this is the cleanest test of
how a method places a curved wall:

* staircase (plain bounce-back): the wall sits halfway along each link,
  wherever that happens to be, so the effective radius is off by up to half a
  node and moves irregularly as R changes. Roughly first order, with noise.
* Bouzidi: the link fraction q puts every reflection on the true circle,
  so the error falls at second order.

Bouzidi does not conserve mass: the interpolated populations do not return
exactly what left. In this flow the leak is O(Ma^2), about -1.4e-7 of the
mass per step at u_max = 0.01 and R = 4. With a fixed force density the
density drop makes u creep up forever. The steady Stokes balance
rho nu laplacian(u) = -F fixes the momentum density, rho u = F (R^2 - r^2) /
(4 nu), whatever rho is, so the case observes and compares rho u.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import numpy as np

from lbm.backends import make_solver
from lbm.cases.common import add_backend_arguments, run_until_steady
from lbm.geometry import Pipe, node_positions, voxelize, wall_links
from lbm.lattice import CS2, D3Q19
from lbm.link_rules import bouzidi_rules
from lbm.params import SimulationParams

PIPE_LENGTH = 4
METHODS = ("staircase", "bouzidi")


@dataclass(frozen=True)
class PipeResult:
    radius: float
    method: str
    l2_error: float
    """Relative L2 error of u over the fluid cross-section."""
    steps: int


def run_pipe_flow(
    radius: float = 8.3,
    method: str = "bouzidi",
    tau: float = 0.8,
    collision: str = "trt",
    backend: str = "numpy",
    dtype: str = "float64",
    max_velocity: float = 0.01,
    tol: float | None = None,
) -> PipeResult:
    """Run to steady state and compare with the Hagen-Poiseuille profile."""
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}; got {method!r}")
    side = math.ceil(2 * radius) + 4
    shape = (side, side, PIPE_LENGTH)
    wall = Pipe(center=(0.5 * (side - 1), 0.5 * (side - 1)), radius=radius)
    solid = voxelize(wall, shape)
    rules = None
    if method == "bouzidi":
        rules = bouzidi_rules(D3Q19, wall_links(D3Q19, wall, solid), solid)
    nu = CS2 * (tau - 0.5)
    force = 4.0 * nu * max_velocity / radius**2
    params = SimulationParams(
        nx=PIPE_LENGTH,
        ny=side,
        nz=side,
        tau=tau,
        collision=collision,
        force=(force, 0.0, 0.0),
        solid=solid,
        dtype=dtype,
        link_rules=rules,
    )
    solver = make_solver(params, backend=backend)
    solver.initialize()
    fluid = ~solid[..., 0]

    def section(s):
        fields = s.macroscopic()
        return (fields.rho * fields.ux).mean(axis=-1)[fluid] / max_velocity

    if tol is None:
        tol = 1e-9 if dtype == "float64" else 1e-6
    interval = max(2, 2 * round(radius**2 / (10.0 * nu)))
    steps = run_until_steady(solver, section, interval=interval, tol=tol, max_steps=2_000_000)

    positions = node_positions(shape)[..., 0, :]
    r2 = (positions[..., 1] - wall.center[0]) ** 2 + (positions[..., 2] - wall.center[1]) ** 2
    exact = force * (radius**2 - r2[fluid]) / (4.0 * nu)
    u = section(solver) * max_velocity
    error = float(np.linalg.norm(u - exact) / np.linalg.norm(exact))
    return PipeResult(radius=radius, method=method, l2_error=error, steps=steps)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--radii", type=float, nargs="+", default=[4.3, 8.3, 16.3, 32.3])
    parser.add_argument("--tau", type=float, default=0.8)
    add_backend_arguments(parser)
    args = parser.parse_args(argv)

    print(f"Pipe flow, tau = {args.tau}, {args.collision}, backend = {args.backend}")
    print(f"{'R':>7} {'method':>10} {'L2 error':>11} {'order':>7}")
    for method in METHODS:
        previous = None
        for radius in args.radii:
            r = run_pipe_flow(radius, method, args.tau, args.collision, args.backend, args.dtype)
            order = math.nan
            if previous:
                order = math.log(previous[1] / r.l2_error) / math.log(radius / previous[0])
            print(f"{radius:7.2f} {method:>10} {r.l2_error:11.3e} {order:7.2f}")
            previous = (radius, r.l2_error)


if __name__ == "__main__":
    main()
