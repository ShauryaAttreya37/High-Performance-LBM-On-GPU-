"""Stokes flow through a periodic array of cylinders (2D) or spheres (3D).

One body sits in a periodic cell of side L; a uniform body force f drives
the fluid. At steady state and Re << 1 the dimensionless drag is known from
the dilute-array theory of Hasimoto (1959) and Sangani & Acrivos (1982),
with c the solid area/volume fraction and U the superficial velocity
(fluid velocity averaged over the whole cell):

    square array of cylinders:  F / (mu U)   = 4 pi / (-ln(c)/2 - 0.738 + c - 0.887 c^2 + 2.038 c^3)
    simple cubic of spheres:    F / (mu U a) = 6 pi / (1 - 1.7601 c^(1/3) + c - 1.5593 c^2)

F is the drag per body in a pressure-gradient-driven flow, F = G L^d (the
gradient also pushes on the body's own volume). A body force acting on fluid
nodes only is equivalent to G = f, so the comparable drag is F = f L^d.

The momentum-exchange force on the body (``lbm.forces``) is measured too. In
steady state it must balance the total force on the fluid, f times the number
of fluid nodes: exactly for plain bounce-back, which conserves momentum link
by link, and up to a small interpolation leak for Bouzidi.

What the case compares: the staircase body (plain bounce-back) sees the grid's
idea of the circle, whose area jumps as the circle cuts the grid differently;
Bouzidi sees the true surface and converges smoothly; the immersed boundary
(IBM) has no solid nodes at all, the body force acts on every node (inside
the body too) and the drag is the force the markers exert. For IBM the
"momentum exchange" column reports the marker force divided by f L^d.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import numpy as np

from lbm.backends import make_solver
from lbm.cases.common import add_backend_arguments, run_until_steady
from lbm.forces import ForceProbe
from lbm.geometry import Ball, voxelize, wall_links
from lbm.ibm import ImmersedBoundary
from lbm.lattice import CS2, default_lattice
from lbm.link_rules import bouzidi_rules
from lbm.params import SimulationParams

METHODS = ("staircase", "bouzidi", "ibm")
REYNOLDS = 0.1
"""Particle Reynolds number U (2a) / nu of the runs: well inside the Stokes regime."""


def reference_drag(d: int, c: float) -> float:
    """F / (mu U) in 2D, F / (mu U a) in 3D, from the dilute-array expansions."""
    if d == 2:
        return 4.0 * math.pi / (-0.5 * math.log(c) - 0.738 + c - 0.887 * c**2 + 2.038 * c**3)
    return 6.0 * math.pi / (1.0 - 1.7601 * c ** (1.0 / 3.0) + c - 1.5593 * c**2)


@dataclass(frozen=True)
class StokesArrayResult:
    d: int
    size: int
    radius: float
    method: str
    drag: float
    """Dimensionless drag from F = f L^d (same normalisation as ``reference_drag``)."""
    reference: float
    momentum_exchange: float
    """Momentum-exchange force along x divided by f * (number of fluid nodes)."""
    steps: int

    @property
    def error(self) -> float:
        return self.drag / self.reference - 1.0


def run_stokes_array(
    d: int = 2,
    size: int = 32,
    solid_fraction: float = 0.05,
    method: str = "bouzidi",
    tau: float = 1.0,
    collision: str = "trt",
    backend: str = "numpy",
    dtype: str = "float64",
    tol: float | None = None,
) -> StokesArrayResult:
    """Run one periodic cell to steady state and compare its drag with theory."""
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}; got {method!r}")
    lattice = default_lattice(d)
    shape = (size,) * d
    centre = (0.5 * (size - 1),) * d
    if d == 2:
        radius = math.sqrt(solid_fraction / math.pi) * size
    else:
        radius = (3.0 * solid_fraction / (4.0 * math.pi)) ** (1.0 / 3.0) * size
    body = Ball(centre, radius)
    inside = voxelize(body, shape)
    immersed = ImmersedBoundary.from_ball(body) if method == "ibm" else None
    solid = None if immersed is not None else inside
    links = wall_links(lattice, body, inside) if immersed is None else None
    rules = bouzidi_rules(lattice, links, inside) if method == "bouzidi" else None

    nu = CS2 * (tau - 0.5)
    velocity = REYNOLDS * nu / (2.0 * radius)
    reference = reference_drag(d, solid_fraction)
    length_scale = 1.0 if d == 2 else radius
    force = reference * nu * velocity * length_scale / size**d
    sizes = {"nx": size, "ny": size} | ({"nz": size} if d == 3 else {})
    params = SimulationParams(
        **sizes,
        tau=tau,
        collision=collision,
        force=(force,) + (0.0,) * (d - 1),
        solid=solid,
        dtype=dtype,
        link_rules=rules,
        immersed=immersed,
    )
    solver = make_solver(params, backend=backend)
    solver.initialize()

    def superficial(s):
        # Volume flux per cell: the body's inside counts as u = 0 (IBM leaves a little flow there).
        ux = np.where(inside, 0.0, s.macroscopic().ux)
        return np.array([ux.sum() / size**d / velocity])

    if tol is None:
        # Bouzidi does not conserve mass exactly: a closed periodic cell loses
        # ~1e-11 of its mass per step, a slow linear drift that a 1e-10
        # tolerance would never get below. 1e-7 is far below the drag errors.
        tol = 1e-7 if dtype == "float64" else 1e-6
    interval = max(2, 2 * round(size**2 / (40.0 * nu)))
    steps = run_until_steady(solver, superficial, interval=interval, tol=tol, max_steps=2_000_000)
    u_mean = float(superficial(solver)[0]) * velocity
    drag = force * size**d / (nu * u_mean * length_scale)
    if immersed is None:
        probe = ForceProbe.build(params, links)
        exchange = probe.evaluate(solver)[0] / (force * int(params.fluid.sum()))
    else:
        exchange = solver.immersed_force[0] / (force * size**d)
    return StokesArrayResult(
        d=d,
        size=size,
        radius=radius,
        method=method,
        drag=drag,
        reference=reference,
        momentum_exchange=float(exchange),
        steps=steps,
    )


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dim", type=int, default=2, choices=(2, 3))
    parser.add_argument("--sizes", type=int, nargs="+", default=[32, 48, 64, 96])
    parser.add_argument("--fraction", type=float, default=0.05)
    parser.add_argument("--tau", type=float, default=1.0)
    add_backend_arguments(parser)
    args = parser.parse_args(argv)

    print(f"Stokes array, {args.dim}D, c = {args.fraction}, tau = {args.tau}, {args.collision}")
    print(
        f"{'L':>5} {'R':>7} {'method':>10} {'drag':>9} {'error':>9} {'ME / f N':>10} {'steps':>7}"
    )
    for size in args.sizes:
        for method in METHODS:
            r = run_stokes_array(
                args.dim,
                size,
                args.fraction,
                method,
                args.tau,
                args.collision,
                args.backend,
                args.dtype,
            )
            print(
                f"{size:5d} {r.radius:7.2f} {method:>10} {r.drag:9.4f} {100 * r.error:8.3f}% "
                f"{r.momentum_exchange:10.6f} {r.steps:7d}"
            )


if __name__ == "__main__":
    main()
