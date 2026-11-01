"""Drag on a sphere in uniform flow (3D), compared with the Schiller-Naumann fit.

A sphere of diameter D sits 5 D behind a Zou-He velocity inlet with uniform
inflow U, in a box of cross-section W x W (periodic in y and z) that ends
in a Zou-He pressure outlet. With W = 10 D the sphere blocks 0.8 % of the
cross-section, and the periodic images are 10 D apart.

The reference is the standard empirical fit for a single sphere,

    C_D = 24 / Re (1 + 0.15 Re^0.687),        Re = U D / nu <= 800,

which reproduces measurements to about +-5 %, and the fit of Clift, Grace &
Weber (1978) for 20 <= Re <= 260,

    C_D = 24 / Re (1 + 0.1935 Re^0.6305).

They agree at Re = 100 (1.092 vs 1.088) but differ by 5 % at Re = 20 (2.61
vs 2.735), which sets how tightly this case can be judged.
C_D = 2 F_x / (rho U^2 pi D^2 / 4).

Measured (Bouzidi, TRT, float32): Re = 20: C_D = 2.769 (N = 10), 2.770
(N = 16); a 15 D wide box gives 2.759, so confinement adds ~0.4 %.
Re = 100, N = 16: 1.128. Staircase 2.82 / 1.151, IBM 2.96 / 1.184 (first
order: 3.11 at N = 10).

Two things make this a looser check than the cylinder benchmark:

* Confinement. The periodic images 10 D apart slow the flow past the sphere
  less than walls would, but still raise the drag: in Stokes flow the same
  spacing adds 16 % (Hasimoto); inertia shrinks the effect as Re grows.
* Acoustics. With no walls, a plane sound wave bouncing between the two
  reflecting Zou-He ends meets no shear and is barely damped; it leaves a
  ~1 % ripple on the drag that never dies out. The run therefore starts
  from the uniform inflow (no impulsive start) and reports the drag averaged
  over the second half of a fixed run instead of waiting for it to settle.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import numpy as np

from lbm.backends import make_solver
from lbm.cases.common import add_backend_arguments
from lbm.forces import ForceProbe
from lbm.geometry import sphere, voxelize, wall_links
from lbm.ibm import ImmersedBoundary
from lbm.lattice import CS2, D3Q19
from lbm.link_rules import LinkRules, bouzidi_rules
from lbm.open_boundaries import open_channel
from lbm.params import SimulationParams

METHODS = ("bouzidi", "staircase", "ibm")
UPSTREAM, DOWNSTREAM, WIDTH = 5, 15, 10
"""Box size in diameters."""


def schiller_naumann(reynolds: float) -> float:
    return 24.0 / reynolds * (1.0 + 0.15 * reynolds**0.687)


def clift_grace_weber(reynolds: float) -> float:
    return 24.0 / reynolds * (1.0 + 0.1935 * reynolds**0.6305)


@dataclass(frozen=True)
class SphereResult:
    n: int
    reynolds: float
    method: str
    cd: float
    reference: float
    """Schiller-Naumann."""
    reference_cgw: float
    """Clift-Grace-Weber."""
    steps: int

    @property
    def error(self) -> float:
        return self.cd / self.reference - 1.0


def run_sphere_drag(
    n: int = 10,
    reynolds: float = 20.0,
    method: str = "bouzidi",
    velocity: float = 0.05,
    collision: str = "trt",
    backend: str = "cuda-aa",
    dtype: str = "float32",
    convective_times: int = 60,
    samples_per_time: int = 10,
) -> SphereResult:
    """Run ``convective_times`` times D / U and average C_D over the second half."""
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}; got {method!r}")
    width = WIDTH * n
    length = (UPSTREAM + DOWNSTREAM) * n + 3
    shape = (width, width, length)
    centre = (1.0 + UPSTREAM * n, 0.5 * (width - 1), 0.5 * (width - 1))
    body = sphere(*centre, 0.5 * n)
    immersed = ImmersedBoundary.from_ball(body) if method == "ibm" else None
    solid = np.zeros(shape, dtype=bool)
    if immersed is None:
        solid |= voxelize(body, shape)
    inflow = np.zeros((3, width, width))
    inflow[0] = velocity
    solid, rules = open_channel(D3Q19, solid, inflow)
    links = wall_links(D3Q19, body, solid) if immersed is None else None
    if method == "bouzidi":
        rules = LinkRules.concat(rules, bouzidi_rules(D3Q19, links, solid))
    nu = velocity * n / reynolds
    params = SimulationParams(
        nx=length,
        ny=width,
        nz=width,
        tau=nu / CS2 + 0.5,
        collision=collision,
        solid=solid,
        dtype=dtype,
        link_rules=rules,
        immersed=immersed,
    )
    del solid
    solver = make_solver(params, backend=backend)
    probe = None if immersed is not None else ForceProbe.build(params, links)
    scale = 0.5 * velocity**2 * math.pi * n**2 / 4.0

    solver.initialize(1.0, np.where(params.solid, 0.0, velocity), 0.0, 0.0)
    samples = []

    def record(s):
        force = s.immersed_force if probe is None else probe.evaluate(s)
        samples.append(force[0] / scale)

    every = max(2, 2 * round(n / velocity / samples_per_time / 2))
    steps = every * samples_per_time * convective_times
    solver.run(steps, every=every, callback=record)
    cd = float(np.mean(samples[len(samples) // 2 :]))
    return SphereResult(
        n, reynolds, method, cd, schiller_naumann(reynolds), clift_grace_weber(reynolds), steps
    )


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sizes", type=int, nargs="+", default=[10, 16])
    parser.add_argument("--re", type=float, nargs="+", default=[20.0, 100.0])
    parser.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    add_backend_arguments(parser)
    parser.set_defaults(backend="cuda-aa", dtype="float32")
    args = parser.parse_args(argv)
    print(f"Sphere drag, box {UPSTREAM + DOWNSTREAM} D x {WIDTH} D x {WIDTH} D, {args.collision}")
    print(f"{'Re':>6} {'N':>4} {'method':>10} {'C_D':>8} {'S-N':>8} {'C-G-W':>8} {'steps':>8}")
    for re in args.re:
        for n in args.sizes:
            for method in args.methods:
                r = run_sphere_drag(
                    n, re, method, collision=args.collision, backend=args.backend, dtype=args.dtype
                )
                print(
                    f"{re:6.0f} {n:4d} {method:>10} {r.cd:8.4f} {r.reference:8.4f} "
                    f"{r.reference_cgw:8.4f} {r.steps:8d}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
