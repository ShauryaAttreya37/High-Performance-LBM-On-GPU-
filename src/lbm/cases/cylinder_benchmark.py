"""Flow around a cylinder in a channel: the Schaefer-Turek DFG benchmark (2D-1, 2D-2).

Schaefer & Turek (1996) defined this case so that many codes could be
compared: a channel of height H = 0.41 m and length L = 2.2 m, a cylinder of
diameter D = 0.1 m centred at (0.2, 0.2) (slightly off the centreline, so lift
is not zero), kinematic viscosity nu = 1e-3 m^2/s and a parabolic inflow::

    u(0, y) = 4 U_m y (H - y) / H^2,     U_mean = 2 U_m / 3,     Re = U_mean D / nu

Test 2D-1: U_m = 0.3 m/s, Re = 20, steady.
Test 2D-2: U_m = 1.5 m/s, Re = 100, periodic vortex shedding.

Reported quantities and the intervals the benchmark accepts:

    C_D = 2 F_x / (rho U_mean^2 D),  C_L = 2 F_y / (rho U_mean^2 D)
    dp  = p(0.15, 0.2) - p(0.25, 0.2)       (front minus back of the cylinder)
    St  = D f / U_mean                       (f: lift frequency)

    2D-1:  C_D in [5.57, 5.59]     C_L in [0.0104, 0.0110]   dp in [0.1172, 0.1176]
    2D-2:  C_D,max in [3.22, 3.24] C_L,max in [0.99, 1.01]   St in [0.295, 0.305]

Lattice set-up: N nodes per diameter, dx = D / N. The x = 0 layer is a
Zou-He velocity inlet acting at x = 1, the last layer a Zou-He pressure
outlet; channel walls use halfway bounce-back. The cylinder is one of:

* ``bouzidi``: solid nodes plus interpolated bounce-back; force by momentum
  exchange.
* ``staircase``: solid nodes with plain bounce-back; momentum exchange.
* ``ibm``: no solid nodes, surface markers and a local force (immersed
  boundary method); the force is what the markers exert on the fluid.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field

import numpy as np

from lbm.backends import make_solver
from lbm.cases.common import add_backend_arguments, run_until_steady
from lbm.forces import ForceProbe
from lbm.geometry import circle, voxelize, wall_links
from lbm.ibm import ImmersedBoundary
from lbm.lattice import CS2, D2Q9
from lbm.link_rules import AffineSet, LinkRules, bouzidi_rules, compile_affine
from lbm.open_boundaries import open_channel
from lbm.params import SimulationParams

HEIGHT, LENGTH, DIAMETER, CENTRE = 0.41, 2.2, 0.1, (0.2, 0.2)
VISCOSITY = 1e-3
TESTS = {"2D-1": 0.3, "2D-2": 1.5}
"""Maximum inflow velocity U_m [m/s] of each test."""
DEFAULT_U_MAX = {"2D-1": 0.01875, "2D-2": 0.05}
"""Peak inflow in lattice units. LBM is weakly compressible: the channel's
pressure drop raises the density near the cylinder by O(Ma^2), and the drag
with it. 2D-1 at N = 40 gives C_D = 5.694 / 5.609 / 5.589 for u_max = 0.075 /
0.0375 / 0.01875 (error ~ Ma^2, independent of N). At Re = 100 the viscous
pressure drop is five times smaller, and a lower u_max would push tau too
close to 1/2."""
INTERVALS = {
    "2D-1": {"cd": (5.57, 5.59), "cl": (0.0104, 0.0110), "dp": (0.1172, 0.1176)},
    "2D-2": {"cd": (3.22, 3.24), "cl": (0.99, 1.01), "st": (0.295, 0.305)},
}


@dataclass(frozen=True)
class Geometry:
    """Lattice version of the benchmark for N nodes per diameter."""

    n: int
    params: SimulationParams
    cylinder_links: object
    u_mean: float
    """Mean inflow velocity in lattice units."""
    velocity_scale: float
    """Physical velocity per lattice velocity."""
    inflow: np.ndarray
    """Inflow profile u_x(y), shape (ny,)."""

    def start(self, solver) -> None:
        """Initial state: the inflow profile everywhere (zero inside the cylinder).

        Starting from rest is an impulsive start: the inlet launches pressure
        waves that bounce between the two reflecting Zou-He ends and, at
        tau ~ 0.52, take ~100 convective times to die out, polluting C_D,max.
        """
        ux = np.where(self.params.solid, 0.0, self.inflow[:, None])
        solver.initialize(1.0, ux, 0.0)


def build(
    test: str,
    n: int,
    u_max: float | None = None,
    collision: str = "trt",
    dtype: str = "float64",
    method: str = "bouzidi",
) -> Geometry:
    """Grid, boundaries and parameters for one test at resolution N = D / dx."""
    if test not in TESTS:
        raise ValueError(f"test must be one of {tuple(TESTS)}; got {test!r}")
    if n % 10:
        raise ValueError("N must be a multiple of 10 so that H = 4.1 D spans whole nodes")
    u_max = DEFAULT_U_MAX[test] if u_max is None else u_max
    dx = DIAMETER / n
    ny = round(HEIGHT / dx) + 2  # fluid rows 1 .. ny-2, walls halfway at y = 0.5 and ny - 1.5
    nx = round(LENGTH / dx) + 3  # inlet acts at x = 1, outlet at x = nx - 2
    height = ny - 2
    y = np.arange(ny) - 0.5
    profile = np.where((y > 0) & (y < height), 4.0 * u_max * y * (height - y) / height**2, 0.0)
    inflow = np.stack([profile, np.zeros(ny)])

    solid = np.zeros((ny, nx), dtype=bool)
    solid[0, :] = solid[-1, :] = True
    body = circle(1.0 + CENTRE[0] / dx, 0.5 + CENTRE[1] / dx, 0.5 * n)
    immersed = ImmersedBoundary.from_ball(body) if method == "ibm" else None
    if immersed is None:
        solid |= voxelize(body, solid.shape)
    solid, open_rules = open_channel(D2Q9, solid, inflow)
    links = wall_links(D2Q9, body, solid) if immersed is None else None
    rules = open_rules
    if method == "bouzidi":
        rules = LinkRules.concat(open_rules, bouzidi_rules(D2Q9, links, solid))

    u_mean = 2.0 * u_max / 3.0
    reynolds = TESTS[test] * 2.0 / 3.0 * DIAMETER / VISCOSITY
    nu = u_mean * n / reynolds
    params = SimulationParams(
        nx=nx,
        ny=ny,
        tau=nu / CS2 + 0.5,
        collision=collision,
        solid=solid,
        dtype=dtype,
        link_rules=rules,
        immersed=immersed,
    )
    return Geometry(n, params, links, u_mean, TESTS[test] / u_max, profile)


@dataclass
class Observer:
    """Records C_D, C_L and dp of a running benchmark."""

    geometry: Geometry
    probe: ForceProbe | None = field(init=False)
    history: list = field(default_factory=list)

    def __post_init__(self) -> None:
        g = self.geometry
        links = g.cylinder_links
        self.probe = None if links is None else ForceProbe.build(g.params, links)
        # Density at the two pressure points (each the mean of the two rows around
        # y = 0.2), read on the device: rho - 1 = sum_k f~*_k (collision keeps mass).
        dx = DIAMETER / g.n
        row = 0.5 + CENTRE[1] / dx
        rows = [math.floor(row), math.ceil(row)]
        columns = [round(1 + 0.15 / dx), round(1 + 0.25 / dx)]
        shape = g.params.shape
        lattice = g.params.lattice
        expressions = []
        for x in columns:
            nodes = [int(np.ravel_multi_index((y, x), shape)) for y in rows]
            expressions.append([(k, y, 0.5) for y in nodes for k in range(lattice.q)])
        self._pressure_points = AffineSet.from_lists(expressions)
        self._compiled = {}

    def _densities(self, solver) -> np.ndarray:
        layout = solver.state_layout
        if layout not in self._compiled:
            p = self.geometry.params
            self._compiled[layout] = compile_affine(
                p.lattice, self._pressure_points, layout, p.solid, p.wall_velocity
            )
        return 1.0 + solver.evaluate_affine(self._compiled[layout])

    def __call__(self, solver) -> np.ndarray:
        g = self.geometry
        if self.probe is None:
            fx, fy = solver.immersed_force
        else:
            fx, fy = self.probe.evaluate(solver)
        scale = 0.5 * g.u_mean**2 * g.n
        front, back = self._densities(solver)
        dp = CS2 * (front - back) * g.velocity_scale**2
        values = np.array([fx / scale, fy / scale, dp])
        self.history.append((solver.time_step, *values))
        return values


@dataclass(frozen=True)
class SteadyResult:
    n: int
    cd: float
    cl: float
    dp: float
    steps: int


def run_steady(n: int = 20, backend: str = "numpy", dtype: str = "float64", **options):
    """Test 2D-1 to steady state."""
    geometry = build("2D-1", n, dtype=dtype, **options)
    solver = make_solver(geometry.params, backend=backend)
    geometry.start(solver)
    observe = Observer(geometry)
    interval = 2 * round(n / geometry.u_mean / 2)  # one convective time D / U
    tol = 1e-7 if dtype == "float64" else 1e-5
    steps = run_until_steady(solver, observe, interval=interval, tol=tol, max_steps=5_000_000)
    cd, cl, dp = observe(solver)
    return SteadyResult(n, float(cd), float(cl), float(dp), steps)


@dataclass(frozen=True)
class PeriodicResult:
    n: int
    cd_max: float
    cl_max: float
    strouhal: float
    steps: int
    history: np.ndarray


def run_periodic(
    n: int = 20,
    backend: str = "cuda-aa",
    dtype: str = "float32",
    convective_times: float = 150.0,
    samples_per_time: int = 40,
    **options,
):
    """Test 2D-2: run, then measure the last quarter of the record."""
    geometry = build("2D-2", n, dtype=dtype, **options)
    solver = make_solver(geometry.params, backend=backend)
    geometry.start(solver)
    observe = Observer(geometry)
    t_conv = n / geometry.u_mean
    every = max(2, 2 * round(t_conv / samples_per_time / 2))
    total = every * round(convective_times * t_conv / every)
    solver.run(total, every=every, callback=observe)
    history = np.asarray(observe.history)
    tail = history[len(history) * 3 // 4 :]
    t, cd, cl = tail[:, 0], tail[:, 1], tail[:, 2]
    strouhal = n / (lift_period(t, cl) * geometry.u_mean)
    return PeriodicResult(n, float(cd.max()), float(cl.max()), strouhal, total, history)


def lift_period(t: np.ndarray, cl: np.ndarray) -> float:
    """Mean period of C_L(t) [time steps], from its upward zero crossings.

    Each crossing is located by linear interpolation between the two samples
    around it, so the period is much finer than the sampling interval.
    """
    up = np.flatnonzero((cl[:-1] < 0) & (cl[1:] >= 0))
    if len(up) < 2:
        raise ValueError("need at least two upward zero crossings of C_L to measure a period")
    crossings = t[up] - cl[up] * (t[up + 1] - t[up]) / (cl[up + 1] - cl[up])
    return float(np.mean(np.diff(crossings)))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--test", default="2D-1", choices=tuple(TESTS))
    parser.add_argument("--sizes", type=int, nargs="+", default=[20, 40])
    parser.add_argument("--method", default="bouzidi", choices=("bouzidi", "staircase", "ibm"))
    parser.add_argument("--u-max", type=float, default=None, help="peak inflow, lattice units")
    add_backend_arguments(parser)
    args = parser.parse_args(argv)
    bounds = INTERVALS[args.test]
    print(f"Schaefer-Turek {args.test}, {args.method}, {args.collision}, backend = {args.backend}")
    print("accepted: " + ", ".join(f"{k} in [{lo}, {hi}]" for k, (lo, hi) in bounds.items()))
    for n in args.sizes:
        options = {"collision": args.collision, "method": args.method, "u_max": args.u_max}
        if args.test == "2D-1":
            r = run_steady(n, args.backend, args.dtype, **options)
            print(
                f"N = {n:3d}: C_D = {r.cd:.4f}  C_L = {r.cl:.5f}  dp = {r.dp:.4f}  "
                f"({r.steps} steps)"
            )
        else:
            r = run_periodic(n, args.backend, args.dtype, **options)
            print(
                f"N = {n:3d}: C_D,max = {r.cd_max:.4f}  C_L,max = {r.cl_max:.4f}  "
                f"St = {r.strouhal:.4f}  ({r.steps} steps)"
            )


if __name__ == "__main__":
    main()
