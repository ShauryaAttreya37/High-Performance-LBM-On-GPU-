"""Force-driven flow through a rectangular duct: a genuinely 3D steady test.

Set-up: periodic in x, solid layers at y = 0, y = ny - 1, z = 0 and
z = nz - 1, uniform body force F along x. With halfway bounce-back the
cross-section is H x W with H = ny - 2, W = nz - 2; measured from a wall,
fluid node j sits at j - 1/2. The steady solution solves
nu (d^2/dy^2 + d^2/dz^2) u = -F with u = 0 on the walls. As a Fourier series
in y (odd n only)::

    u(y, z) = 4 F H^2 / (nu pi^3) sum_n 1/n^3 sin(n pi y / H)
              * [1 - cosh(n pi (z - W/2) / H) / cosh(n pi W / (2 H))]

Unlike the plate channel, the profile is not a polynomial, so even TRT with
Lambda = 3/16 is no longer exact: the error is O(1/H^2) and the convergence
order under refinement checks the walls in two directions at once.

The flow has u only along x, varying in y and z, so (u . grad) u = 0 and the
Stokes solution above is exact at any Reynolds number.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import numpy as np

from lbm.backends import make_solver
from lbm.cases.common import add_backend_arguments, run_until_steady
from lbm.lattice import CS2
from lbm.params import SimulationParams

DUCT_LENGTH = 4
"""Nodes along x; the flow is uniform in x, so a short periodic strip suffices."""

SERIES_TERMS = 2000
"""Odd Fourier modes kept; the truncation error is ~1 / (8 SERIES_TERMS^2) relative."""


def exact_profile(
    y: np.ndarray, z: np.ndarray, height: float, width: float, force: float, viscosity: float
) -> np.ndarray:
    """Series solution u(y, z) on the grid spanned by 1D arrays y and z (result indexed [z, y])."""
    n = (2 * np.arange(SERIES_TERMS) + 1).astype(np.float64)[:, None, None]
    yy = y[None, None, :]
    zz = z[None, :, None]
    a = n * math.pi * np.abs(zz - 0.5 * width) / height
    b = n * math.pi * 0.5 * width / height
    # cosh(a) / cosh(b) without overflow for large n (a <= b inside the duct).
    ratio = np.exp(a - b) * (1.0 + np.exp(-2.0 * a)) / (1.0 + np.exp(-2.0 * b))
    terms = np.sin(n * math.pi * yy / height) * (1.0 - ratio) / n**3
    return 4.0 * force * height**2 / (viscosity * math.pi**3) * terms.sum(axis=0)


@dataclass(frozen=True)
class DuctResult:
    u: np.ndarray
    """Simulated cross-section velocity, indexed [z, y] over fluid nodes."""
    u_exact: np.ndarray
    steps: int

    @property
    def l2_error(self) -> float:
        """Relative L2 error ||u - u_exact|| / ||u_exact||."""
        return float(np.linalg.norm(self.u - self.u_exact) / np.linalg.norm(self.u_exact))


def run_duct_flow(
    height: int = 16,
    width: int | None = None,
    tau: float = 1.0,
    collision: str = "trt",
    backend: str = "numpy",
    dtype: str = "float64",
    max_velocity: float = 0.01,
    tol: float | None = None,
    max_steps: int = 1_000_000,
) -> DuctResult:
    """Run to steady state and compare with the series solution.

    Args:
        height: Duct size H along y, in nodes.
        width: Duct size W along z; defaults to a square duct (W = H).
        tau: Relaxation time.
        collision: ``"bgk"`` or ``"trt"``.
        backend: Solver backend name (any 3D-capable one).
        dtype: ``"float32"`` or ``"float64"``.
        max_velocity: Target centre velocity; sets F from the exact solution.
        tol: Steady-state threshold on the change of u / u_max per check
            interval. Defaults to a level just above round-off for ``dtype``.
        max_steps: Give up after this many steps.
    """
    width = height if width is None else width
    ny, nz = height + 2, width + 2
    viscosity = CS2 * (tau - 0.5)
    y = np.arange(1, ny - 1) - 0.5
    z = np.arange(1, nz - 1) - 0.5
    centre = exact_profile(
        np.array([0.5 * height]), np.array([0.5 * width]), height, width, 1.0, viscosity
    )
    force = max_velocity / float(centre[0, 0])

    solid = np.zeros((nz, ny, DUCT_LENGTH), dtype=bool)
    solid[:, [0, -1], :] = True
    solid[[0, -1], :, :] = True
    params = SimulationParams(
        nx=DUCT_LENGTH,
        ny=ny,
        nz=nz,
        tau=tau,
        collision=collision,
        force=(force, 0.0, 0.0),
        solid=solid,
        dtype=dtype,
    )
    solver = make_solver(params, backend=backend)
    solver.initialize()

    def cross_section(s):
        return s.macroscopic().ux[1:-1, 1:-1, :].mean(axis=-1)

    if tol is None:
        tol = 1e-12 if dtype == "float64" else 1e-6
    # The slowest mode decays like exp(-pi^2 nu (1/H^2 + 1/W^2) t).
    smaller = min(height, width)
    interval = max(2, 2 * round(smaller**2 / (20.0 * viscosity)))
    steps = run_until_steady(
        solver,
        lambda s: cross_section(s) / max_velocity,
        interval=interval,
        tol=tol,
        max_steps=max_steps,
    )
    return DuctResult(
        u=cross_section(solver),
        u_exact=exact_profile(y, z, height, width, force, viscosity),
        steps=steps,
    )


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--heights", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--tau", type=float, default=1.0)
    add_backend_arguments(parser)
    args = parser.parse_args(argv)

    print(f"Square duct, tau = {args.tau}, {args.collision}, backend = {args.backend}")
    print(f"{'H':>6} {'steps':>8} {'L2 error':>12} {'order':>7}")
    previous = None
    for h in args.heights:
        r = run_duct_flow(
            h, tau=args.tau, collision=args.collision, backend=args.backend, dtype=args.dtype
        )
        order = math.nan
        if previous:
            order = math.log(previous[1] / r.l2_error) / math.log(h / previous[0])
        print(f"{h:6d} {r.steps:8d} {r.l2_error:12.3e} {order:7.2f}")
        previous = (h, r.l2_error)


if __name__ == "__main__":
    main()
