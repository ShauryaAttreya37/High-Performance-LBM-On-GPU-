"""Decaying Taylor-Green vortex in a fully periodic box.

An exact solution of the incompressible Navier-Stokes equations. On an
N x N periodic domain with k = 2 pi / N::

    ux(x, y, t) = -U cos(kx) sin(ky) exp(-2 nu k^2 t)
    uy(x, y, t) =  U sin(kx) cos(ky) exp(-2 nu k^2 t)
    p(x, y, t)  = -(rho0 U^2 / 4) (cos 2kx + cos 2ky) exp(-4 nu k^2 t)

There are no walls, so this isolates the bulk scheme: the decay rate checks
that the viscosity really is nu = cs^2 (tau - 1/2), and refining the grid in
*diffusive scaling* (U ~ 1/N, tau fixed, steps ~ N^2) checks second-order
convergence. Diffusive scaling is needed because the compressibility error of
LBM scales with Ma^2 ~ U^2; keeping U fixed while refining would stall the
convergence at that error.

The pressure enters through the density, p = cs^2 (rho - rho0).
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import numpy as np

from lbm.backends import make_solver
from lbm.cases.common import add_backend_arguments, even, output_path
from lbm.diagnostics import kinetic_energy
from lbm.lattice import CS2
from lbm.params import SimulationParams

REFERENCE_SIZE = 32
REFERENCE_VELOCITY = 0.02
"""Lattice velocity at N = 32; diffusive scaling uses U = 0.02 * 32 / N."""


def exact_fields(n: int, velocity: float, viscosity: float, t: float):
    """Analytical (rho, ux, uy) on an n x n grid at time t (lattice units)."""
    k = 2.0 * math.pi / n
    y, x = np.mgrid[0:n, 0:n].astype(np.float64)
    decay = math.exp(-2.0 * viscosity * k * k * t)
    ux = -velocity * np.cos(k * x) * np.sin(k * y) * decay
    uy = velocity * np.sin(k * x) * np.cos(k * y) * decay
    p = -0.25 * velocity**2 * (np.cos(2 * k * x) + np.cos(2 * k * y)) * decay**2
    return 1.0 + p / CS2, ux, uy


@dataclass(frozen=True)
class TaylorGreenResult:
    n: int
    steps: int
    l2_error: float
    """Relative L2 velocity error at the final time."""
    decay_rate: float
    """Measured kinetic-energy decay rate -d ln E / dt."""
    exact_decay_rate: float
    """4 nu k^2."""

    @property
    def decay_rate_error(self) -> float:
        return abs(self.decay_rate / self.exact_decay_rate - 1.0)


def run_taylor_green(
    n: int = REFERENCE_SIZE,
    tau: float = 0.8,
    collision: str = "trt",
    backend: str = "numpy",
    dtype: str = "float64",
    velocity: float | None = None,
    decay_times: float = 1.0,
) -> TaylorGreenResult:
    """Run until the velocity amplitude has decayed by ``exp(-decay_times)``.

    Args:
        n: Grid size (n x n, periodic).
        tau: Relaxation time (fixed under refinement).
        collision: ``"bgk"`` or ``"trt"``.
        backend: Solver backend name.
        dtype: ``"float32"`` or ``"float64"``.
        velocity: Initial amplitude U; defaults to diffusive scaling 0.02 * 32 / n.
        decay_times: Simulated time in units of 1 / (2 nu k^2).
    """
    if velocity is None:
        velocity = REFERENCE_VELOCITY * REFERENCE_SIZE / n
    viscosity = CS2 * (tau - 0.5)
    k = 2.0 * math.pi / n
    steps = even(decay_times / (2.0 * viscosity * k * k))

    params = SimulationParams(nx=n, ny=n, tau=tau, collision=collision, dtype=dtype)
    solver = make_solver(params, backend=backend)
    solver.initialize(*exact_fields(n, velocity, viscosity, 0.0))
    start = solver.macroscopic()
    solver.run(steps)
    end = solver.macroscopic()

    _, ux_exact, uy_exact = exact_fields(n, velocity, viscosity, steps)
    error = np.sqrt(np.sum((end.ux - ux_exact) ** 2 + (end.uy - uy_exact) ** 2))
    error /= np.sqrt(np.sum(ux_exact**2 + uy_exact**2))
    energy_ratio = kinetic_energy(end.rho, end.ux, end.uy) / kinetic_energy(
        start.rho, start.ux, start.uy
    )
    return TaylorGreenResult(
        n=n,
        steps=steps,
        l2_error=float(error),
        decay_rate=-math.log(energy_ratio) / steps,
        exact_decay_rate=4.0 * viscosity * k * k,
    )


def convergence_order(results: list[TaylorGreenResult]) -> list[float]:
    """Observed order log(e_coarse / e_fine) / log(n_fine / n_coarse) per refinement."""
    return [
        math.log(coarse.l2_error / fine.l2_error) / math.log(fine.n / coarse.n)
        for coarse, fine in zip(results, results[1:], strict=False)
    ]


def _plot(results: list[TaylorGreenResult], path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = np.array([r.n for r in results], dtype=float)
    err = np.array([r.l2_error for r in results])
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.loglog(n, err, "o-", label="LBM")
    ax.loglog(n, err[0] * (n[0] / n) ** 2, "k--", label="slope -2")
    ax.set(xlabel="N", ylabel="relative L2 velocity error", title="Taylor-Green convergence")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sizes", type=int, nargs="+", default=[16, 32, 64, 128])
    parser.add_argument("--tau", type=float, default=0.8)
    add_backend_arguments(parser)
    args = parser.parse_args(argv)

    results = [
        run_taylor_green(n, args.tau, args.collision, args.backend, args.dtype) for n in args.sizes
    ]
    orders = [math.nan, *convergence_order(results)]
    print(f"Taylor-Green vortex, tau = {args.tau}, {args.collision}, backend = {args.backend}")
    print(f"{'N':>6} {'steps':>8} {'L2 error':>12} {'order':>7} {'decay-rate error':>17}")
    for r, order in zip(results, orders, strict=True):
        print(f"{r.n:6d} {r.steps:8d} {r.l2_error:12.3e} {order:7.2f} {r.decay_rate_error:17.3e}")
    if args.plot:
        path = output_path("taylor_green_convergence.png")
        _plot(results, path)
        print(f"figure written to {path}")


if __name__ == "__main__":
    main()
