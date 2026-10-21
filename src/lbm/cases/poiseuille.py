"""Force-driven Poiseuille flow between two plates.

Set-up: channel periodic in x, solid rows at y = 0 and y = ny - 1, uniform
body force F along x. Halfway bounce-back puts the walls at y = 0.5 and
y = ny - 1.5, so the width is H = ny - 2. Measuring y from the lower wall,
fluid node j sits at y_j = j - 1/2 and the exact steady solution is::

    u(y) = F / (2 rho nu) * y (H - y),       u_max = F H^2 / (8 rho nu)

What this case teaches: the steady LBM solution is itself a parabola, so the
only error left is *where the walls effectively are*. For BGK the bounce-back
wall sits at a tau-dependent position, which shows up as a slip velocity that
grows with tau. TRT with Lambda = 3/16 makes the wall land exactly halfway for
every tau, so the error drops to round-off. BGK is also exact at the single
tau where (tau - 1/2)^2 = 3/16, i.e. tau = 1/2 + sqrt(3)/4.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import numpy as np

from lbm.backends import make_solver
from lbm.boundaries import channel_walls
from lbm.cases.common import add_backend_arguments, output_path, run_until_steady
from lbm.lattice import CS2
from lbm.params import SimulationParams

CHANNEL_LENGTH = 4
"""Nodes along x; the flow is uniform in x, so a short periodic strip suffices."""

BGK_EXACT_TAU = 0.5 + math.sqrt(3.0) / 4.0
"""The one BGK relaxation time with (tau - 1/2)^2 = 3/16 (exact wall placement)."""


@dataclass(frozen=True)
class PoiseuilleResult:
    y: np.ndarray
    """Distance of each fluid node from the lower wall."""
    u: np.ndarray
    """Simulated velocity profile."""
    u_exact: np.ndarray
    steps: int

    @property
    def l2_error(self) -> float:
        """Relative L2 error ||u - u_exact|| / ||u_exact||."""
        return float(np.linalg.norm(self.u - self.u_exact) / np.linalg.norm(self.u_exact))


def exact_profile(y: np.ndarray, height: int, force: float, viscosity: float) -> np.ndarray:
    return force / (2.0 * viscosity) * y * (height - y)


def run_poiseuille(
    height: int = 16,
    tau: float = 0.8,
    collision: str = "trt",
    backend: str = "numpy",
    dtype: str = "float64",
    max_velocity: float = 0.01,
    tol: float | None = None,
    max_steps: int = 500_000,
) -> PoiseuilleResult:
    """Run to steady state and compare with the analytical parabola.

    Args:
        height: Channel width H in nodes.
        tau: Relaxation time.
        collision: ``"bgk"`` or ``"trt"``.
        backend: Solver backend name.
        dtype: ``"float32"`` or ``"float64"``.
        max_velocity: Target centreline velocity; sets F = 8 nu u_max / H^2.
        tol: Steady-state threshold on the change of u / u_max per check
            interval. Defaults to a level just above round-off for ``dtype``.
        max_steps: Give up after this many steps.
    """
    ny = height + 2
    viscosity = CS2 * (tau - 0.5)
    force = 8.0 * viscosity * max_velocity / height**2
    params = SimulationParams(
        nx=CHANNEL_LENGTH,
        ny=ny,
        tau=tau,
        collision=collision,
        force=(force, 0.0),
        solid=channel_walls(CHANNEL_LENGTH, ny),
        dtype=dtype,
    )
    solver = make_solver(params, backend=backend)
    solver.initialize()

    def profile(s):
        return s.macroscopic().ux[1:-1].mean(axis=1) / max_velocity

    if tol is None:
        tol = 1e-12 if dtype == "float64" else 1e-6
    # The slowest mode decays like exp(-pi^2 nu t / H^2); check a few times per decay time.
    interval = max(2, 2 * round(height**2 / (20.0 * viscosity)))
    steps = run_until_steady(solver, profile, interval=interval, tol=tol, max_steps=max_steps)

    y = np.arange(1, ny - 1) - 0.5
    u = solver.macroscopic().ux[1:-1].mean(axis=1)
    return PoiseuilleResult(
        y=y, u=u, u_exact=exact_profile(y, height, force, viscosity), steps=steps
    )


def _plot(results: dict[str, PoiseuilleResult], path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_profile, ax_error) = plt.subplots(1, 2, figsize=(10, 4))
    first = next(iter(results.values()))
    ax_profile.plot(first.u_exact / first.u_exact.max(), first.y, "k-", label="exact")
    for label, r in results.items():
        ax_profile.plot(r.u / r.u_exact.max(), r.y, "o", ms=4, label=label)
        ax_error.semilogy(r.y, np.abs(r.u - r.u_exact) / r.u_exact.max() + 1e-17, label=label)
    ax_profile.set(xlabel="u / u_max", ylabel="y (lattice units)", title="Velocity profile")
    ax_error.set(xlabel="y", ylabel="|u - u_exact| / u_max", title="Pointwise error")
    ax_profile.legend()
    ax_error.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--height", type=int, default=16)
    parser.add_argument("--taus", type=float, nargs="+", default=[0.6, 0.8, 1.0, 1.5, 2.0])
    add_backend_arguments(parser)
    args = parser.parse_args(argv)

    print(f"Poiseuille flow, H = {args.height}, backend = {args.backend}, dtype = {args.dtype}")
    print(f"{'tau':>8} {'BGK L2 error':>14} {'TRT L2 error':>14}")
    results = {}
    for tau in args.taus:
        errors = []
        for collision in ("bgk", "trt"):
            r = run_poiseuille(args.height, tau, collision, args.backend, args.dtype)
            results[f"{collision.upper()} tau={tau}"] = r
            errors.append(r.l2_error)
        print(f"{tau:8.3f} {errors[0]:14.3e} {errors[1]:14.3e}")
    if args.plot:
        path = output_path("poiseuille.png")
        _plot(results, path)
        print(f"figure written to {path}")


if __name__ == "__main__":
    main()
