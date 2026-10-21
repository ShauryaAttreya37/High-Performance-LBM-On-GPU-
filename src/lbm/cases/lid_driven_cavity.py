"""Lid-driven square cavity, compared with Ghia, Ghia & Shin (1982).

A square box of side n nodes; the top wall slides in +x at speed U, the other
three walls are at rest. Moving-wall bounce-back drives the flow. The
Reynolds number Re = U n / nu sets the viscosity::

    nu = U n / Re,      tau = 3 nu + 1/2

After the flow reaches a steady state, the velocity along the two
centrelines (u on x = 0.5, v on y = 0.5) is compared with Ghia's tabulated
multigrid solution.

Why this case matters: it has a primary vortex, corner eddies and thin
shear layers along the lid, and two corner singularities where the moving
lid meets a fixed wall. It exercises moving-wall bounce-back, mass
conservation in a closed box, and stability as tau approaches 1/2 at
higher Re (Re = 1000 at n = 128 needs tau = 0.538).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from lbm.backends import make_solver
from lbm.boundaries import cavity_walls
from lbm.cases.common import add_backend_arguments, output_path, run_until_steady
from lbm.cases.ghia1982 import REYNOLDS_NUMBERS, ghia_centerlines
from lbm.diagnostics import vorticity
from lbm.params import SimulationParams


@dataclass(frozen=True)
class CavityResult:
    reynolds: int
    n: int
    steps: int
    y: np.ndarray
    """Positions along the vertical centreline, including both walls."""
    u: np.ndarray
    """u / U along the vertical centreline."""
    x: np.ndarray
    """Positions along the horizontal centreline, including both walls."""
    v: np.ndarray
    """v / U along the horizontal centreline."""
    vorticity: np.ndarray
    """Vorticity field over the fluid nodes, in units of U / L."""

    def errors_vs_ghia(self) -> tuple[np.ndarray, np.ndarray]:
        """Differences (simulated - Ghia) of u and v at Ghia's sample points."""
        ref = ghia_centerlines(self.reynolds)
        u_error = np.interp(ref.y, self.y, self.u) - ref.u
        v_error = np.interp(ref.x, self.x, self.v) - ref.v
        return u_error, v_error

    @property
    def max_error(self) -> float:
        """Largest absolute centreline deviation from Ghia, in units of U."""
        u_error, v_error = self.errors_vs_ghia()
        return float(max(np.abs(u_error).max(), np.abs(v_error).max()))


def _centre(field: np.ndarray, axis: int) -> np.ndarray:
    """Values on the mid-line of ``field`` along ``axis`` (average of two lines if even)."""
    n = field.shape[axis]
    if n % 2:
        return np.take(field, n // 2, axis=axis)
    return 0.5 * (np.take(field, n // 2 - 1, axis=axis) + np.take(field, n // 2, axis=axis))


def run_cavity(
    reynolds: int = 100,
    n: int = 64,
    lid_velocity: float = 0.1,
    collision: str = "trt",
    backend: str = "numpy",
    dtype: str = "float64",
    tol: float = 1e-6,
    max_steps: int = 2_000_000,
) -> CavityResult:
    """Run the cavity to a steady state.

    Args:
        reynolds: One of Ghia's Reynolds numbers (100, 400, 1000).
        n: Fluid nodes per side.
        lid_velocity: Lid speed U in lattice units (Ma = U sqrt(3)).
        collision: ``"bgk"`` or ``"trt"``.
        backend: Solver backend name.
        dtype: ``"float32"`` or ``"float64"``.
        tol: Steady when no velocity component changes by more than tol * U
            over one check interval.
        max_steps: Give up after this many steps.
    """
    if reynolds not in REYNOLDS_NUMBERS:
        raise ValueError(f"reynolds must be one of {REYNOLDS_NUMBERS}; got {reynolds}")
    viscosity = lid_velocity * n / reynolds
    solid, wall = cavity_walls(n, lid_velocity)
    params = SimulationParams(
        nx=n + 2,
        ny=n + 2,
        tau=3.0 * viscosity + 0.5,
        collision=collision,
        solid=solid,
        wall_velocity=wall,
        dtype=dtype,
    )
    solver = make_solver(params, backend=backend)
    solver.initialize()

    def velocity(s):
        fields = s.macroscopic()
        return np.stack([fields.ux, fields.uy]) / lid_velocity

    # One lid-transit time n / U between checks.
    interval = 2 * round(n / lid_velocity / 2)
    steps = run_until_steady(solver, velocity, interval=interval, tol=tol, max_steps=max_steps)

    fields = solver.macroscopic()
    ux = fields.ux[1:-1, 1:-1] / lid_velocity
    uy = fields.uy[1:-1, 1:-1] / lid_velocity
    coords = (np.arange(n) + 0.5) / n
    return CavityResult(
        reynolds=reynolds,
        n=n,
        steps=steps,
        y=np.concatenate([[0.0], coords, [1.0]]),
        u=np.concatenate([[0.0], _centre(ux, axis=1), [1.0]]),
        x=np.concatenate([[0.0], coords, [1.0]]),
        v=np.concatenate([[0.0], _centre(uy, axis=0), [0.0]]),
        vorticity=vorticity(ux, uy) * n,
    )


def _plot(result: CavityResult, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ref = ghia_centerlines(result.reynolds)
    fig, (ax_u, ax_v, ax_w) = plt.subplots(1, 3, figsize=(14, 4.2))
    ax_u.plot(result.u, result.y, "-", label=f"LBM n={result.n}")
    ax_u.plot(ref.u, ref.y, "ko", ms=4, label="Ghia et al. 1982")
    ax_u.set(xlabel="u / U", ylabel="y", title=f"u on x = 0.5, Re = {result.reynolds}")
    ax_v.plot(result.x, result.v, "-", label=f"LBM n={result.n}")
    ax_v.plot(ref.x, ref.v, "ko", ms=4, label="Ghia et al. 1982")
    ax_v.set(xlabel="x", ylabel="v / U", title=f"v on y = 0.5, Re = {result.reynolds}")
    limit = np.percentile(np.abs(result.vorticity), 98)
    image = ax_w.imshow(
        result.vorticity,
        origin="lower",
        extent=(0, 1, 0, 1),
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
    )
    ax_w.set(title="vorticity (U / L)", xlabel="x", ylabel="y")
    fig.colorbar(image, ax=ax_w)
    ax_u.legend()
    ax_v.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--re", type=int, default=100, choices=REYNOLDS_NUMBERS)
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--lid-velocity", type=float, default=0.1)
    add_backend_arguments(parser)
    args = parser.parse_args(argv)

    result = run_cavity(
        args.re, args.n, args.lid_velocity, args.collision, args.backend, args.dtype
    )
    u_error, v_error = result.errors_vs_ghia()
    print(f"Lid-driven cavity Re = {args.re}, n = {args.n}, {args.collision}, {args.backend}")
    print(f"steady after {result.steps} steps")
    print(f"max |u - u_Ghia| = {np.abs(u_error).max():.4f} U")
    print(f"max |v - v_Ghia| = {np.abs(v_error).max():.4f} U")
    if args.plot:
        path = output_path(f"cavity_re{args.re}_n{args.n}.png")
        _plot(result, path)
        print(f"figure written to {path}")


if __name__ == "__main__":
    main()
