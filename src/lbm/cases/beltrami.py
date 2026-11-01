"""Decaying ABC (Beltrami) flow in a fully periodic cube: an exact 3D test.

The 3D analogue of the Taylor-Green check. The usual 3D Taylor-Green initial
field (u = U sin kx cos ky cos kz, ...) is *not* an exact solution: its
nonlinear term transfers energy to smaller scales, so its decay only matches
2 nu k^2 in the limit Re -> 0, and a convergence study measures that model
error as well as the scheme's. The Arnold-Beltrami-Childress flow with
A = B = C = U on an N^3 periodic box, k = 2 pi / N::

    ux = U (sin kz + cos ky) e^{-nu k^2 t}
    uy = U (sin kx + cos kz) e^{-nu k^2 t}
    uz = U (sin ky + cos kx) e^{-nu k^2 t}
    p  = -rho0 |u|^2 / 2

is exact for any Re. It is a Beltrami field (curl u = k u), so the vorticity
is parallel to the velocity, the Lamb vector u x omega vanishes and the
nonlinear term reduces to the gradient grad(|u|^2 / 2), which the pressure
balances. What is left is pure viscous decay at rate nu k^2 for u (2 nu k^2
for the kinetic energy). Every velocity component varies along two axes and
all three axes are used, so this exercises the full D3Q19 scheme.

As in 2D, the grid is refined in diffusive scaling (U ~ 1/N, tau fixed) so
the O(Ma^2) compressibility error falls with the discretisation error.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import numpy as np

from lbm.backends import make_solver
from lbm.cases.common import add_backend_arguments, even
from lbm.diagnostics import kinetic_energy
from lbm.lattice import CS2
from lbm.params import SimulationParams

REFERENCE_SIZE = 32
REFERENCE_VELOCITY = 0.01
"""Amplitude U at N = 32; diffusive scaling uses U = 0.01 * 32 / N (|u| <= 2U)."""


def exact_fields(n: int, velocity: float, viscosity: float, t: float):
    """Analytical (rho, ux, uy, uz) on an n^3 grid, arrays indexed [z, y, x]."""
    k = 2.0 * math.pi / n
    z, y, x = np.meshgrid(*(np.arange(n, dtype=np.float64),) * 3, indexing="ij")
    amplitude = velocity * math.exp(-viscosity * k * k * t)
    ux = amplitude * (np.sin(k * z) + np.cos(k * y))
    uy = amplitude * (np.sin(k * x) + np.cos(k * z))
    uz = amplitude * (np.sin(k * y) + np.cos(k * x))
    p = -0.5 * (ux**2 + uy**2 + uz**2)
    return 1.0 + p / CS2, ux, uy, uz


@dataclass(frozen=True)
class BeltramiResult:
    n: int
    steps: int
    l2_error: float
    """Relative L2 velocity error at the final time."""
    decay_rate: float
    """Measured kinetic-energy decay rate -d ln E / dt."""
    exact_decay_rate: float
    """2 nu k^2."""

    @property
    def decay_rate_error(self) -> float:
        return abs(self.decay_rate / self.exact_decay_rate - 1.0)


def run_beltrami(
    n: int = REFERENCE_SIZE,
    tau: float = 0.8,
    collision: str = "trt",
    backend: str = "numpy",
    dtype: str = "float64",
    velocity: float | None = None,
    decay_times: float = 1.0,
) -> BeltramiResult:
    """Run until the velocity amplitude has decayed by ``exp(-decay_times)``.

    Args:
        n: Grid size (n^3, periodic).
        tau: Relaxation time (fixed under refinement).
        collision: ``"bgk"`` or ``"trt"``.
        backend: Solver backend name.
        dtype: ``"float32"`` or ``"float64"``.
        velocity: Amplitude U; defaults to diffusive scaling 0.01 * 32 / n.
        decay_times: Simulated time in units of 1 / (nu k^2).
    """
    if velocity is None:
        velocity = REFERENCE_VELOCITY * REFERENCE_SIZE / n
    viscosity = CS2 * (tau - 0.5)
    k = 2.0 * math.pi / n
    steps = even(decay_times / (viscosity * k * k))

    params = SimulationParams(nx=n, ny=n, nz=n, tau=tau, collision=collision, dtype=dtype)
    solver = make_solver(params, backend=backend)
    solver.initialize(*exact_fields(n, velocity, viscosity, 0.0))
    start = solver.macroscopic()
    solver.run(steps)
    end = solver.macroscopic()

    exact = exact_fields(n, velocity, viscosity, steps)[1:]
    error = math.sqrt(sum(np.sum((u - e) ** 2) for u, e in zip(end.u, exact, strict=True)))
    error /= math.sqrt(sum(np.sum(e**2) for e in exact))
    energy_ratio = kinetic_energy(end.rho, *end.u) / kinetic_energy(start.rho, *start.u)
    return BeltramiResult(
        n=n,
        steps=steps,
        l2_error=error,
        decay_rate=-math.log(energy_ratio) / steps,
        exact_decay_rate=2.0 * viscosity * k * k,
    )


def main(argv=None) -> None:
    from lbm.cases.taylor_green import convergence_order

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sizes", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument("--tau", type=float, default=0.8)
    add_backend_arguments(parser)
    args = parser.parse_args(argv)

    results = [
        run_beltrami(n, args.tau, args.collision, args.backend, args.dtype) for n in args.sizes
    ]
    orders = [math.nan, *convergence_order(results)]
    print(f"ABC (Beltrami) flow, tau = {args.tau}, {args.collision}, backend = {args.backend}")
    print(f"{'N':>6} {'steps':>8} {'L2 error':>12} {'order':>7} {'decay-rate error':>17}")
    for r, order in zip(results, orders, strict=True):
        print(f"{r.n:6d} {r.steps:8d} {r.l2_error:12.3e} {order:7.2f} {r.decay_rate_error:17.3e}")


if __name__ == "__main__":
    main()
