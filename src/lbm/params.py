"""Validated, immutable description of one simulation.

All checks happen here, once, so every backend can trust its input.

Relaxation time and viscosity (lattice units, dx = dt = 1)::

    nu = cs^2 (tau - 1/2) = (tau - 1/2) / 3

tau <= 1/2 means zero or negative viscosity, which is unphysical, so it is
rejected. As tau approaches 1/2 from above, BGK loses its numerical damping of
non-hydrodynamic modes and blows up in shear layers; values below
``TAU_WARNING`` raise a ``StabilityWarning``.

TRT (two-relaxation-time) splits populations into symmetric and antisymmetric
parts with rates omega_plus = 1/tau (sets viscosity) and omega_minus, chosen
through the "magic" parameter::

    Lambda = (1/omega_plus - 1/2) (1/omega_minus - 1/2)

Lambda = 3/16 places the halfway bounce-back wall exactly midway between
nodes for Poiseuille flow, independent of tau. BGK is the special case
omega_minus = omega_plus.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from lbm.lattice import CS2

TAU_WARNING = 0.51
"""Below this relaxation time BGK is close to its stability limit."""

COLLISION_MODELS = ("bgk", "trt")
SUPPORTED_DTYPES = ("float32", "float64")


class StabilityWarning(UserWarning):
    """Parameters are valid but numerically risky (tau near 1/2, high Mach number)."""


def check_tau(tau: float) -> None:
    """Raise for tau <= 1/2, warn when tau is close to the stability limit."""
    if not tau > 0.5:
        raise ValueError(f"tau must be > 0.5 (viscosity nu = (tau - 0.5)/3 > 0); got tau={tau}")
    if tau < TAU_WARNING:
        warnings.warn(
            f"tau={tau:.6f} is close to 0.5: BGK becomes unstable in shear layers; "
            "use TRT, refine the grid, or lower the lattice velocity",
            StabilityWarning,
            stacklevel=3,
        )


def _frozen_copy(array: np.ndarray) -> np.ndarray:
    copy = np.array(array, copy=True)
    copy.setflags(write=False)
    return copy


@dataclass(frozen=True, eq=False)
class SimulationParams:
    """Everything a solver needs to advance one D2Q9 simulation.

    Attributes:
        nx, ny: Grid size in nodes. Arrays are indexed ``[y, x]``.
        tau: BGK relaxation time / TRT symmetric relaxation time (> 0.5).
        collision: ``"bgk"`` or ``"trt"``.
        magic: TRT magic parameter Lambda (ignored for BGK). 3/16 by default.
        force: Uniform body-force density (Fx, Fy) in lattice units.
        solid: Boolean mask, True on solid nodes. Default: all fluid. Domain
            edges are periodic unless solid nodes close them off.
        wall_velocity: Shape (2, ny, nx) wall velocity, read only at solid
            nodes (moving-wall bounce-back). Values on fluid nodes are zeroed.
        dtype: ``"float32"`` (performance) or ``"float64"`` (reference tests).
    """

    nx: int
    ny: int
    tau: float
    collision: str = "bgk"
    magic: float = 3.0 / 16.0
    force: tuple[float, float] = (0.0, 0.0)
    solid: np.ndarray | None = field(default=None, repr=False)
    wall_velocity: np.ndarray | None = field(default=None, repr=False)
    dtype: str = "float32"

    def __post_init__(self) -> None:
        if self.nx <= 0 or self.ny <= 0:
            raise ValueError(f"nx and ny must be positive; got nx={self.nx}, ny={self.ny}")
        if self.collision not in COLLISION_MODELS:
            raise ValueError(f"collision must be one of {COLLISION_MODELS}; got {self.collision!r}")
        if not self.magic > 0:
            raise ValueError(f"TRT magic parameter must be > 0; got magic={self.magic}")
        if self.dtype not in SUPPORTED_DTYPES:
            raise ValueError(f"dtype must be one of {SUPPORTED_DTYPES}; got {self.dtype!r}")
        check_tau(self.tau)

        shape = (self.ny, self.nx)
        solid = np.zeros(shape, dtype=bool) if self.solid is None else np.asarray(self.solid)
        if solid.shape != shape:
            raise ValueError(f"solid mask must have shape (ny, nx)={shape}; got {solid.shape}")
        solid = solid.astype(bool)

        if self.wall_velocity is None:
            wall = np.zeros((2, *shape))
        else:
            wall = np.asarray(self.wall_velocity, dtype=np.float64)
        if wall.shape != (2, *shape):
            raise ValueError(
                f"wall_velocity must have shape (2, ny, nx)={(2, *shape)}; got {wall.shape}"
            )
        wall = np.where(solid, wall, 0.0)

        fx, fy = (float(c) for c in self.force)
        object.__setattr__(self, "solid", _frozen_copy(solid))
        object.__setattr__(self, "wall_velocity", _frozen_copy(wall))
        object.__setattr__(self, "force", (fx, fy))
        object.__setattr__(self, "tau", float(self.tau))
        object.__setattr__(self, "magic", float(self.magic))

    @property
    def fluid(self) -> np.ndarray:
        """Boolean mask, True on fluid nodes."""
        return ~self.solid

    @property
    def viscosity(self) -> float:
        """Kinematic viscosity in lattice units, nu = cs^2 (tau - 1/2)."""
        return CS2 * (self.tau - 0.5)

    @property
    def relaxation_rates(self) -> tuple[float, float]:
        """(omega_plus, omega_minus) for the TRT operator; equal for BGK."""
        omega_plus = 1.0 / self.tau
        if self.collision == "bgk":
            return omega_plus, omega_plus
        tau_minus = 0.5 + self.magic / (self.tau - 0.5)
        return omega_plus, 1.0 / tau_minus
