"""Validated, immutable description of one simulation, in 2D or 3D.

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

Dimension: a simulation is 2D unless ``nz`` is given. Arrays are indexed
``[y, x]`` in 2D and ``[z, y, x]`` in 3D, with x varying fastest in memory.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from lbm.ibm import ImmersedBoundary
from lbm.lattice import LATTICES, VelocitySet, default_lattice
from lbm.link_rules import LinkRules

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


def _resolve_lattice(spec: str | VelocitySet | None, d: int) -> VelocitySet:
    if spec is None:
        return default_lattice(d)
    if isinstance(spec, VelocitySet):
        lattice = spec
    elif spec in LATTICES:
        lattice = LATTICES[spec]
    else:
        raise ValueError(f"unknown lattice {spec!r}; choose from {tuple(LATTICES)}")
    if lattice.d != d:
        raise ValueError(
            f"lattice {lattice.name} is {lattice.d}D but the grid is {d}D "
            "(set nz for 3D, leave it unset for 2D)"
        )
    return lattice


@dataclass(frozen=True, eq=False)
class SimulationParams:
    """Everything a solver needs to advance one simulation.

    Attributes:
        nx, ny: Grid size in nodes along x and y.
        tau: BGK relaxation time / TRT symmetric relaxation time (> 0.5).
        collision: ``"bgk"`` or ``"trt"``.
        magic: TRT magic parameter Lambda (ignored for BGK). 3/16 by default.
        force: Uniform body-force density in lattice units, one component per
            dimension: (Fx, Fy) or (Fx, Fy, Fz). Default: no force.
        solid: Boolean mask of shape ``shape``, True on solid nodes. Default:
            all fluid. Domain edges are periodic unless solid nodes close them.
        wall_velocity: Shape (d, *shape) wall velocity, read only at solid
            nodes (moving-wall bounce-back). Values on fluid nodes are zeroed.
        dtype: ``"float32"`` (performance) or ``"float64"`` (reference tests).
        nz: Nodes along z for a 3D simulation; ``None`` (default) means 2D.
        lattice: Velocity-set name (``"D2Q9"``, ``"D3Q19"``) or a
            ``VelocitySet``; default D2Q9 in 2D and D3Q19 in 3D. After
            construction this attribute always holds the ``VelocitySet``.
        link_rules: Optional ``lbm.link_rules.LinkRules`` that replace plain
            bounce-back on selected links (interpolated bounce-back on curved
            bodies, outflow). Each rule must sit on a link from a fluid node
            to a solid node at rest.
        immersed: Optional ``lbm.ibm.ImmersedBoundary``: a body represented by
            surface markers and a local force (immersed boundary method)
            instead of solid nodes. Its stencil must stay clear of solid nodes.
    """

    nx: int
    ny: int
    tau: float
    collision: str = "bgk"
    magic: float = 3.0 / 16.0
    force: tuple[float, ...] | None = None
    solid: np.ndarray | None = field(default=None, repr=False)
    wall_velocity: np.ndarray | None = field(default=None, repr=False)
    dtype: str = "float32"
    nz: int | None = field(default=None, kw_only=True)
    lattice: str | VelocitySet | None = field(default=None, kw_only=True)
    link_rules: LinkRules | None = field(default=None, kw_only=True, repr=False)
    immersed: ImmersedBoundary | None = field(default=None, kw_only=True, repr=False)

    def __post_init__(self) -> None:
        sizes = {"nx": self.nx, "ny": self.ny} | ({} if self.nz is None else {"nz": self.nz})
        if any(n <= 0 for n in sizes.values()):
            got = ", ".join(f"{k}={v}" for k, v in sizes.items())
            raise ValueError(f"{' and '.join(sizes)} must be positive; got {got}")
        if self.collision not in COLLISION_MODELS:
            raise ValueError(f"collision must be one of {COLLISION_MODELS}; got {self.collision!r}")
        if not self.magic > 0:
            raise ValueError(f"TRT magic parameter must be > 0; got magic={self.magic}")
        if self.dtype not in SUPPORTED_DTYPES:
            raise ValueError(f"dtype must be one of {SUPPORTED_DTYPES}; got {self.dtype!r}")
        check_tau(self.tau)

        d = 2 if self.nz is None else 3
        lattice = _resolve_lattice(self.lattice, d)
        shape = self._shape_for(d)

        solid = np.zeros(shape, dtype=bool) if self.solid is None else np.asarray(self.solid)
        if solid.shape != shape:
            raise ValueError(f"solid mask must have shape {shape}; got {solid.shape}")
        solid = solid.astype(bool)

        if self.wall_velocity is None:
            wall = np.zeros((d, *shape))
        else:
            wall = np.asarray(self.wall_velocity, dtype=np.float64)
        if wall.shape != (d, *shape):
            raise ValueError(f"wall_velocity must have shape {(d, *shape)}; got {wall.shape}")
        wall = np.where(solid, wall, 0.0)

        force = (0.0,) * d if self.force is None else tuple(float(c) for c in self.force)
        if len(force) != d:
            raise ValueError(f"force must have {d} components for a {d}D grid; got {force}")

        if self.immersed is not None and self.immersed.d != d:
            raise ValueError(f"the immersed body is {self.immersed.d}D but the grid is {d}D")
        if self.link_rules is not None:
            self.link_rules.validate(lattice, solid, wall)

        object.__setattr__(self, "lattice", lattice)
        object.__setattr__(self, "solid", _frozen_copy(solid))
        object.__setattr__(self, "wall_velocity", _frozen_copy(wall))
        object.__setattr__(self, "force", force)
        object.__setattr__(self, "tau", float(self.tau))
        object.__setattr__(self, "magic", float(self.magic))

    def _shape_for(self, d: int) -> tuple[int, ...]:
        return (self.ny, self.nx) if d == 2 else (self.nz, self.ny, self.nx)

    @property
    def d(self) -> int:
        """Number of spatial dimensions (2 or 3)."""
        return self.lattice.d

    @property
    def shape(self) -> tuple[int, ...]:
        """Array shape of one field: (ny, nx) in 2D, (nz, ny, nx) in 3D."""
        return self._shape_for(self.d)

    @property
    def fluid(self) -> np.ndarray:
        """Boolean mask, True on fluid nodes."""
        return ~self.solid

    @property
    def viscosity(self) -> float:
        """Kinematic viscosity in lattice units, nu = cs^2 (tau - 1/2)."""
        return self.lattice.cs2 * (self.tau - 0.5)

    @property
    def relaxation_rates(self) -> tuple[float, float]:
        """(omega_plus, omega_minus) for the TRT operator; equal for BGK."""
        omega_plus = 1.0 / self.tau
        if self.collision == "bgk":
            return omega_plus, omega_plus
        tau_minus = 0.5 + self.magic / (self.tau - 0.5)
        return omega_plus, 1.0 / tau_minus
