"""Conversion between physical units and lattice units.

Lattice units set dx = dt = 1 and the reference density rho0 = 1. Three free
choices fix the mapping from a physical problem (length L, velocity U,
viscosity nu):

* resolution N: lattice nodes across L, so dx = L / N
* lattice velocity u_lb: what U becomes on the lattice, so dt = u_lb dx / U
* the Reynolds number Re = U L / nu, which must be the same in both systems

From these::

    nu_lb = nu dt / dx^2 = u_lb N / Re
    tau   = nu_lb / cs^2 + 1/2 = 3 nu_lb + 1/2
    Ma    = u_lb / cs = u_lb sqrt(3)

The trade-off every LBM user faces is visible here: at fixed Re and N, a
smaller u_lb lowers the compressibility error (which scales as Ma^2) but drives
tau toward 1/2 (unstable) and needs more time steps per physical second.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

from lbm.lattice import CS2
from lbm.params import StabilityWarning, check_tau

MAX_LATTICE_VELOCITY = 0.1
"""Above this lattice velocity (Ma ~ 0.17) compressibility errors grow quickly."""


@dataclass(frozen=True)
class UnitConverter:
    """Maps one physical flow problem onto the lattice.

    Attributes:
        length: Characteristic physical length L [m].
        velocity: Characteristic physical velocity U [m/s].
        viscosity: Physical kinematic viscosity nu [m^2/s].
        resolution: Lattice nodes across L.
        lattice_velocity: U expressed in lattice units.
    """

    length: float
    velocity: float
    viscosity: float
    resolution: int
    lattice_velocity: float = 0.05

    def __post_init__(self) -> None:
        for name in ("length", "velocity", "viscosity", "resolution", "lattice_velocity"):
            if not getattr(self, name) > 0:
                raise ValueError(f"{name} must be positive; got {getattr(self, name)}")
        check_tau(self.tau)
        if self.lattice_velocity > MAX_LATTICE_VELOCITY:
            warnings.warn(
                f"lattice velocity {self.lattice_velocity} > {MAX_LATTICE_VELOCITY} "
                f"(Ma = {self.mach:.3f}): compressibility error ~ Ma^2 is no longer small",
                StabilityWarning,
                stacklevel=3,
            )

    @classmethod
    def from_reynolds(
        cls, reynolds: float, resolution: int, lattice_velocity: float = 0.05
    ) -> UnitConverter:
        """Dimensionless set-up: L = U = 1 and nu = 1/Re."""
        return cls(
            length=1.0,
            velocity=1.0,
            viscosity=1.0 / reynolds,
            resolution=resolution,
            lattice_velocity=lattice_velocity,
        )

    @property
    def reynolds(self) -> float:
        return self.velocity * self.length / self.viscosity

    @property
    def dx(self) -> float:
        """Physical size of one lattice spacing [m]."""
        return self.length / self.resolution

    @property
    def dt(self) -> float:
        """Physical duration of one time step [s]."""
        return self.lattice_velocity * self.dx / self.velocity

    @property
    def lattice_viscosity(self) -> float:
        return self.viscosity * self.dt / self.dx**2

    @property
    def tau(self) -> float:
        return self.lattice_viscosity / CS2 + 0.5

    @property
    def mach(self) -> float:
        return self.lattice_velocity / math.sqrt(CS2)

    def velocity_to_lattice(self, velocity: float) -> float:
        return velocity * self.dt / self.dx

    def velocity_to_physical(self, lattice_velocity: float) -> float:
        return lattice_velocity * self.dx / self.dt

    def time_to_physical(self, steps: float) -> float:
        return steps * self.dt

    def steps_for_time(self, time: float) -> int:
        """Number of whole time steps covering a physical duration."""
        return round(time / self.dt)
