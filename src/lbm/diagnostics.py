"""Macroscopic fields and derived quantities, in 2D and 3D.

Solvers store post-collision populations f*. Their moments give the
macroscopic state at the same time level as the collision that produced them:

    rho   = 1 + sum_i f~*_i
    rho u = sum_i c_i f*_i - F/2

The -F/2 appears because one collision adds exactly F of momentum (see
``lbm.forcing``), while the physical velocity is defined half-way through
the forcing: rho u = sum_i c_i f_i + F/2 = sum_i c_i f*_i - F/2.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from lbm.collision import momentum
from lbm.lattice import VelocitySet


@dataclass(frozen=True, eq=False)
class MacroscopicFields:
    """Host-side snapshot of density and velocity, read-only.

    Attributes:
        rho: Density, shape (ny, nx) or (nz, ny, nx).
        u: Velocity components (ux, uy) or (ux, uy, uz), same shape as rho.
        solid: Solid mask; solid nodes report rho = 1 and u = 0.
    """

    rho: np.ndarray
    u: tuple[np.ndarray, ...]
    solid: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "u", tuple(self.u))
        for array in (self.rho, self.solid, *self.u):
            array.setflags(write=False)

    @property
    def d(self) -> int:
        return len(self.u)

    @property
    def ux(self) -> np.ndarray:
        return self.u[0]

    @property
    def uy(self) -> np.ndarray:
        return self.u[1]

    @property
    def uz(self) -> np.ndarray:
        if self.d < 3:
            raise AttributeError("2D fields have no uz component")
        return self.u[2]

    @property
    def speed(self) -> np.ndarray:
        return np.sqrt(sum(u_a * u_a for u_a in self.u))

    @property
    def total_mass(self) -> float:
        """Sum of rho over fluid nodes."""
        return float(self.rho[~self.solid].sum(dtype=np.float64))


def macroscopic_from_post_collision(
    lattice: VelocitySet, f_post, force: Sequence[float], solid, xp=np
):
    """(rho, u) on the device from stored post-collision populations; u is a tuple."""
    rho = 1.0 + f_post.sum(axis=0)
    u = tuple(
        (j_a - 0.5 * f_a) / rho for j_a, f_a in zip(momentum(lattice, f_post), force, strict=True)
    )
    return xp.where(solid, 1.0, rho), tuple(xp.where(solid, 0.0, u_a) for u_a in u)


def vorticity(*u: np.ndarray) -> np.ndarray:
    """Vorticity by central differences.

    2D, ``vorticity(ux, uy)``: the scalar omega_z = d(uy)/dx - d(ux)/dy.
    3D, ``vorticity(ux, uy, uz)``: the vector (omega_x, omega_y, omega_z),
    stacked along a new first axis. Arrays are indexed [y, x] / [z, y, x].
    """
    if len(u) == 2:
        ux, uy = u
        return np.gradient(uy, axis=1) - np.gradient(ux, axis=0)
    if len(u) == 3:
        ux, uy, uz = u

        def d(field, axis_name):
            return np.gradient(field, axis={"z": 0, "y": 1, "x": 2}[axis_name])

        return np.stack([d(uz, "y") - d(uy, "z"), d(ux, "z") - d(uz, "x"), d(uy, "x") - d(ux, "y")])
    raise ValueError(f"vorticity needs 2 or 3 velocity components; got {len(u)}")


def kinetic_energy(rho: np.ndarray, *u: np.ndarray) -> float:
    """Total kinetic energy sum(rho |u|^2 / 2) in lattice units."""
    return float(0.5 * np.sum(rho * sum(u_a * u_a for u_a in u), dtype=np.float64))
