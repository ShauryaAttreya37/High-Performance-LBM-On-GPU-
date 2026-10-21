"""Macroscopic fields and derived quantities.

Solvers store post-collision populations f*. Their moments give the
macroscopic state at the same time level as the collision that produced them:

    rho   = 1 + sum_i f~*_i
    rho u = sum_i c_i f*_i - F/2

The -F/2 appears because one collision adds exactly F of momentum (see
``lbm.forcing``), while the physical velocity is defined half-way through
the forcing: rho u = sum_i c_i f_i + F/2 = sum_i c_i f*_i - F/2.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lbm.collision import momentum


@dataclass(frozen=True, eq=False)
class MacroscopicFields:
    """Host-side snapshot of density and velocity, shape (ny, nx), read-only.

    Solid nodes report rho = 1 and u = 0.
    """

    rho: np.ndarray
    ux: np.ndarray
    uy: np.ndarray
    solid: np.ndarray

    def __post_init__(self) -> None:
        for name in ("rho", "ux", "uy", "solid"):
            getattr(self, name).setflags(write=False)

    @property
    def speed(self) -> np.ndarray:
        return np.hypot(self.ux, self.uy)

    @property
    def total_mass(self) -> float:
        """Sum of rho over fluid nodes."""
        return float(self.rho[~self.solid].sum(dtype=np.float64))


def macroscopic_from_post_collision(f_post, force, solid, xp=np):
    """(rho, ux, uy) on the device from stored post-collision populations."""
    rho = 1.0 + f_post.sum(axis=0)
    jx, jy = momentum(f_post)
    fx, fy = force
    ux = (jx - 0.5 * fx) / rho
    uy = (jy - 0.5 * fy) / rho
    return xp.where(solid, 1.0, rho), xp.where(solid, 0.0, ux), xp.where(solid, 0.0, uy)


def vorticity(ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
    """Out-of-plane vorticity omega_z = d(uy)/dx - d(ux)/dy (central differences)."""
    return np.gradient(uy, axis=1) - np.gradient(ux, axis=0)


def kinetic_energy(rho: np.ndarray, ux: np.ndarray, uy: np.ndarray) -> float:
    """Total kinetic energy sum(rho |u|^2 / 2) in lattice units."""
    return float(0.5 * np.sum(rho * (ux * ux + uy * uy), dtype=np.float64))
