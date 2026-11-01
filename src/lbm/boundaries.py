"""Halfway bounce-back on solid nodes, and builders for simple wall geometries.

Halfway bounce-back: a population that would stream from fluid node x into a
solid node is reflected back to x, arriving one time step later in the opposite
direction. The wall it bounced off therefore sits halfway between x and the
solid node. In the pull formulation, for every direction i whose source node
x - c_i is solid::

    f_i(x, t + 1) = f*_opp(i)(x, t) + 2 w_i rho_w (c_i . u_w) / cs^2

(= 6 w_i rho_w (c_i . u_w) for cs^2 = 1/3). The second term (Ladd 1994)
transfers momentum from a wall moving with velocity u_w, evaluated at the
solid node x - c_i. This code uses rho_w = rho_0 = 1; the error is O(Ma^2)
and, for walls moving tangentially, the terms of opposite links cancel so
mass is still conserved exactly.

Known limitation: the effective wall position depends on the collision
operator. With BGK it drifts with tau (a viscosity-dependent slip); TRT with
magic parameter Lambda = 3/16 pins it exactly halfway for Poiseuille flow.
The Poiseuille validation case measures exactly this.

Shifted storage needs no special handling: w_i = w_opp(i), so the shift
cancels on both sides of the reflection. Everything here works for 2D
(``[y, x]``) and 3D (``[z, y, x]``) arrays alike.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from lbm.lattice import VelocitySet
from lbm.streaming import index_shift


@dataclass(frozen=True)
class BounceBackLinks:
    """Precomputed links between fluid nodes and solid neighbours.

    Attributes:
        mask: Boolean, shape (q, *shape). True where fluid node x pulls
            direction i from a solid node x - c_i.
        wall_term: Shape (q, *shape). The moving-wall term
            2 w_i (c_i . u_w) / cs^2 on those links, zero elsewhere.
    """

    mask: Any
    wall_term: Any


def bounce_back_links(
    lattice: VelocitySet, solid, wall_velocity, xp=np, dtype=np.float64
) -> BounceBackLinks:
    """Find every fluid-to-solid link and its moving-wall momentum term."""
    solid = xp.asarray(solid, dtype=bool)
    fluid = ~solid
    walls = [xp.asarray(wall_velocity[a], dtype=dtype) for a in range(lattice.d)]
    axes = tuple(range(lattice.d))
    masks, terms = [], []
    for c_i, w_i in zip(lattice.c, lattice.w, strict=True):
        shift = index_shift(c_i)
        # roll by +c_i puts the value at x - c_i onto x.
        from_solid = xp.roll(solid, shift, axis=axes) & fluid
        c_dot_uw = sum(
            c_a * xp.roll(walls[a], shift, axis=axes) for a, c_a in enumerate(c_i) if c_a
        )
        term = (2.0 * w_i / lattice.cs2) * c_dot_uw
        masks.append(from_solid)
        terms.append(xp.where(from_solid, term, 0.0).astype(dtype))
    return BounceBackLinks(mask=xp.stack(masks), wall_term=xp.stack(terms))


def apply_bounce_back(lattice: VelocitySet, f_streamed, f_post, links: BounceBackLinks, xp=np):
    """Replace populations pulled from solid nodes by their reflection; returns a new array."""
    reflected = f_post[list(lattice.opposite)] + links.wall_term
    return xp.where(links.mask, reflected, f_streamed)


def channel_walls(nx: int, ny: int, nz: int | None = None) -> np.ndarray:
    """Solid mask with walls on the first and last y rows; periodic in x (and z).

    In 2D this is a channel, in 3D (``nz`` given) a slab between two plates.
    With halfway bounce-back the walls sit at y = 0.5 and y = ny - 1.5, so the
    gap is H = ny - 2.
    """
    if nx <= 0 or ny < 3 or (nz is not None and nz <= 0):
        raise ValueError(f"channel needs nx > 0, ny >= 3, nz > 0; got nx={nx}, ny={ny}, nz={nz}")
    shape = (ny, nx) if nz is None else (nz, ny, nx)
    solid = np.zeros(shape, dtype=bool)
    solid[..., 0, :] = True
    solid[..., -1, :] = True
    return solid


def cavity_walls(n: int, lid_velocity: float, dimensions: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Square (2D) or cubic (3D) cavity of side n inside a one-node solid frame.

    The top face (y = n + 1, the lid) moves in +x with ``lid_velocity``. The
    grid has n + 2 nodes per side; walls sit at 0.5 and n + 0.5, so the
    cavity side is n.

    Returns:
        ``(solid, wall_velocity)`` ready for ``SimulationParams``.
    """
    if dimensions not in (2, 3):
        raise ValueError(f"cavity dimensions must be 2 or 3; got {dimensions}")
    if n <= 0:
        raise ValueError(f"cavity size must be positive; got n={n}")
    size = n + 2
    solid = np.ones((size,) * dimensions, dtype=bool)
    solid[(slice(1, -1),) * dimensions] = False
    wall = np.zeros((dimensions, *solid.shape))
    wall[0, ..., -1, :] = lid_velocity
    return solid, wall
