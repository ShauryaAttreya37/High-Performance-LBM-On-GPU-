"""Halfway bounce-back on solid nodes, and builders for simple wall geometries.

Halfway bounce-back: a population that would stream from fluid node x into a
solid node is reflected back to x, arriving one time step later in the opposite
direction. The wall it bounced off therefore sits halfway between x and the
solid node. In the pull formulation, for every direction i whose source node
x - c_i is solid::

    f_i(x, t + 1) = f*_opp(i)(x, t) + 6 w_i rho_w (c_i . u_w)

The second term (Ladd 1994) transfers momentum from a wall moving with
velocity u_w, evaluated at the solid node x - c_i. This code uses
rho_w = rho_0 = 1; the error is O(Ma^2) and, for walls moving tangentially,
the terms of opposite links cancel so mass is still conserved exactly.

Known limitation: the effective wall position depends on the collision
operator. With BGK it drifts with tau (a viscosity-dependent slip); TRT with
magic parameter Lambda = 3/16 pins it exactly halfway for Poiseuille flow.
The Poiseuille validation case measures exactly this.

Shifted storage needs no special handling: w_i = w_opp(i), so the shift
cancels on both sides of the reflection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from lbm.lattice import CX_TUPLE, CY_TUPLE, OPPOSITE_TUPLE, W_TUPLE, Q

_OPPOSITE = list(OPPOSITE_TUPLE)


@dataclass(frozen=True)
class BounceBackLinks:
    """Precomputed links between fluid nodes and solid neighbours.

    Attributes:
        mask: Boolean, shape (Q, ny, nx). True where fluid node x pulls
            direction i from a solid node x - c_i.
        wall_term: Shape (Q, ny, nx). The moving-wall term 6 w_i (c_i . u_w)
            on those links, zero elsewhere.
    """

    mask: Any
    wall_term: Any


def bounce_back_links(solid, wall_velocity, xp=np, dtype=np.float64) -> BounceBackLinks:
    """Find every fluid-to-solid link and its moving-wall momentum term."""
    solid = xp.asarray(solid, dtype=bool)
    fluid = ~solid
    wall_x = xp.asarray(wall_velocity[0], dtype=dtype)
    wall_y = xp.asarray(wall_velocity[1], dtype=dtype)
    masks, terms = [], []
    for i in range(Q):
        shift = (CY_TUPLE[i], CX_TUPLE[i])
        # roll by +c_i puts the value at x - c_i onto x.
        from_solid = xp.roll(solid, shift, axis=(0, 1)) & fluid
        c_dot_uw = CX_TUPLE[i] * xp.roll(wall_x, shift, axis=(0, 1)) + CY_TUPLE[i] * xp.roll(
            wall_y, shift, axis=(0, 1)
        )
        term = (6.0 * W_TUPLE[i]) * c_dot_uw
        masks.append(from_solid)
        terms.append(xp.where(from_solid, term, 0.0).astype(dtype))
    return BounceBackLinks(mask=xp.stack(masks), wall_term=xp.stack(terms))


def apply_bounce_back(f_streamed, f_post, links: BounceBackLinks, xp=np):
    """Replace populations pulled from solid nodes by their reflection; returns a new array."""
    reflected = f_post[_OPPOSITE] + links.wall_term
    return xp.where(links.mask, reflected, f_streamed)


def channel_walls(nx: int, ny: int) -> np.ndarray:
    """Solid mask for a channel: bottom and top rows solid, periodic in x.

    With halfway bounce-back the walls sit at y = 0.5 and y = ny - 1.5, so the
    channel width is H = ny - 2.
    """
    if nx <= 0 or ny < 3:
        raise ValueError(f"channel needs nx > 0 and ny >= 3; got nx={nx}, ny={ny}")
    solid = np.zeros((ny, nx), dtype=bool)
    solid[0, :] = True
    solid[-1, :] = True
    return solid


def cavity_walls(n: int, lid_velocity: float) -> tuple[np.ndarray, np.ndarray]:
    """Square cavity of n x n fluid nodes inside a one-node solid frame.

    The top row (lid) moves in +x with ``lid_velocity``. The grid is
    (n + 2) x (n + 2); walls sit at 0.5 and n + 0.5, so the cavity side is n.

    Returns:
        ``(solid, wall_velocity)`` ready for ``SimulationParams``.
    """
    if n <= 0:
        raise ValueError(f"cavity size must be positive; got n={n}")
    size = n + 2
    solid = np.ones((size, size), dtype=bool)
    solid[1:-1, 1:-1] = False
    wall = np.zeros((2, size, size))
    wall[0, -1, :] = lid_velocity
    return solid, wall
