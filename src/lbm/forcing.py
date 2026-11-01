"""Guo forcing scheme for a body-force density F, for any velocity set.

Guo, Zheng & Shi (2002) showed that adding a body force correctly to second
order needs two changes:

1. The velocity used in the equilibrium includes half a time step of force::

       rho u = sum_i c_i f_i + F / 2

2. A source term is added in the collision::

       S_i = (1 - omega/2) w_i [3 (c_i - u).F + 9 (c_i.u)(c_i.F)]

For TRT the source is split like the populations, into a part even in c_i
(relaxed with omega_plus) and a part odd in c_i (relaxed with omega_minus)::

    S_i^+ = w_i [9 (c_i.u)(c_i.F) - 3 u.F]
    S_i^- = 3 w_i (c_i.F)
    S_i   = (1 - omega_plus/2) S_i^+ + (1 - omega_minus/2) S_i^-

Moments: sum_i S_i = 0 (no mass is created) and
sum_i c_i S_i = (1 - omega_minus/2) F. Together with the half-force velocity
shift, one collision changes the momentum by exactly F per time step.
With omega_plus == omega_minus this reduces to the BGK formula above.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from lbm.equilibrium import dot
from lbm.lattice import VelocitySet


def guo_source(
    lattice: VelocitySet,
    u: Sequence,
    force: Sequence[float],
    omega_plus: float,
    omega_minus: float,
    xp=np,
):
    """Source term S_i for a uniform force, shape ``(q, *u[0].shape)``."""
    u_dot_f = sum(u_a * f_a for u_a, f_a in zip(u, force, strict=True))
    a_plus = 1.0 - 0.5 * omega_plus
    a_minus = 1.0 - 0.5 * omega_minus
    terms = []
    for c_i, w_i in zip(lattice.c, lattice.w, strict=True):
        c_dot_f = dot(c_i, force)
        c_dot_u = dot(c_i, u)
        symmetric = 9.0 * c_dot_u * c_dot_f - 3.0 * u_dot_f
        antisymmetric = 3.0 * c_dot_f
        terms.append(w_i * (a_plus * symmetric + a_minus * antisymmetric))
    return xp.stack(terms)
