"""Guo forcing scheme for a body-force density F.

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

import numpy as np

from lbm.lattice import CX_TUPLE, CY_TUPLE, W_TUPLE, Q


def guo_source(ux, uy, force, omega_plus, omega_minus, xp=np):
    """Source term S_i for a uniform force, shape ``(Q, *ux.shape)``."""
    fx, fy = force
    u_dot_f = ux * fx + uy * fy
    a_plus = 1.0 - 0.5 * omega_plus
    a_minus = 1.0 - 0.5 * omega_minus
    terms = []
    for i in range(Q):
        c_dot_f = CX_TUPLE[i] * fx + CY_TUPLE[i] * fy
        c_dot_u = CX_TUPLE[i] * ux + CY_TUPLE[i] * uy
        symmetric = 9.0 * c_dot_u * c_dot_f - 3.0 * u_dot_f
        antisymmetric = 3.0 * c_dot_f
        terms.append(W_TUPLE[i] * (a_plus * symmetric + a_minus * antisymmetric))
    return xp.stack(terms)
