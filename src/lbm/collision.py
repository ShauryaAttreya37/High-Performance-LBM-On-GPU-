"""Node-local collision: TRT relaxation plus Guo forcing (BGK is a special case).

Every population is split into a part that is symmetric and a part that is
antisymmetric under velocity reversal (opp(i) is the direction with
c_opp(i) = -c_i)::

    f_i^+ = (f_i + f_opp(i)) / 2        f_i^- = (f_i - f_opp(i)) / 2

The symmetric part carries the even moments (density, momentum flux) and its
relaxation rate omega_plus = 1/tau sets the viscosity. The antisymmetric part
carries the odd moments (momentum, heat flux) and omega_minus is free; it
controls numerical properties such as where a bounce-back wall effectively
sits. The TRT update is::

    f*_i = f_i - omega_plus (f_i^+ - f_i^eq+) - omega_minus (f_i^- - f_i^eq-) + S_i

Writing n_i = f_i - f_i^eq for the non-equilibrium part, this is::

    f*_i = f_i - omega_plus (n_i + n_opp(i)) / 2 - omega_minus (n_i - n_opp(i)) / 2 + S_i

which is the form used here and, line for line, in the CUDA kernels.
BGK is the case omega_minus = omega_plus, where it collapses to
f*_i = f_i - omega (f_i - f_i^eq) + S_i.
"""

from __future__ import annotations

import numpy as np

from lbm.equilibrium import equilibrium
from lbm.forcing import guo_source
from lbm.lattice import CX_TUPLE, CY_TUPLE, OPPOSITE_TUPLE, Q

_OPPOSITE = list(OPPOSITE_TUPLE)


def momentum(f):
    """First moments (sum_i c_ix f_i, sum_i c_iy f_i) of a population array."""
    jx = sum(CX_TUPLE[i] * f[i] for i in range(Q) if CX_TUPLE[i] != 0)
    jy = sum(CY_TUPLE[i] * f[i] for i in range(Q) if CY_TUPLE[i] != 0)
    return jx, jy


def pre_collision_moments(f, force, xp=np):
    """Density deviation and Guo-corrected velocity from streamed populations.

    Returns:
        ``(delta_rho, ux, uy)`` where rho = 1 + delta_rho and
        rho u = sum_i c_i f_i + F/2.
    """
    delta_rho = f.sum(axis=0)
    rho = 1.0 + delta_rho
    jx, jy = momentum(f)
    fx, fy = force
    return delta_rho, (jx + 0.5 * fx) / rho, (jy + 0.5 * fy) / rho


def collide(f, omega_plus, omega_minus, force, xp=np):
    """Apply one TRT + Guo collision to every node of ``f`` (shape ``(Q, ...)``).

    Args:
        f: Shifted populations after streaming and boundary handling.
        omega_plus: Symmetric relaxation rate, 1/tau.
        omega_minus: Antisymmetric relaxation rate (equal to omega_plus for BGK).
        force: Uniform body-force density (Fx, Fy).
        xp: Array namespace.

    Returns:
        New array of post-collision shifted populations; ``f`` is not modified.
    """
    delta_rho, ux, uy = pre_collision_moments(f, force, xp)
    non_eq = f - equilibrium(delta_rho, ux, uy, xp)
    non_eq_opp = non_eq[_OPPOSITE]
    post = (
        f - (0.5 * omega_plus) * (non_eq + non_eq_opp) - (0.5 * omega_minus) * (non_eq - non_eq_opp)
    )
    if tuple(force) != (0.0, 0.0):
        post = post + guo_source(ux, uy, force, omega_plus, omega_minus, xp)
    return post
