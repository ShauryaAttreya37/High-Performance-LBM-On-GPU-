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
f*_i = f_i - omega (f_i - f_i^eq) + S_i. Nothing in it depends on the
dimension: D2Q9 and D3Q19 differ only in their velocity sets.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from lbm.equilibrium import equilibrium
from lbm.forcing import guo_source
from lbm.lattice import VelocitySet


def momentum(lattice: VelocitySet, f) -> tuple:
    """First moments (sum_i c_ia f_i for each axis a) of a population array."""
    return tuple(
        sum(c_a * f[i] for i, c_a in enumerate(lattice.component(axis)) if c_a)
        for axis in range(lattice.d)
    )


def pre_collision_moments(lattice: VelocitySet, f, force: Sequence[float]):
    """Density deviation and Guo-corrected velocity from streamed populations.

    Returns:
        ``(delta_rho, u)`` with rho = 1 + delta_rho, ``u`` a tuple of d
        components, and rho u = sum_i c_i f_i + F/2.
    """
    delta_rho = f.sum(axis=0)
    rho = 1.0 + delta_rho
    u = tuple((j_a + 0.5 * f_a) / rho for j_a, f_a in zip(momentum(lattice, f), force, strict=True))
    return delta_rho, u


def collide(
    lattice: VelocitySet,
    f,
    omega_plus: float,
    omega_minus: float,
    force: Sequence[float],
    xp=np,
):
    """Apply one TRT + Guo collision to every node of ``f`` (shape ``(q, ...)``).

    Args:
        lattice: Velocity set.
        f: Shifted populations after streaming and boundary handling.
        omega_plus: Symmetric relaxation rate, 1/tau.
        omega_minus: Antisymmetric relaxation rate (equal to omega_plus for BGK).
        force: Body-force density, d components, each a scalar or a field-shaped array.
        xp: Array namespace.

    Returns:
        New array of post-collision shifted populations; ``f`` is not modified.
    """
    delta_rho, u = pre_collision_moments(lattice, f, force)
    non_eq = f - equilibrium(lattice, delta_rho, u, xp)
    non_eq_opp = non_eq[list(lattice.opposite)]
    post = (
        f - (0.5 * omega_plus) * (non_eq + non_eq_opp) - (0.5 * omega_minus) * (non_eq - non_eq_opp)
    )
    if any(not _is_zero(f_a) for f_a in force):
        post = post + guo_source(lattice, u, force, omega_plus, omega_minus, xp)
    return post


def _is_zero(value) -> bool:
    """True for a scalar zero force component; arrays always count as a force."""
    return isinstance(value, int | float) and value == 0.0
