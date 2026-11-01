"""Second-order discrete equilibrium, in shifted storage, for any velocity set.

The equilibrium is a truncated Hermite expansion of the Maxwell-Boltzmann
distribution, accurate to second order in the velocity::

    f_i^eq = w_i rho [1 + c_i.u / cs^2 + (c_i.u)^2 / (2 cs^4) - u.u / (2 cs^2)]
           = w_i rho [1 + 3 c_i.u + 4.5 (c_i.u)^2 - 1.5 u.u]          (cs^2 = 1/3)

Shifted storage keeps f~_i = f_i - w_i instead of f_i. At rest every f_i is
close to w_i, so f_i itself carries a large constant plus a tiny signal; in
float32 that signal loses about two decimal digits to round-off. Storing the
deviation keeps the full mantissa for the part that matters::

    f~_i^eq = w_i [drho + rho (3 c_i.u + 4.5 (c_i.u)^2 - 1.5 u.u)],   rho = 1 + drho

The same formula serves D2Q9 and D3Q19: only the velocity set and the number
of velocity components change. All functions here are written against an array
namespace ``xp`` (NumPy or CuPy), so the same code runs on the CPU and,
unfused, on the GPU.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from lbm.lattice import VelocitySet


def dot(c: Sequence[int], vector: Sequence) -> object:
    """c . vector for an integer lattice velocity c and a sequence of components.

    Components where c is zero are skipped, so the rest velocity gives the
    integer 0 and costs no array work.
    """
    return sum(c_a * v_a for c_a, v_a in zip(c, vector, strict=True) if c_a)


def equilibrium(lattice: VelocitySet, delta_rho, u: Sequence, xp=np):
    """Shifted equilibrium populations f~_i^eq for every direction.

    Args:
        lattice: Velocity set.
        delta_rho: Density deviation rho - 1, any shape.
        u: The d velocity components, each broadcastable with ``delta_rho``.
        xp: Array namespace (``numpy`` or ``cupy``).

    Returns:
        Array of shape ``(q, *shape)`` with the dtype of the inputs.
    """
    rho = 1.0 + delta_rho
    usq = 1.5 * sum(u_a * u_a for u_a in u)
    terms = []
    for c_i, w_i in zip(lattice.c, lattice.w, strict=True):
        cu = 3.0 * dot(c_i, u)
        terms.append(w_i * (delta_rho + rho * (cu + 0.5 * cu * cu - usq)))
    return xp.stack(terms)
