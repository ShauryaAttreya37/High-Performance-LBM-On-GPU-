"""Second-order discrete equilibrium, in shifted storage.

The equilibrium is a truncated Hermite expansion of the Maxwell-Boltzmann
distribution, accurate to second order in the velocity::

    f_i^eq = w_i rho [1 + c_i.u / cs^2 + (c_i.u)^2 / (2 cs^4) - u.u / (2 cs^2)]
           = w_i rho [1 + 3 c_i.u + 4.5 (c_i.u)^2 - 1.5 u.u]          (cs^2 = 1/3)

Shifted storage keeps f~_i = f_i - w_i instead of f_i. At rest every f_i is
close to w_i, so f_i itself carries a large constant plus a tiny signal; in
float32 that signal loses about two decimal digits to round-off. Storing the
deviation keeps the full mantissa for the part that matters::

    f~_i^eq = w_i [drho + rho (3 c_i.u + 4.5 (c_i.u)^2 - 1.5 u.u)],   rho = 1 + drho

All functions here are written against an array namespace ``xp`` (NumPy or
CuPy), so the same code runs on the CPU and, unfused, on the GPU.
"""

from __future__ import annotations

import numpy as np

from lbm.lattice import CX_TUPLE, CY_TUPLE, W_TUPLE, Q


def equilibrium(delta_rho, ux, uy, xp=np):
    """Shifted equilibrium populations f~_i^eq for every direction.

    Args:
        delta_rho: Density deviation rho - 1, any shape.
        ux, uy: Velocity components, broadcastable with ``delta_rho``.
        xp: Array namespace (``numpy`` or ``cupy``).

    Returns:
        Array of shape ``(Q, *shape)`` with the dtype of the inputs.
    """
    rho = 1.0 + delta_rho
    usq = 1.5 * (ux * ux + uy * uy)
    terms = []
    for i in range(Q):
        cu = 3.0 * (CX_TUPLE[i] * ux + CY_TUPLE[i] * uy)
        terms.append(W_TUPLE[i] * (delta_rho + rho * (cu + 0.5 * cu * cu - usq)))
    return xp.stack(terms)
