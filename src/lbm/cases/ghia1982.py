"""Lid-driven cavity reference data of Ghia, Ghia & Shin (1982).

U. Ghia, K. N. Ghia, C. T. Shin, "High-Re solutions for incompressible flow
using the Navier-Stokes equations and a multigrid method", Journal of
Computational Physics 48 (1982) 387-411. Tables I and II: velocity along the
cavity centrelines from a 129 x 129 (Re <= 1000) vorticity-stream-function
multigrid solution. Coordinates and velocities are normalised by the cavity
side and the lid velocity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CenterlineData:
    y: np.ndarray
    """Positions along the vertical centreline x = 0.5."""
    u: np.ndarray
    """Horizontal velocity u / U at those positions."""
    x: np.ndarray
    """Positions along the horizontal centreline y = 0.5."""
    v: np.ndarray
    """Vertical velocity v / U at those positions."""


_Y = (1.0000, 0.9766, 0.9688, 0.9609, 0.9531, 0.8516, 0.7344, 0.6172, 0.5000,
      0.4531, 0.2813, 0.1719, 0.1016, 0.0703, 0.0625, 0.0547, 0.0000)  # fmt: skip
_X = (1.0000, 0.9688, 0.9609, 0.9531, 0.9453, 0.9063, 0.8594, 0.8047, 0.5000,
      0.2344, 0.2266, 0.1563, 0.0938, 0.0781, 0.0703, 0.0625, 0.0000)  # fmt: skip

_U = {
    100: (1.00000, 0.84123, 0.78871, 0.73722, 0.68717, 0.23151, 0.00332, -0.13641, -0.20581,
          -0.21090, -0.15662, -0.10150, -0.06434, -0.04775, -0.04192, -0.03717, 0.00000),
    400: (1.00000, 0.75837, 0.68439, 0.61756, 0.55892, 0.29093, 0.16256, 0.02135, -0.11477,
          -0.17119, -0.32726, -0.24299, -0.14612, -0.10338, -0.09266, -0.08186, 0.00000),
    1000: (1.00000, 0.65928, 0.57492, 0.51117, 0.46604, 0.33304, 0.18719, 0.05702, -0.06080,
           -0.10648, -0.27805, -0.38289, -0.29730, -0.22220, -0.20196, -0.18109, 0.00000),
}  # fmt: skip

_V = {
    100: (0.00000, -0.05906, -0.07391, -0.08864, -0.10313, -0.16914, -0.22445, -0.24533,
          0.05454, 0.17527, 0.17507, 0.16077, 0.12317, 0.10890, 0.10091, 0.09233, 0.00000),
    400: (0.00000, -0.12146, -0.15663, -0.19254, -0.22847, -0.23827, -0.44993, -0.38598,
          0.05186, 0.30174, 0.30203, 0.28124, 0.22965, 0.20920, 0.19713, 0.18360, 0.00000),
    1000: (0.00000, -0.21388, -0.27669, -0.33714, -0.39188, -0.51550, -0.42665, -0.31966,
           0.02526, 0.32235, 0.33075, 0.37095, 0.32627, 0.30353, 0.29012, 0.27485, 0.00000),
}  # fmt: skip

REYNOLDS_NUMBERS = tuple(sorted(_U))


def ghia_centerlines(reynolds: int) -> CenterlineData:
    """Reference centreline velocities for Re in ``REYNOLDS_NUMBERS``."""
    if reynolds not in _U:
        raise ValueError(f"Ghia et al. data available for Re in {REYNOLDS_NUMBERS}; got {reynolds}")

    def frozen(values):
        array = np.array(values, dtype=np.float64)
        array.setflags(write=False)
        return array

    return CenterlineData(
        y=frozen(_Y), u=frozen(_U[reynolds]), x=frozen(_X), v=frozen(_V[reynolds])
    )
