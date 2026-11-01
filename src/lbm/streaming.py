"""Streaming: populations hop one lattice link along their velocity per time step.

Pull formulation: each node *gathers* what its neighbours sent it::

    f_i(x, t + 1) = f*_i(x - c_i, t)

The alternative, push, scatters f*_i(x) to x + c_i. Both give the same result;
pull is preferred on GPUs because the expensive operation, the global-memory
write, is then aligned and coalesced (every thread writes its own node), while
the misaligned accesses are reads, which caches handle well.

Here streaming uses ``roll``, so every domain edge is periodic. Walls are made
by marking nodes solid; bounce-back then replaces the populations that would
have been pulled out of a solid node (see ``lbm.boundaries``).

Arrays are indexed ``[y, x]`` or ``[z, y, x]`` while velocities are written
``(cx, cy[, cz])``, so the roll shift is the velocity in reverse order.
"""

from __future__ import annotations

import numpy as np

from lbm.lattice import VelocitySet


def index_shift(c) -> tuple[int, ...]:
    """Array-axis shift for lattice velocity ``c`` (axes ordered z, y, x)."""
    return tuple(reversed(c))


def pull_stream(lattice: VelocitySet, f_post, xp=np):
    """Return f_i(x) = f_post_i(x - c_i) with periodic wrap-around, as a new array."""
    axes = tuple(range(lattice.d))
    return xp.stack(
        [xp.roll(f_post[i], shift=index_shift(c_i), axis=axes) for i, c_i in enumerate(lattice.c)]
    )
