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
"""

from __future__ import annotations

import numpy as np

from lbm.lattice import CX_TUPLE, CY_TUPLE, Q


def pull_stream(f_post, xp=np):
    """Return f_i(x) = f_post_i(x - c_i) with periodic wrap-around, as a new array."""
    return xp.stack(
        [xp.roll(f_post[i], shift=(CY_TUPLE[i], CX_TUPLE[i]), axis=(0, 1)) for i in range(Q)]
    )
