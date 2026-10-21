"""Velocity sets (DdQq lattices): the single source of truth for lattice constants.

A velocity set fixes the discrete velocities c_i and weights w_i of a lattice
Boltzmann model. Every Python module and every CUDA kernel takes its lattice
constants from a ``VelocitySet``; the CUDA preamble is generated from it when a
kernel is compiled, so the CPU reference and the GPU kernels cannot disagree.

D2Q9 direction numbering (x to the right, y up)::

    6   2   5
      \\ | /
    3 - 0 - 1
      / | \\
    7   4   8

D3Q19: the rest velocity, the 6 axis links (+x, -x, +y, -y, +z, -z) and the 12
edge links (pairs of non-zero components), each listed next to its opposite.

Weights: D2Q9 4/9 (rest), 1/9 (axis), 1/36 (diagonal);
D3Q19 1/3 (rest), 1/18 (axis), 1/36 (edge). Both have cs^2 = 1/3.

These weights make the discrete velocity moments match those of the
continuous Maxwell-Boltzmann distribution up to fourth order::

    sum_i w_i                     = 1
    sum_i w_i c_ia                = 0
    sum_i w_i c_ia c_ib           = cs^2 delta_ab
    sum_i w_i c_ia c_ib c_ic      = 0
    sum_i w_i c_ia c_ib c_ic c_id = cs^4 (d_ab d_cd + d_ac d_bd + d_ad d_bc)

which is exactly what the Chapman-Enskog expansion needs to recover the
weakly compressible Navier-Stokes equations.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np

_SUPPORTED_REAL_TYPES = ("float", "double")
_WEIGHT_SUM_TOLERANCE = 1e-12


def _read_only(values, dtype) -> np.ndarray:
    array = np.array(values, dtype=dtype)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class VelocitySet:
    """A DdQq lattice: ``d`` dimensions, ``q = len(c)`` discrete velocities.

    Attributes:
        name: Conventional name, e.g. ``"D3Q19"``.
        d: Number of spatial dimensions.
        c: The q velocity vectors, each a tuple of d integers.
        w: The q weights.
        cs2: Squared lattice speed of sound.

    Scalar loops should use the tuples (``c``, ``w``, ``component(axis)``):
    Python ints and floats are "weak" scalars under NumPy 2 type promotion, so
    ``float32_array * w[i]`` stays float32, whereas multiplying by a NumPy
    float64 scalar would silently upcast the whole array. The ``*_array`` views
    are for vectorised checks.
    """

    name: str
    d: int
    c: tuple[tuple[int, ...], ...]
    w: tuple[float, ...]
    cs2: float = 1.0 / 3.0

    def __post_init__(self) -> None:
        if len(self.w) != len(self.c):
            raise ValueError(f"{self.name}: {len(self.c)} velocities but {len(self.w)} weights")
        if any(len(v) != self.d for v in self.c):
            raise ValueError(f"{self.name}: every velocity must have length d={self.d}")
        if abs(sum(self.w) - 1.0) > _WEIGHT_SUM_TOLERANCE:
            raise ValueError(f"{self.name}: weights must sum to 1, got {sum(self.w)}")
        missing = [v for v in self.c if tuple(-a for a in v) not in self.c]
        if missing:
            raise ValueError(f"{self.name}: no opposite velocity for {missing}")

    @property
    def q(self) -> int:
        """Number of discrete velocities."""
        return len(self.c)

    @cached_property
    def opposite(self) -> tuple[int, ...]:
        """Index of the velocity pointing the other way: c[opposite[i]] == -c[i]."""
        return tuple(self.c.index(tuple(-a for a in v)) for v in self.c)

    def component(self, axis: int) -> tuple[int, ...]:
        """Velocity components along ``axis`` (0 = x, 1 = y, 2 = z); zeros beyond ``d``."""
        if axis >= self.d:
            return (0,) * self.q
        return tuple(v[axis] for v in self.c)

    @cached_property
    def c_array(self) -> np.ndarray:
        """Read-only (q, d) integer array of velocities."""
        return _read_only(self.c, np.int64)

    @cached_property
    def w_array(self) -> np.ndarray:
        """Read-only (q,) float64 array of weights."""
        return _read_only(self.w, np.float64)


D2Q9 = VelocitySet(
    name="D2Q9",
    d=2,
    c=((0, 0), (1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, 1), (-1, -1), (1, -1)),
    w=(4 / 9, *(1 / 9,) * 4, *(1 / 36,) * 4),
)

D3Q19 = VelocitySet(
    name="D3Q19",
    d=3,
    c=(
        (0, 0, 0),
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
        (1, 1, 0),
        (-1, -1, 0),
        (1, -1, 0),
        (-1, 1, 0),
        (1, 0, 1),
        (-1, 0, -1),
        (1, 0, -1),
        (-1, 0, 1),
        (0, 1, 1),
        (0, -1, -1),
        (0, 1, -1),
        (0, -1, 1),
    ),  # fmt: skip
    w=(1 / 3, *(1 / 18,) * 6, *(1 / 36,) * 12),
)

LATTICES: dict[str, VelocitySet] = {vs.name: vs for vs in (D2Q9, D3Q19)}
"""Every velocity set the engine supports, by name."""

_DEFAULT_BY_DIMENSION = {2: D2Q9, 3: D3Q19}


def default_lattice(d: int) -> VelocitySet:
    """The standard velocity set for ``d`` dimensions (D2Q9 or D3Q19)."""
    if d not in _DEFAULT_BY_DIMENSION:
        raise ValueError(f"no default lattice for dimension {d}; supported: 2, 3")
    return _DEFAULT_BY_DIMENSION[d]


# ---- Transitional D2Q9 aliases ------------------------------------------------
# Used by the physics modules until they take a VelocitySet (build step 3 of the
# dimension-generic refactor); removed then.
Q = D2Q9.q
D = D2Q9.d
CS2 = D2Q9.cs2
CX_TUPLE = D2Q9.component(0)
CY_TUPLE = D2Q9.component(1)
W_TUPLE = D2Q9.w
OPPOSITE_TUPLE = D2Q9.opposite
CX = _read_only(CX_TUPLE, np.int64)
CY = _read_only(CY_TUPLE, np.int64)
W = D2Q9.w_array
OPPOSITE = _read_only(OPPOSITE_TUPLE, np.int64)
# -------------------------------------------------------------------------------


def _accessor(return_type: str, name: str, values) -> str:
    """``constexpr`` function returning ``values[i]`` as a ternary chain."""
    if len(set(values)) == 1:
        body = f"{values[0]}"
    else:
        body = "".join(f"i == {i} ? {v} : " for i, v in enumerate(values[:-1])) + f"{values[-1]}"
    return f"__host__ __device__ constexpr {return_type} {name}(int i) {{ return {body}; }}"


def cuda_preamble(lattice: VelocitySet, real_type: str) -> str:
    """Return CUDA C++ source declaring ``real_t`` and the lattice constants.

    Emits ``DIM``, ``Q`` and ``constexpr`` accessor functions ``cx(i)``,
    ``cy(i)``, ``cz(i)``, ``opp(i)``, ``weight(i)``. For a 2D lattice ``cz(i)``
    returns 0 for every direction, so kernels written for three components
    compile down to 2D code with the z terms folded away.

    Accessor functions are used instead of ``__constant__`` arrays because
    kernels loop over directions with ``#pragma unroll``, which makes ``i`` a
    compile-time constant in every iteration: the compiler folds each call
    into an immediate value, and per-thread arrays such as ``real_t f[Q]`` are
    indexed only by constants and can live entirely in registers. Indexing a
    ``__constant__`` array would be a runtime load, and indexing ``f[opp[i]]``
    with a runtime value would push ``f`` out to (slow) local memory.

    Args:
        lattice: The velocity set.
        real_type: ``"float"`` or ``"double"``; becomes ``typedef ... real_t``.

    Returns:
        Source text to prepend to every kernel file.
    """
    if real_type not in _SUPPORTED_REAL_TYPES:
        raise ValueError(f"unsupported real type {real_type!r}; use one of {_SUPPORTED_REAL_TYPES}")
    weights = [f"(real_t){w!r}" for w in lattice.w]
    return "\n".join(
        [
            f"// ---- Generated by lbm.lattice.cuda_preamble({lattice.name}); do not edit. ----",
            f"typedef {real_type} real_t;",
            f"#define DIM {lattice.d}",
            f"#define Q {lattice.q}",
            _accessor("int", "cx", lattice.component(0)),
            _accessor("int", "cy", lattice.component(1)),
            _accessor("int", "cz", lattice.component(2)),
            _accessor("int", "opp", lattice.opposite),
            _accessor("real_t", "weight", weights),
            "// ---- End of generated preamble. ----",
            "",
        ]
    )
