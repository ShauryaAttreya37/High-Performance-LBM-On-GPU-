"""Link rules: boundaries that set a streamed population from several post-collision ones.

Plain halfway bounce-back sets the population arriving at fluid node x from a
solid neighbour x - c_j to a single value, f_j(x, t+1) = f*_opp(j)(x, t).
Better boundaries need more: interpolated bounce-back mixes populations of
two nodes, an outflow copies them from upstream. All of them fit one shape,
an affine combination of post-collision populations of the previous step::

    f_j(x, t+1) = b + sum_m a_m f*_{k_m}(y_m, t)                      (one "rule")

A ``LinkRules`` object lists such rules in geometric terms (node indices and
lattice directions). Each solver *compiles* them into flat indices for the
place where its memory layout keeps f*_k(y), and into the slot its next
kernel reads as "the population that bounced back":

=================  ===========================  ===========================  ==========
layout             used by                      f*_k(y) is stored in slot    rule writes
=================  ===========================  ===========================  ==========
``reference``      numpy / cupy reference       (k, y) of f_post             (j, x) of the
                                                                             streamed array
``pull``           cuda-naive, before a step    (k, y)                       (opp j, x)
``aa_odd``         cuda-aa, before odd step     (opp k, y)                   (j, x)
``aa_even``        cuda-aa, before even step    (k, y + c_k), or (opp k, y)  (j, x)
                                                if y + c_k is solid
=================  ===========================  ===========================  ==========

For the GPU layouts the written slot is exactly the one the fused kernel's
bounce-back branch reads, so the kernels themselves are unchanged: a small
pre-pass evaluates every rule into a buffer, then scatters the buffer. Two
separate passes mean no rule can read a slot another rule has already
overwritten, whatever the geometry.

Rules are only allowed on links whose solid end does not move (bodies at
rest, outlets): the kernels add the moving-wall term of the solid node to the
value they read, and for rule links that term must be zero.

Shifted storage (f~ = f - w) needs no correction when the coefficients of a
rule sum to one and every term has the weight of direction j, which holds for
all rules built here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lbm.geometry import WallLinks, neighbour_index
from lbm.lattice import VelocitySet

LAYOUTS = ("reference", "pull", "aa_odd", "aa_even")


@dataclass(frozen=True)
class AffineSet:
    """n affine expressions over post-collision populations, padded to m terms each.

    value_r = constant[r] + sum_t coeff[r, t] * f*_{direction[r, t]}(node[r, t])
    Padding terms have coefficient 0 (and point at direction 0 of node 0).
    """

    direction: np.ndarray
    node: np.ndarray
    coeff: np.ndarray
    constant: np.ndarray

    def __len__(self) -> int:
        return len(self.constant)

    @property
    def terms(self) -> int:
        return self.coeff.shape[1]

    @staticmethod
    def from_lists(rows: list[list[tuple[int, int, float]]], constant=None) -> AffineSet:
        """Build from rows of (direction, node, coefficient) terms."""
        m = max((len(r) for r in rows), default=1)
        n = len(rows)
        direction = np.zeros((n, m), dtype=np.int64)
        node = np.zeros((n, m), dtype=np.int64)
        coeff = np.zeros((n, m))
        for r, row in enumerate(rows):
            for t, (k, y, a) in enumerate(row):
                direction[r, t], node[r, t], coeff[r, t] = k, y, a
        const = np.zeros(n) if constant is None else np.asarray(constant, dtype=np.float64)
        return AffineSet(direction, node, coeff, const)

    @staticmethod
    def concat(*sets: AffineSet) -> AffineSet:
        sets = tuple(s for s in sets if len(s))
        if not sets:
            return AffineSet.from_lists([])
        m = max(s.terms for s in sets)

        def pad(a, fill=0):
            return np.pad(a, ((0, 0), (0, m - a.shape[1])), constant_values=fill)

        return AffineSet(
            np.concatenate([pad(s.direction) for s in sets]),
            np.concatenate([pad(s.node) for s in sets]),
            np.concatenate([pad(s.coeff) for s in sets]),
            np.concatenate([s.constant for s in sets]),
        )


@dataclass(frozen=True)
class LinkRules:
    """Rules f_j(x, t+1) = values[r] for links (node[r], direction[r])."""

    node: np.ndarray
    direction: np.ndarray
    values: AffineSet

    def __post_init__(self) -> None:
        if not (len(self.node) == len(self.direction) == len(self.values)):
            raise ValueError("node, direction and values must have the same length")
        keys = np.stack([self.node, self.direction])
        if len(self.node) and np.unique(keys, axis=1).shape[1] != len(self.node):
            raise ValueError("a link has more than one rule")

    def __len__(self) -> int:
        return len(self.node)

    @staticmethod
    def concat(*rules: LinkRules) -> LinkRules:
        rules = tuple(r for r in rules if r is not None)
        return LinkRules(
            np.concatenate([r.node for r in rules]).astype(np.int64),
            np.concatenate([r.direction for r in rules]).astype(np.int64),
            AffineSet.concat(*(r.values for r in rules)),
        )

    def validate(self, lattice: VelocitySet, solid: np.ndarray, wall_velocity: np.ndarray) -> None:
        """Every rule must sit on a fluid-to-resting-solid link."""
        shape = solid.shape
        flat_solid = solid.ravel()
        if np.any(self.direction >= lattice.q) or np.any(self.direction < 0):
            raise ValueError(f"rule directions must be in [0, {lattice.q})")
        if np.any(flat_solid[self.node]):
            raise ValueError("link rules must target fluid nodes")
        for j in np.unique(self.direction):
            nodes = self.node[self.direction == j]
            source = neighbour_index(nodes, [-c for c in lattice.c[j]], shape)
            if not np.all(flat_solid[source]):
                raise ValueError(f"a rule for direction {j} does not start at a solid node")
            moving = np.abs(wall_velocity.reshape(lattice.d, -1)[:, source]).max(initial=0.0)
            if moving > 0:
                raise ValueError("link rules are only allowed next to solid nodes at rest")


# ---------------------------------------------------------------- builders


def bouzidi_rules(lattice: VelocitySet, links: WallLinks, solid: np.ndarray) -> LinkRules:
    """Linear interpolated bounce-back (Bouzidi, Firdaouss & Lallemand 2001).

    For a link from fluid node x towards the body node x - c_j, with the wall
    at fraction q, and i = opp(j) the direction pointing into the wall::

        q <  1/2:  f_j(x) = 2q f*_i(x) + (1 - 2q) f*_i(x + c_j)
        q >= 1/2:  f_j(x) = f*_i(x) / (2q) + (1 - 1/(2q)) f*_j(x)

    Both are linear interpolations that place the reflection point exactly
    on the wall; q = 1/2 reduces to plain bounce-back. Where the q < 1/2
    formula needs x + c_j but that node is solid (a gap one node wide), the
    link keeps plain bounce-back.
    """
    shape = solid.shape
    flat_solid = np.asarray(solid, dtype=bool).ravel()
    opposite = np.asarray(lattice.opposite)
    rows, nodes, directions = [], [], []
    for x, j, q in zip(links.node, links.direction, links.q, strict=True):
        i = int(opposite[j])
        if q < 0.5:
            behind = int(neighbour_index([x], lattice.c[j], shape)[0])
            if flat_solid[behind]:
                continue
            row = [(i, int(x), 2.0 * q), (i, behind, 1.0 - 2.0 * q)]
        else:
            row = [(i, int(x), 0.5 / q), (int(j), int(x), 1.0 - 0.5 / q)]
        rows.append(row)
        nodes.append(int(x))
        directions.append(int(j))
    return LinkRules(
        np.asarray(nodes, dtype=np.int64),
        np.asarray(directions, dtype=np.int64),
        AffineSet.from_lists(rows),
    )


# ---------------------------------------------------------------- compilation


@dataclass(frozen=True)
class CompiledAffine:
    """An ``AffineSet`` resolved to flat slots of one memory layout (host arrays)."""

    slots: np.ndarray
    coeff: np.ndarray
    constant: np.ndarray


def compile_affine(
    lattice: VelocitySet,
    values: AffineSet,
    layout: str,
    solid: np.ndarray,
    wall_velocity: np.ndarray,
) -> CompiledAffine:
    """Flat slots (into the (q, *shape) population buffer) for every term."""
    if layout not in LAYOUTS:
        raise ValueError(f"unknown layout {layout!r}; choose from {LAYOUTS}")
    shape = solid.shape
    n_nodes = int(np.prod(shape))
    k = values.direction
    y = values.node
    constant = values.constant.copy()
    opposite = np.asarray(lattice.opposite)
    if layout in ("reference", "pull"):
        slots = k * n_nodes + y
    elif layout == "aa_odd":
        slots = opposite[k] * n_nodes + y
    else:
        # After the odd step f*_k(y) has been pushed to y + c_k, unless that
        # node is solid: then it was reflected into slot (opp k, y), with the
        # moving-wall term 6 w_opp(k) (c_opp(k) . u_w) added on the way.
        # Reading the stored value back therefore needs + 6 w_k (c_k . u_w).
        slots = np.empty_like(k)
        wall_terms = np.zeros(k.shape)
        flat_solid = np.asarray(solid, dtype=bool).ravel()
        walls = np.asarray(wall_velocity, dtype=np.float64).reshape(lattice.d, -1)
        for kk in np.unique(k):
            sel = k == kk
            dest = neighbour_index(y[sel], lattice.c[kk], shape)
            blocked = flat_solid[dest]
            slots[sel] = np.where(blocked, opposite[kk] * n_nodes + y[sel], kk * n_nodes + dest)
            c_dot_uw = np.tensordot(np.asarray(lattice.c[kk], dtype=np.float64), walls[:, dest], 1)
            wall_terms[sel] = (2.0 * lattice.w[kk] / lattice.cs2) * c_dot_uw * blocked
        constant += (values.coeff * wall_terms).sum(axis=1)
    return CompiledAffine(slots=slots, coeff=values.coeff, constant=constant)


def compile_targets(lattice: VelocitySet, rules: LinkRules, layout: str, n_nodes: int):
    """Flat slot each rule writes, for the given layout."""
    if layout not in LAYOUTS:
        raise ValueError(f"unknown layout {layout!r}; choose from {LAYOUTS}")
    direction = rules.direction
    if layout == "pull":
        direction = np.asarray(lattice.opposite)[direction]
    return direction * n_nodes + rules.node


@dataclass(frozen=True)
class CompiledRules:
    target: np.ndarray
    values: CompiledAffine


def compile_rules(
    lattice: VelocitySet, rules: LinkRules, layout: str, solid, wall_velocity
) -> CompiledRules:
    n_nodes = int(np.prod(solid.shape))
    return CompiledRules(
        target=compile_targets(lattice, rules, layout, n_nodes),
        values=compile_affine(lattice, rules.values, layout, solid, wall_velocity),
    )


def evaluate_affine(compiled, flat_populations, xp=np):
    """Values of compiled expressions (device arrays in ``compiled``) on a flat buffer."""
    gathered = flat_populations[compiled.slots]
    return compiled.constant + (compiled.coeff * gathered).sum(axis=1)
