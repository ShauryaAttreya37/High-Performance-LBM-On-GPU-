"""Body geometry on the lattice: shapes, voxelisation and wall-link fractions.

Coordinates: node (x, y[, z]) sits at the integer position equal to its index,
so the node stored at array index [y, x] or [z, y, x] is at (x, y) or
(x, y, z). Shapes are described in those continuous coordinates.

A body enters the solver in two ways:

1. ``voxelize`` marks every node whose position lies inside the body as solid
   (the "staircase" approximation that plain bounce-back sees).
2. ``wall_links`` finds every link from a fluid node x to a body node
   x - c_j and measures where the true surface cuts it::

       q = |x - x_wall| / |c_j|,   0 < q <= 1

   Plain halfway bounce-back assumes q = 1/2 everywhere; interpolated
   bounce-back (``lbm.link_rules.bouzidi_rules``) uses the measured q, which
   moves the wall to its true sub-grid position.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from lbm.lattice import VelocitySet


class Shape(Protocol):
    """A closed body in d dimensions."""

    @property
    def d(self) -> int: ...

    def contains(self, points: np.ndarray) -> np.ndarray:
        """True for points (shape (..., d), ordered x, y[, z]) inside the body."""

    def ray_fraction(self, origin: np.ndarray, direction: np.ndarray) -> np.ndarray:
        """First t in (0, 1] where origin + t * direction meets the surface.

        ``origin`` (outside) and ``direction`` have shape (n, d); the segment
        must end inside the body.
        """


@dataclass(frozen=True)
class Ball:
    """Circle (d = 2) or sphere (d = 3) with the given centre and radius."""

    center: tuple[float, ...]
    radius: float

    def __post_init__(self) -> None:
        if len(self.center) not in (2, 3):
            raise ValueError(f"a ball needs a 2D or 3D centre; got {self.center}")
        if not self.radius > 0:
            raise ValueError(f"radius must be positive; got {self.radius}")
        object.__setattr__(self, "center", tuple(float(c) for c in self.center))

    @property
    def d(self) -> int:
        return len(self.center)

    @property
    def measure(self) -> float:
        """Area (2D) or volume (3D)."""
        if self.d == 2:
            return math.pi * self.radius**2
        return 4.0 / 3.0 * math.pi * self.radius**3

    def contains(self, points: np.ndarray) -> np.ndarray:
        offset = np.asarray(points, dtype=np.float64) - np.asarray(self.center)
        return np.sum(offset * offset, axis=-1) <= self.radius**2

    def ray_fraction(self, origin: np.ndarray, direction: np.ndarray) -> np.ndarray:
        # |o + t d - c|^2 = R^2  ->  a t^2 + b t + k = 0; the first root is the entry point.
        o = np.asarray(origin, dtype=np.float64) - np.asarray(self.center)
        d = np.asarray(direction, dtype=np.float64)
        a = np.sum(d * d, axis=-1)
        b = 2.0 * np.sum(o * d, axis=-1)
        k = np.sum(o * o, axis=-1) - self.radius**2
        disc = np.maximum(b * b - 4.0 * a * k, 0.0)
        t = (-b - np.sqrt(disc)) / (2.0 * a)
        return np.clip(t, np.finfo(np.float64).tiny, 1.0)


@dataclass(frozen=True)
class Pipe:
    """Solid *outside* a circular cylinder whose axis runs along x (3D).

    The fluid region is the inside, y-z distance from (y, z) = center below
    ``radius``; everything else is wall.
    """

    center: tuple[float, float]
    radius: float

    def __post_init__(self) -> None:
        if len(self.center) != 2:
            raise ValueError(f"a pipe axis needs a (y, z) centre; got {self.center}")
        if not self.radius > 0:
            raise ValueError(f"radius must be positive; got {self.radius}")
        object.__setattr__(self, "center", tuple(float(c) for c in self.center))

    @property
    def d(self) -> int:
        return 3

    def contains(self, points: np.ndarray) -> np.ndarray:
        offset = np.asarray(points, dtype=np.float64)[..., 1:] - np.asarray(self.center)
        return np.sum(offset * offset, axis=-1) >= self.radius**2

    def ray_fraction(self, origin: np.ndarray, direction: np.ndarray) -> np.ndarray:
        # The origin is inside the circle (k < 0), so the exit point is the positive root.
        o = np.asarray(origin, dtype=np.float64)[..., 1:] - np.asarray(self.center)
        d = np.asarray(direction, dtype=np.float64)[..., 1:]
        a = np.sum(d * d, axis=-1)
        b = 2.0 * np.sum(o * d, axis=-1)
        k = np.sum(o * o, axis=-1) - self.radius**2
        disc = np.maximum(b * b - 4.0 * a * k, 0.0)
        t = (-b + np.sqrt(disc)) / (2.0 * a)
        return np.clip(t, np.finfo(np.float64).tiny, 1.0)


def circle(x: float, y: float, radius: float) -> Ball:
    return Ball((x, y), radius)


def sphere(x: float, y: float, z: float, radius: float) -> Ball:
    return Ball((x, y, z), radius)


def node_positions(shape: tuple[int, ...]) -> np.ndarray:
    """Positions of every node, shape (*shape, d), components ordered x, y[, z]."""
    grids = np.meshgrid(*(np.arange(n, dtype=np.float64) for n in shape), indexing="ij")
    return np.stack(grids[::-1], axis=-1)


def voxelize(body: Shape, shape: tuple[int, ...]) -> np.ndarray:
    """Boolean mask of the nodes inside ``body`` on a grid of array shape ``shape``."""
    if len(shape) != body.d:
        raise ValueError(f"grid shape {shape} does not match a {body.d}D body")
    return body.contains(node_positions(shape))


def neighbour_index(nodes, c, shape: tuple[int, ...]) -> np.ndarray:
    """Flat index of node + c (periodic wrap), for flat node indices ``nodes``."""
    coords = np.unravel_index(np.asarray(nodes, dtype=np.int64), shape)
    shifted = [(xi + s) % n for xi, s, n in zip(coords, reversed(tuple(c)), shape, strict=True)]
    return np.ravel_multi_index(shifted, shape)


@dataclass(frozen=True)
class WallLinks:
    """Links from fluid nodes into one body.

    Attributes:
        node: Flat index of the fluid node x, shape (n,).
        direction: Incoming direction j (the body node is x - c_j), shape (n,).
        q: Wall position along the link from x towards x - c_j, in (0, 1].
    """

    node: np.ndarray
    direction: np.ndarray
    q: np.ndarray

    def __len__(self) -> int:
        return len(self.node)


def wall_links(lattice: VelocitySet, body: Shape, solid: np.ndarray) -> WallLinks:
    """Every link from a fluid node (``~solid``) into ``body``.

    ``solid`` is the full solid mask of the simulation (body plus any other
    walls); only links whose far end lies inside ``body`` are returned, so
    channel walls and other bodies keep their own treatment. Distances use the
    unwrapped position x - c_j, so a body must not cross a periodic edge.
    """
    solid = np.asarray(solid, dtype=bool)
    shape = solid.shape
    if lattice.d != body.d or len(shape) != body.d:
        raise ValueError(f"{lattice.name}, a {body.d}D body and grid {shape} must agree")
    inside = voxelize(body, shape)
    if np.any(inside & ~solid):
        raise ValueError("every node inside the body must be marked solid")
    fluid_nodes = np.flatnonzero(~solid)
    positions = node_positions(shape).reshape(-1, body.d)
    nodes, directions, fractions = [], [], []
    for j, c_j in enumerate(lattice.c):
        if not any(c_j):
            continue
        source = neighbour_index(fluid_nodes, [-c for c in c_j], shape)
        x = fluid_nodes[inside.ravel()[source]]
        if x.size == 0:
            continue
        step = np.broadcast_to(-np.asarray(c_j, dtype=np.float64), (x.size, body.d))
        nodes.append(x)
        directions.append(np.full(x.size, j, dtype=np.int64))
        fractions.append(body.ray_fraction(positions[x], step))
    if not nodes:
        empty = np.zeros(0, dtype=np.int64)
        return WallLinks(empty, empty.copy(), np.zeros(0))
    return WallLinks(np.concatenate(nodes), np.concatenate(directions), np.concatenate(fractions))
