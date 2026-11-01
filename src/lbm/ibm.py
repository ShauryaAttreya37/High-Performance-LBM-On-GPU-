"""Immersed boundary method: direct forcing with multi-direct iterations.

The body is not part of the grid at all. It is a set of Lagrangian markers
X_l on its surface, each carrying a patch of surface dA_l (arc length in
2D). Every time step, before the collision, a force field is added to the
fluid near the markers so that the fluid velocity interpolated at each marker
equals the body velocity U_b (zero for a body at rest):

1. Predict: u*(x) from the streamed populations (moments before forcing).
2. Interpolate: U*_l = sum_x u*(x) delta(x - X_l).
3. Marker force: F_l = 2 rho_l (U_b - U*_l). With Guo forcing the velocity
   seen by the collision is u = u* + F / (2 rho), so a force density F at a
   node shifts its velocity by F / (2 rho).
4. Spread: f(x) = sum_l F_l delta(x - X_l) dA_l.

One pass leaves a residual slip because the spread forces of neighbouring
markers overlap; "multi-direct forcing" (Wang, Fan & Luo 2008) repeats steps
2-4 a few times, each time with the velocity already corrected by the force
so far. The force on the body is minus the total force on the fluid,
-sum_l F_l dA_l (no internal-mass term: the body does not move).

delta is the 3-point kernel of Roma, Peskin & Berger (1999), a product of
1D weights phi(r) with support |r| <= 1.5 nodes:

    phi(r) = (1 + sqrt(1 - 3 r^2)) / 3                     |r| <= 1/2
    phi(r) = (5 - 3|r| - sqrt(1 - 3 (1 - |r|)^2)) / 6      1/2 <= |r| <= 3/2

Compared with interpolated bounce-back the wall is diffuse: it is smeared
over ~3 nodes, the no-slip condition is met only approximately, and accuracy
is first order. In exchange it needs no link geometry, works for any shape
given surface points, and extends naturally to moving bodies.

The predicted moments in step 1 are affine in the stored post-collision
populations (f_k(x) = f*_k(x - c_k)), so they are compiled with the same
layout machinery as link rules and read straight from any solver's state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from lbm.geometry import Ball, neighbour_index
from lbm.lattice import VelocitySet
from lbm.link_rules import AffineSet, compile_affine, evaluate_affine

STENCIL = 3
"""Nodes per axis covered by the Roma kernel."""


def roma(r: np.ndarray) -> np.ndarray:
    """1D weight of the 3-point Roma-Peskin-Berger kernel."""
    a = np.abs(np.asarray(r, dtype=np.float64))
    inner = (1.0 + np.sqrt(np.maximum(1.0 - 3.0 * a * a, 0.0))) / 3.0
    outer = (5.0 - 3.0 * a - np.sqrt(np.maximum(1.0 - 3.0 * (1.0 - a) ** 2, 0.0))) / 6.0
    return np.where(a <= 0.5, inner, np.where(a <= 1.5, outer, 0.0))


@dataclass(frozen=True, eq=False)
class ImmersedBoundary:
    """Surface markers of one body at rest.

    Attributes:
        markers: Positions, shape (n, d), ordered x, y[, z], in node units.
        areas: Surface element of each marker (arc length in 2D), shape (n,).
        iterations: Multi-direct forcing passes per time step.
    """

    markers: np.ndarray
    areas: np.ndarray
    iterations: int = 3

    def __post_init__(self) -> None:
        if self.markers.ndim != 2 or self.markers.shape[1] not in (2, 3):
            raise ValueError("markers must have shape (n, 2) or (n, 3)")
        if self.areas.shape != (len(self.markers),):
            raise ValueError("areas must have one entry per marker")
        if self.iterations < 1:
            raise ValueError("iterations must be >= 1")

    @property
    def d(self) -> int:
        return self.markers.shape[1]

    @staticmethod
    def from_ball(ball: Ball, spacing: float = 0.8, iterations: int = 3) -> ImmersedBoundary:
        """Markers on a circle (evenly spaced) or sphere (Fibonacci lattice)."""
        r = ball.radius
        centre = np.asarray(ball.center)
        if ball.d == 2:
            n = max(8, math.ceil(2.0 * math.pi * r / spacing))
            angle = 2.0 * math.pi * (np.arange(n) + 0.5) / n
            points = np.stack([np.cos(angle), np.sin(angle)], axis=1)
            areas = np.full(n, 2.0 * math.pi * r / n)
        else:
            n = max(32, math.ceil(4.0 * math.pi * r * r / spacing**2))
            k = np.arange(n) + 0.5
            z = 1.0 - 2.0 * k / n
            phi = math.pi * (1.0 + math.sqrt(5.0)) * k
            rho = np.sqrt(1.0 - z * z)
            points = np.stack([rho * np.cos(phi), rho * np.sin(phi), z], axis=1)
            areas = np.full(n, 4.0 * math.pi * r * r / n)
        return ImmersedBoundary(centre + r * points, areas, iterations)

    def stencil(self, shape: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
        """Flat node indices and kernel weights, each of shape (n, 3^d)."""
        d = self.d
        base = np.floor(self.markers - 0.5).astype(np.int64) + 0  # lowest node of the stencil
        offsets = np.stack(np.meshgrid(*(np.arange(STENCIL),) * d, indexing="ij"), axis=-1).reshape(
            -1, d
        )
        nodes = base[:, None, :] + offsets[None, :, :]  # (n, 27 or 9, d), x y z order
        weights = np.prod(roma(nodes - self.markers[:, None, :]), axis=-1)
        sizes = shape[::-1]  # x, y, z sizes
        if np.any(nodes < 0) or np.any(nodes >= np.asarray(sizes)):
            raise ValueError("every marker must sit at least 2 nodes inside the domain")
        flat = np.ravel_multi_index(tuple(nodes[..., a] for a in range(d))[::-1], shape)
        return flat, weights


class ImmersedForcing:
    """Device-side state of one immersed body for one solver (any array namespace).

    ``force_field(state, layout)`` returns the Eulerian force at ``self.nodes``
    for the coming collision and remembers the force on the body.
    """

    def __init__(self, params, xp=np):
        body: ImmersedBoundary = params.immersed
        lattice: VelocitySet = params.lattice
        shape = params.shape
        stencil, weights = body.stencil(shape)
        nodes, inverse = np.unique(stencil, return_inverse=True)
        self._check_clear_of_walls(lattice, params.solid, nodes, shape)
        self.nodes_host = nodes
        self.xp = xp
        self.d = lattice.d
        self.iterations = body.iterations
        self.uniform_force = tuple(float(f) for f in params.force)
        self.nodes = xp.asarray(nodes)
        self.inverse = xp.asarray(inverse.reshape(stencil.shape))
        dtype = params.dtype
        self.weights = xp.asarray(weights, dtype=dtype)
        self.areas = xp.asarray(body.areas, dtype=dtype)
        self.n_nodes = len(nodes)
        self.body_force = xp.zeros(self.d, dtype=np.float64)
        self._params = params
        self._moments = self._moment_expressions(lattice, nodes, shape)
        self._compiled = {}

    @staticmethod
    def _check_clear_of_walls(lattice, solid, nodes, shape) -> None:
        flat = np.asarray(solid, dtype=bool).ravel()
        for c in lattice.c:
            if np.any(flat[neighbour_index(nodes, c, shape)]):
                raise ValueError("immersed-boundary stencils must not touch solid nodes")

    @staticmethod
    def _moment_expressions(lattice, nodes, shape) -> AffineSet:
        """rho - 1 and rho u (no force) of the streamed populations, (d + 1) rows per node."""
        sources = [neighbour_index(nodes, [-c for c in c_k], shape) for c_k in lattice.c]
        rows = []
        for component in range(lattice.d + 1):
            for n in range(len(nodes)):
                row = []
                for k, c_k in enumerate(lattice.c):
                    a = 1.0 if component == 0 else float(c_k[component - 1])
                    if a:
                        row.append((k, int(sources[k][n]), a))
                rows.append(row)
        return AffineSet.from_lists(rows)

    def _expressions(self, layout: str):
        if layout not in self._compiled:
            p = self._params
            compiled = compile_affine(p.lattice, self._moments, layout, p.solid, p.wall_velocity)
            xp = self.xp
            from lbm.link_rules import CompiledAffine

            self._compiled[layout] = CompiledAffine(
                slots=xp.asarray(compiled.slots),
                coeff=xp.asarray(compiled.coeff, dtype=p.dtype),
                constant=xp.asarray(compiled.constant, dtype=p.dtype),
            )
        return self._compiled[layout]

    def force_field(self, state_flat, layout: str):
        """Force density at ``self.nodes``, shape (d, n_nodes), for the next collision."""
        xp = self.xp
        moments = evaluate_affine(self._expressions(layout), state_flat, xp).reshape(
            self.d + 1, self.n_nodes
        )
        rho = 1.0 + moments[0]
        u_star = xp.stack(
            [(moments[1 + a] + 0.5 * self.uniform_force[a]) / rho for a in range(self.d)]
        )
        rho_marker = (self.weights * rho[self.inverse]).sum(axis=1)
        force = xp.zeros((self.d, self.n_nodes), dtype=u_star.dtype)
        total = xp.zeros(self.d, dtype=u_star.dtype)
        flat_inverse = self.inverse.reshape(-1)
        for _ in range(self.iterations):
            u = u_star + force / (2.0 * rho)
            u_marker = (self.weights[None] * u[:, self.inverse]).sum(axis=2)  # (d, n_markers)
            f_marker = 2.0 * rho_marker * (0.0 - u_marker) * self.areas
            total = total + f_marker.sum(axis=1)
            spread = (self.weights[None] * f_marker[:, :, None]).reshape(self.d, -1)
            for a in range(self.d):
                force[a] = force[a] + xp.bincount(
                    flat_inverse, weights=spread[a], minlength=self.n_nodes
                )
        self.body_force = -total
        return force
