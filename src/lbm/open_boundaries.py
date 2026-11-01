"""Open boundaries for flow along +x: Zou-He velocity inlet and pressure outlet.

Layout: the x = 0 layer and the x = nx - 1 layer are solid nodes at rest; the
populations that would stream out of them into the first / last fluid column
are set by link rules (``lbm.link_rules``). The kernels need nothing new.

Zou & He (1997), in the general form of Hecht & Harting (2010) that covers
D2Q9 and D3Q19 alike. At a boundary node with inward unit normal n, the
populations with c.n = +1 are unknown, the rest have streamed in. Mass and
normal momentum give::

    rho (1 - u_n) = S_0 + 2 S_-      S_0 = sum over c.n = 0,  S_- = sum over c.n = -1

so prescribing u (inlet) fixes rho, and prescribing rho (outlet, with zero
tangential velocity) fixes u_n. Then every unknown population is the
bounce-back of its opposite plus the equilibrium difference, plus a
transverse correction that makes the tangential momentum come out right::

    f_j = f_opp(j) + 6 w_j rho (c_j . u) - sum_t (c_j . t) N_t
    N_t = 1/2 sum_{c.n = 0} f_k (c_k . t) - 1/3 rho u_t        (t: tangential axes)

The node then has exactly the prescribed velocity (inlet) or density
(outlet). Once u or rho is fixed, every right-hand side above is linear in the
streamed populations, and those are post-collision populations of the
previous step, so each Zou-He population is an affine link rule.

Corner nodes (next to both the open layer and a channel wall) get the full
Zou-He treatment too: every unknown population is set by the rule, also the
one that streams in from the wall corner. Both alternatives were tried and
are unstable in long runs (2D channel, u_max = 0.05, tau = 0.65):

* Zou-He only for the populations coming from the open layer, bounce-back
  for the one from the wall: density and momentum at the node no longer match
  the prescribed values, and mass grows until the run blows up (~150k steps).
* plain bounce-back on the whole corner node: mass still grows, by a factor
  ~2 every 40k steps.

With full Zou-He at the corners the same channel reaches a steady state
whose mass changes by 6e-12 per 40k steps.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from lbm.geometry import neighbour_index
from lbm.lattice import VelocitySet
from lbm.link_rules import AffineSet, LinkRules


class _Expr:
    """Affine expression in *full* populations f*_k(y): const + sum coeff * f*."""

    def __init__(self, terms=None, const: float = 0.0):
        self.terms: dict[tuple[int, int], float] = defaultdict(float, terms or {})
        self.const = const

    def __add__(self, other: _Expr) -> _Expr:
        out = _Expr(dict(self.terms), self.const + other.const)
        for key, a in other.terms.items():
            out.terms[key] += a
        return out

    def scaled(self, factor: float) -> _Expr:
        return _Expr({k: a * factor for k, a in self.terms.items()}, self.const * factor)

    def shifted_row(self, lattice: VelocitySet, target_direction: int):
        """(terms, constant) of the same rule written for shifted storage f~ = f - w."""
        const = self.const - lattice.w[target_direction]
        row = []
        for (k, y), a in self.terms.items():
            if a != 0.0:
                row.append((k, y, a))
                const += a * lattice.w[k]
        return row, const


def _streamed(lattice: VelocitySet, k: int, x: int, solid_flat, shape) -> _Expr:
    """Full streamed population f_k(x, t+1) at a node next to resting walls."""
    source = int(neighbour_index([x], [-c for c in lattice.c[k]], shape)[0])
    if solid_flat[source]:
        return _Expr({(int(lattice.opposite[k]), x): 1.0})
    return _Expr({(k, source): 1.0})


def _boundary_column(solid: np.ndarray, column: int) -> np.ndarray:
    """Flat indices of the fluid nodes in x column ``column``."""
    shape = solid.shape
    cross = np.flatnonzero(~solid[..., column].ravel())
    coords = np.unravel_index(cross, solid[..., column].shape)
    return np.ravel_multi_index((*coords, np.full(cross.size, column)), shape)


def _zou_he(lattice, solid, layer_mask, column, normal_sign, velocity=None, rho_out=None):
    """Zou-He rules for the fluid column next to ``layer_mask`` (inlet or outlet layer)."""
    shape = solid.shape
    solid_flat = solid.ravel()
    layer_flat = layer_mask.ravel()
    c = np.asarray(lattice.c)
    c_n = normal_sign * c[:, 0]
    tangential = range(1, lattice.d)
    nodes, directions, rows, constants = [], [], [], []
    for x in _boundary_column(solid, column).tolist():
        streamed = {k: _streamed(lattice, k, x, solid_flat, shape) for k in range(lattice.q)}
        s0 = _Expr()
        for k in np.flatnonzero(c_n == 0):
            s0 = s0 + streamed[k]
        s_minus = _Expr()
        for k in np.flatnonzero(c_n == -1):
            s_minus = s_minus + streamed[k]
        known = s0 + s_minus.scaled(2.0)
        if velocity is not None:
            u = velocity[(slice(None), *np.unravel_index(x, shape))]
            u_n = normal_sign * u[0]
            rho = known.scaled(1.0 / (1.0 - u_n))
            rho_u = [rho.scaled(float(u_a)) for u_a in u]
        else:
            rho = _Expr(const=rho_out)
            rho_u_n = rho + known.scaled(-1.0)
            rho_u = [rho_u_n.scaled(float(normal_sign))] + [_Expr() for _ in tangential]
        transverse = {}
        for t in tangential:
            half_sum = _Expr()
            for k in np.flatnonzero(c_n == 0):
                if c[k, t]:
                    half_sum = half_sum + streamed[k].scaled(0.5 * c[k, t])
            transverse[t] = half_sum + rho_u[t].scaled(-1.0 / 3.0)
        unknown = np.flatnonzero(c_n == 1)
        sources = [
            int(neighbour_index([x], [-cc for cc in lattice.c[j]], shape)[0]) for j in unknown
        ]
        if not any(layer_flat[s] for s in sources):
            continue
        for j in unknown:
            value = streamed[int(lattice.opposite[j])]
            for a in range(lattice.d):
                if c[j, a]:
                    value = value + rho_u[a].scaled(6.0 * lattice.w[j] * c[j, a])
            for t in tangential:
                if c[j, t]:
                    value = value + transverse[t].scaled(-float(c[j, t]))
            row, const = value.shifted_row(lattice, int(j))
            nodes.append(x)
            directions.append(int(j))
            rows.append(row)
            constants.append(const)
    return LinkRules(
        np.asarray(nodes, dtype=np.int64),
        np.asarray(directions, dtype=np.int64),
        AffineSet.from_lists(rows, constants),
    )


def open_channel(lattice: VelocitySet, solid: np.ndarray, inflow, rho_out: float = 1.0):
    """Close the x = 0 and x = nx - 1 layers and return Zou-He inlet + outlet rules.

    Args:
        lattice: Velocity set.
        solid: Solid mask (walls, bodies) of shape (ny, nx) or (nz, ny, nx).
            The first and last x layers are overwritten.
        inflow: Inlet velocity, broadcastable to (d, *cross_section), applied
            at the first fluid column x = 1.
        rho_out: Density imposed at the last fluid column x = nx - 2.

    Returns:
        ``(solid, rules)``: the new solid mask and the ``LinkRules``.
    """
    solid = np.array(solid, dtype=bool)
    if solid.shape[-1] < 4:
        raise ValueError("an open channel needs at least 4 nodes along x")
    inlet = np.zeros_like(solid)
    outlet = np.zeros_like(solid)
    inlet[..., 0] = ~solid[..., 0]
    outlet[..., -1] = ~solid[..., -1]
    solid[..., 0] = True
    solid[..., -1] = True
    shape = solid.shape
    velocity = np.zeros((lattice.d, *shape))
    velocity[..., 1] = np.broadcast_to(np.asarray(inflow, dtype=np.float64), velocity[..., 1].shape)
    rules = LinkRules.concat(
        _zou_he(lattice, solid, inlet, 1, +1, velocity=velocity),
        _zou_he(lattice, solid, outlet, shape[-1] - 2, -1, rho_out=rho_out),
    )
    return solid, rules
