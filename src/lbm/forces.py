"""Force on a body by momentum exchange (Ladd 1994; Mei et al. 2002).

Every time step, each population that crosses the body surface along a wall
link and comes back hands its momentum change to the body. For a link from
fluid node x into the body, i the direction pointing into the body and
j = opp(i) the returning one::

    F_body = sum_links c_i [ f*_i(x, t) + f_j(x, t + 1) ]

with full (unshifted) populations. The outgoing f*_i is a stored population;
the returning f_j is what the boundary condition makes of it: plain
bounce-back gives f*_i(x) back (+ a moving-wall term), interpolated
bounce-back gives its ``LinkRules`` combination. Either way f_j(t + 1) is an
affine combination of post-collision populations at time t, so the whole
force is a fixed linear functional of the stored state. ``ForceProbe``
builds that functional once; a solver evaluates it on the device in its own
memory layout, so a force costs one sparse gather, not a time step.

The shifted-storage constant: f = f~ + w and w_i = w_j, so each link adds
2 w_i c_i. Over a closed body these cancel exactly (every lattice line that
enters the body also leaves it), but they are kept so that open link sets
(e.g. one wall of a channel) give the right answer too.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from lbm.geometry import WallLinks, neighbour_index
from lbm.link_rules import AffineSet, compile_affine
from lbm.params import SimulationParams


@dataclass(frozen=True, eq=False)
class ForceProbe:
    """Momentum-exchange force on one body, as sum_r weight[r] * value[r]."""

    params: SimulationParams
    weight: np.ndarray
    """Shape (n_links, d): c_i of each link (pointing into the body)."""
    values: AffineSet
    """f~*_i(x) + f~_j(x, t + 1) + 2 w_i for each link."""
    _compiled: dict = field(default_factory=dict, repr=False)

    @staticmethod
    def build(params: SimulationParams, links: WallLinks) -> ForceProbe:
        """Probe for the links of one body; uses the params' link rules where they exist."""
        lattice = params.lattice
        shape = params.shape
        opposite = np.asarray(lattice.opposite)
        rules = params.link_rules
        rule_rows = {}
        if rules is not None:
            for r, key in enumerate(
                zip(rules.node.tolist(), rules.direction.tolist(), strict=True)
            ):
                rule_rows[key] = r
        walls = params.wall_velocity.reshape(lattice.d, -1)
        rows, constants = [], []
        for x, j in zip(links.node.tolist(), links.direction.tolist(), strict=True):
            i = int(opposite[j])
            row = [(i, x, 1.0)]
            constant = 2.0 * lattice.w[i]
            r = rule_rows.get((x, j))
            if r is None:
                # Plain bounce-back: f_j(t + 1) = f*_i(x) + 2 w_j (c_j . u_w) / cs2.
                source = int(neighbour_index([x], [-c for c in lattice.c[j]], shape)[0])
                c_dot_uw = float(np.dot(lattice.c[j], walls[:, source]))
                row.append((i, x, 1.0))
                constant += 2.0 * lattice.w[j] / lattice.cs2 * c_dot_uw
            else:
                v = rules.values
                row += [
                    (int(k), int(y), float(a))
                    for k, y, a in zip(v.direction[r], v.node[r], v.coeff[r], strict=True)
                    if a != 0.0
                ]
                constant += float(v.constant[r])
            rows.append(row)
            constants.append(constant)
        weight = -np.asarray([lattice.c[j] for j in links.direction], dtype=np.float64)
        return ForceProbe(
            params=params,
            weight=weight.reshape(-1, lattice.d),
            values=AffineSet.from_lists(rows, constants),
        )

    def compiled(self, layout: str):
        if layout not in self._compiled:
            p = self.params
            self._compiled[layout] = compile_affine(
                p.lattice, self.values, layout, p.solid, p.wall_velocity
            )
        return self._compiled[layout]

    def evaluate(self, solver) -> np.ndarray:
        """Force on the body from the solver's current state, shape (d,), host float64."""
        if solver.params is not self.params:
            raise ValueError("this probe was built for a different SimulationParams")
        values = solver.evaluate_affine(self.compiled(solver.state_layout))
        return self.weight.T @ values
