"""Solver interface, run loop with divergence watchdog, and the reference solver.

Every backend stores post-collision populations f*(x, t) and advances one
time step as:

1. pull-stream        f_i(x) = f*_i(x - c_i)
2. bounce-back        f_i(x) = f*_opp(i)(x) + 6 w_i (c_i . u_w)   if x - c_i is solid
3. moments            rho = 1 + sum f~_i,  rho u = sum c_i f_i + F/2
4. collide            TRT + Guo, stored at fluid nodes only

The ``ReferenceSolver`` performs those four steps with whole-array operations
written against an array namespace ``xp``. With ``xp = numpy`` it is the
CPU reference used for correctness checks and CI; with ``xp = cupy`` the same
code runs on the GPU, but as a chain of separate kernels, each reading and
writing all populations from global memory. The fused CUDA kernels in
``lbm.backends.cuda_solver`` do the four steps in one memory round trip.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np

from lbm.boundaries import apply_bounce_back, bounce_back_links
from lbm.collision import collide
from lbm.diagnostics import MacroscopicFields, macroscopic_from_post_collision
from lbm.equilibrium import equilibrium
from lbm.params import SimulationParams
from lbm.streaming import pull_stream

DEFAULT_CHECK_EVERY = 500
"""Steps between divergence checks in ``Solver.run``."""


class SimulationDiverged(RuntimeError):
    """The state became non-finite or the density non-positive.

    Attributes:
        step: Time step at which the problem was detected.
        location: ``(y, x)`` of the first bad node found.
    """

    def __init__(self, step: int, location: tuple[int, int], reason: str):
        self.step = step
        self.location = location
        super().__init__(f"simulation diverged by step {step}: {reason} at node (y, x)={location}")


def to_host(array) -> np.ndarray:
    """Copy a NumPy or CuPy array into a new host NumPy array."""
    if hasattr(array, "get"):
        return array.get()
    return np.array(array, copy=True)


class Solver(ABC):
    """Common driver shared by every backend."""

    step_multiple = 1
    """Backends that advance in fixed groups of steps (AA pattern: 2) override this."""

    def __init__(self, params: SimulationParams):
        self.params = params
        self.time_step = 0

    @property
    @abstractmethod
    def xp(self):
        """Array namespace holding the solver state (numpy or cupy)."""

    @abstractmethod
    def initialize(self, rho=1.0, ux=0.0, uy=0.0) -> None:
        """Set an equilibrium state from density and velocity (scalars or (ny, nx) arrays)."""

    @abstractmethod
    def _advance(self, steps: int) -> None:
        """Advance exactly ``steps`` time steps (a multiple of ``step_multiple``)."""

    @abstractmethod
    def _device_populations(self):
        """Post-collision populations in natural direction order, on the device."""

    def _initial_populations(self, rho, ux, uy):
        """Equilibrium populations whose reported velocity is (ux, uy).

        The stored state is post-collision, and post-collision moments report
        rho u = sum c_i f*_i - F/2. Building the equilibrium with u + F/(2 rho)
        makes the reported velocity at t = 0 equal to the requested one.
        Fields are combined in float64 and cast once, to keep float32 accurate.
        """
        p = self.params
        shape = (p.ny, p.nx)
        rho, ux, uy = (
            np.broadcast_to(np.asarray(a, dtype=np.float64), shape) for a in (rho, ux, uy)
        )
        fx, fy = p.force
        fields = (rho - 1.0, ux + 0.5 * fx / rho, uy + 0.5 * fy / rho)
        drho, ux, uy = (self.xp.asarray(a, dtype=p.dtype) for a in fields)
        return equilibrium(drho, ux, uy, self.xp)

    def run(
        self,
        steps: int,
        *,
        every: int = 0,
        callback: Callable[[Solver], None] | None = None,
        check_every: int = DEFAULT_CHECK_EVERY,
    ) -> None:
        """Advance ``steps`` time steps.

        Args:
            steps: Number of time steps.
            every: If > 0, call ``callback(self)`` every ``every`` steps.
            callback: Called every ``every`` steps and after the last step.
            check_every: Steps between divergence checks.

        Raises:
            SimulationDiverged: Non-finite populations or rho <= 0 at a fluid node.
            ValueError: Negative counts, or counts that are not a multiple of
                ``step_multiple``.
        """
        if steps < 0 or every < 0 or check_every <= 0:
            raise ValueError(
                f"need steps >= 0, every >= 0, check_every > 0; got {steps}, {every}, {check_every}"
            )
        m = self.step_multiple
        if steps % m or every % m or check_every % m:
            raise ValueError(
                f"{type(self).__name__} advances in multiples of {m} steps; "
                f"steps, every and check_every must be multiples of {m}"
            )
        done = 0
        while done < steps:
            stop = min(steps, (done // check_every + 1) * check_every)
            if every:
                stop = min(stop, (done // every + 1) * every)
            self._advance(stop - done)
            self.time_step += stop - done
            done = stop
            self._check_health()
            if callback is not None and ((every and done % every == 0) or done == steps):
                callback(self)

    def _check_health(self) -> None:
        xp = self.xp
        delta_rho = self._device_populations().sum(axis=0)
        fluid = xp.asarray(self.params.fluid)
        non_finite = ~xp.isfinite(delta_rho) & fluid
        bad = non_finite | ((delta_rho <= -1.0) & fluid)
        if not bool(bad.any()):
            return
        y, x = divmod(int(xp.argmax(bad)), self.params.nx)
        reason = "non-finite populations" if bool(non_finite.any()) else "non-positive density"
        raise SimulationDiverged(self.time_step, (y, x), reason)

    def populations(self) -> np.ndarray:
        """Host copy of post-collision shifted populations f~*, shape (Q, ny, nx)."""
        return to_host(self._device_populations())

    def macroscopic(self) -> MacroscopicFields:
        """Host snapshot of rho, ux, uy at the current time step."""
        p = self.params
        rho, ux, uy = macroscopic_from_post_collision(
            self._device_populations(), p.force, self.xp.asarray(p.solid), self.xp
        )
        return MacroscopicFields(
            rho=to_host(rho), ux=to_host(ux), uy=to_host(uy), solid=np.array(p.solid)
        )


class ReferenceSolver(Solver):
    """Readable whole-array implementation; runs on NumPy (CPU) or CuPy (GPU)."""

    def __init__(self, params: SimulationParams, xp=np):
        super().__init__(params)
        self._xp = xp
        self._fluid = xp.asarray(params.fluid)
        self._links = bounce_back_links(
            params.solid, params.wall_velocity, xp, dtype=np.dtype(params.dtype)
        )
        self._rates = params.relaxation_rates
        self.initialize()

    @property
    def xp(self):
        return self._xp

    def initialize(self, rho=1.0, ux=0.0, uy=0.0) -> None:
        self._f = self._initial_populations(rho, ux, uy)
        self.time_step = 0

    def _advance(self, steps: int) -> None:
        xp = self._xp
        omega_plus, omega_minus = self._rates
        for _ in range(steps):
            streamed = apply_bounce_back(pull_stream(self._f, xp), self._f, self._links, xp)
            post = collide(streamed, omega_plus, omega_minus, self.params.force, xp)
            # Solid nodes are never read except through bounce-back; keep their old values.
            self._f = xp.where(self._fluid, post, self._f)

    def _device_populations(self):
        return self._f
