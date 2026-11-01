"""Solver interface, run loop with divergence watchdog, and the reference solver.

Every backend stores post-collision populations f*(x, t) and advances one
time step as:

1. pull-stream        f_i(x) = f*_i(x - c_i)
2. bounce-back        f_i(x) = f*_opp(i)(x) + 6 w_i (c_i . u_w)   if x - c_i is solid
3. moments            rho = 1 + sum f~_i,  rho u = sum c_i f_i + F/2
4. collide            TRT + Guo, stored at fluid nodes only

The ``ReferenceSolver`` performs those four steps with whole-array operations
written against an array namespace ``xp`` and a velocity set, so one code
path serves D2Q9 and D3Q19. With ``xp = numpy`` it is the CPU reference used
for correctness checks and CI; with ``xp = cupy`` the same code runs on the
GPU, but as a chain of separate kernels, each reading and writing all
populations from global memory. The fused CUDA kernels in
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
from lbm.link_rules import CompiledAffine, compile_rules, evaluate_affine
from lbm.params import SimulationParams
from lbm.streaming import pull_stream

DEFAULT_CHECK_EVERY = 500
"""Steps between divergence checks in ``Solver.run``."""


class SimulationDiverged(RuntimeError):
    """The state became non-finite or the density non-positive.

    Attributes:
        step: Time step at which the problem was detected.
        location: Array index of the first bad node: ``(y, x)`` or ``(z, y, x)``.
    """

    def __init__(self, step: int, location: tuple[int, ...], reason: str):
        self.step = step
        self.location = location
        names = "(y, x)" if len(location) == 2 else "(z, y, x)"
        super().__init__(f"simulation diverged by step {step}: {reason} at node {names}={location}")


def to_host(array) -> np.ndarray:
    """Copy a NumPy or CuPy array into a new host NumPy array."""
    if hasattr(array, "get"):
        return array.get()
    return np.array(array, copy=True)


class Solver(ABC):
    """Common driver shared by every backend."""

    step_multiple = 1
    """Backends that advance in fixed groups of steps (AA pattern: 2) override this."""

    state_layout = "reference"
    """Where the stored post-collision state keeps f*_k(y) (see ``lbm.link_rules``)."""

    def __init__(self, params: SimulationParams):
        self.params = params
        self.time_step = 0

    @property
    @abstractmethod
    def xp(self):
        """Array namespace holding the solver state (numpy or cupy)."""

    @abstractmethod
    def initialize(self, rho=1.0, ux=0.0, uy=0.0, uz=0.0) -> None:
        """Set an equilibrium state from density and velocity (scalars or field-shaped arrays).

        ``uz`` must stay 0 in 2D.
        """

    @abstractmethod
    def _advance(self, steps: int) -> None:
        """Advance exactly ``steps`` time steps (a multiple of ``step_multiple``)."""

    @abstractmethod
    def _device_populations(self):
        """Post-collision populations in natural direction order, on the device."""

    def _velocity_components(self, ux, uy, uz) -> tuple:
        if self.params.d == 3:
            return (ux, uy, uz)
        if np.any(np.asarray(uz) != 0):
            raise ValueError("a 2D simulation has no z velocity; leave uz at 0")
        return (ux, uy)

    def _initial_populations(self, rho, ux, uy, uz):
        """Equilibrium populations whose reported velocity is the requested one.

        The stored state is post-collision, and post-collision moments report
        rho u = sum c_i f*_i - F/2. Building the equilibrium with u + F/(2 rho)
        makes the reported velocity at t = 0 equal to the requested one.
        Fields are combined in float64 and cast once, to keep float32 accurate.
        """
        p = self.params
        rho = np.broadcast_to(np.asarray(rho, dtype=np.float64), p.shape)
        u = [
            np.broadcast_to(np.asarray(u_a, dtype=np.float64), p.shape) + 0.5 * f_a / rho
            for u_a, f_a in zip(self._velocity_components(ux, uy, uz), p.force, strict=True)
        ]
        xp = self.xp
        delta_rho = xp.asarray(rho - 1.0, dtype=p.dtype)
        return equilibrium(p.lattice, delta_rho, tuple(xp.asarray(a, dtype=p.dtype) for a in u), xp)

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
        location = tuple(int(i) for i in np.unravel_index(int(xp.argmax(bad)), self.params.shape))
        reason = "non-finite populations" if bool(non_finite.any()) else "non-positive density"
        raise SimulationDiverged(self.time_step, location, reason)

    def _state_buffer(self):
        """The stored state buffer, in ``state_layout`` order."""
        return self._device_populations()

    def evaluate_affine(self, compiled: CompiledAffine) -> np.ndarray:
        """Host float64 values of compiled affine expressions on the current state."""
        xp = self.xp
        dtype = self.params.dtype
        on_device = CompiledAffine(
            slots=xp.asarray(compiled.slots),
            coeff=xp.asarray(compiled.coeff, dtype=dtype),
            constant=xp.asarray(compiled.constant, dtype=dtype),
        )
        values = evaluate_affine(on_device, self._state_buffer().reshape(-1), xp)
        return to_host(values).astype(np.float64)

    def populations(self) -> np.ndarray:
        """Host copy of post-collision shifted populations f~*, shape (q, *shape)."""
        return to_host(self._device_populations())

    def _collision_force(self) -> tuple:
        """Force density used by the last collision: uniform, plus the IBM field if any."""
        field = getattr(self, "_field", None)
        if field is None:
            return self.params.force
        return tuple(f_a + field[a] for a, f_a in enumerate(self.params.force))

    @property
    def immersed_force(self) -> np.ndarray:
        """Force on the immersed body during the last time step (host, float64)."""
        forcing = getattr(self, "_ibm", None)
        if forcing is None:
            raise AttributeError("this simulation has no immersed body")
        return to_host(forcing.body_force).astype(np.float64)

    def macroscopic(self) -> MacroscopicFields:
        """Host snapshot of rho and u at the current time step."""
        p = self.params
        rho, u = macroscopic_from_post_collision(
            p.lattice,
            self._device_populations(),
            self._collision_force(),
            self.xp.asarray(p.solid),
            self.xp,
        )
        return MacroscopicFields(
            rho=to_host(rho), u=tuple(to_host(u_a) for u_a in u), solid=np.array(p.solid)
        )


class ReferenceSolver(Solver):
    """Readable whole-array implementation; runs on NumPy (CPU) or CuPy (GPU), 2D or 3D."""

    def __init__(self, params: SimulationParams, xp=np):
        super().__init__(params)
        self._xp = xp
        self._fluid = xp.asarray(params.fluid)
        self._links = bounce_back_links(
            params.lattice, params.solid, params.wall_velocity, xp, dtype=np.dtype(params.dtype)
        )
        self._rates = params.relaxation_rates
        self._rules = None
        if params.link_rules is not None and len(params.link_rules):
            compiled = compile_rules(
                params.lattice, params.link_rules, "reference", params.solid, params.wall_velocity
            )
            self._rules = _to_device(compiled, xp, params.dtype)
        self._ibm = self._field = None
        if params.immersed is not None:
            from lbm.ibm import ImmersedForcing

            self._ibm = ImmersedForcing(params, xp)
            self._field = xp.zeros((params.d, *params.shape), dtype=params.dtype)
        self.initialize()

    @property
    def xp(self):
        return self._xp

    def initialize(self, rho=1.0, ux=0.0, uy=0.0, uz=0.0) -> None:
        self._f = self._initial_populations(rho, ux, uy, uz)
        self.time_step = 0

    def _advance(self, steps: int) -> None:
        xp = self._xp
        lattice = self.params.lattice
        omega_plus, omega_minus = self._rates
        for _ in range(steps):
            force = self.params.force
            if self._ibm is not None:
                local = self._ibm.force_field(self._f.reshape(-1), "reference")
                self._field.reshape(lattice.d, -1)[:, self._ibm.nodes] = local
                force = self._collision_force()
            streamed = pull_stream(lattice, self._f, xp)
            streamed = apply_bounce_back(lattice, streamed, self._f, self._links, xp)
            if self._rules is not None:
                values = evaluate_affine(self._rules.values, self._f.reshape(-1), xp)
                streamed.reshape(-1)[self._rules.target] = values
            post = collide(lattice, streamed, omega_plus, omega_minus, force, xp)
            # Solid nodes are never read except through bounce-back; keep their old values.
            self._f = xp.where(self._fluid, post, self._f)

    def _device_populations(self):
        return self._f


def _to_device(compiled, xp, dtype):
    """Copy a ``CompiledRules`` to the array namespace ``xp`` (coefficients in ``dtype``)."""
    from lbm.link_rules import CompiledAffine, CompiledRules

    values = compiled.values
    return CompiledRules(
        target=xp.asarray(compiled.target),
        values=CompiledAffine(
            slots=xp.asarray(values.slots),
            coeff=xp.asarray(values.coeff, dtype=dtype),
            constant=xp.asarray(values.constant, dtype=dtype),
        ),
    )
