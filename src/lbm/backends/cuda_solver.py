"""CUDA backends: hand-written fused kernels, compiled at runtime and driven from Python.

How a kernel gets onto the GPU:

1. ``build_kernel_source`` concatenates
   the generated lattice preamble (``lbm.lattice.cuda_preamble``, sets ``real_t``),
   ``#define TILE_X / TILE_Y``, ``kernels/common.cuh`` (collision) and one
   variant file (``stream_collide_*.cu``).
2. ``cupy.RawModule`` hands that source to NVRTC, NVIDIA's runtime compiler,
   which produces machine code for the GPU in this machine. No ``nvcc`` and no
   system CUDA toolkit are needed. CuPy caches the binary on disk, so only the
   first run of a given source pays the compile time.
3. Kernels are launched on a grid of (bx, by, 1) blocks with grid
   (ceil(nx/bx), ceil(ny/by), nz); each thread owns one node. A 2D run is the
   same kernel with nz = 1 and a D2Q9 preamble, whose cz(i) = 0 lets the
   compiler drop every z term.

Variants: ``naive`` (two lattices, direct global loads), ``tiled`` (two
lattices, shared-memory staging, 2D only) and ``aa`` (one lattice, AA
pattern). All three call the same ``collide_node`` device function, so they
can only differ in memory access, which is what the benchmark measures.
"""

from __future__ import annotations

import functools
import math
from pathlib import Path

import numpy as np

from lbm.backends import import_cupy
from lbm.lattice import D2Q9, LATTICES, cuda_preamble
from lbm.link_rules import compile_rules
from lbm.params import SimulationParams
from lbm.solver import Solver

KERNEL_DIR = Path(__file__).parent / "kernels"
VARIANTS = ("naive", "tiled", "aa")
DEFAULT_BLOCK = (32, 8)
"""32 threads along x = one warp per row segment, so loads and stores coalesce."""

_SOURCES = {
    "naive": "stream_collide_naive.cu",
    "tiled": "stream_collide_tiled.cu",
    "aa": "stream_collide_aa.cu",
}
_C_TYPES = {"float32": "float", "float64": "double"}
_COMPILE_OPTIONS = ("--std=c++17",)
TWO_D_ONLY = ("tiled",)


def _check_variant(variant: str) -> None:
    if variant not in VARIANTS:
        raise ValueError(f"unknown CUDA variant {variant!r}; choose from {VARIANTS}")


def _check_dimension(variant: str, d: int) -> None:
    if variant in TWO_D_ONLY and d != 2:
        raise ValueError(
            f"CUDA variant {variant!r} is 2D only; use 'naive' or 'aa' for a {d}D grid"
        )


def build_kernel_source(
    variant: str,
    dtype: str,
    block: tuple[int, int] = DEFAULT_BLOCK,
    lattice: str = D2Q9.name,
) -> str:
    """Complete CUDA source for one kernel variant, precision and velocity set."""
    _check_variant(variant)
    if dtype not in _C_TYPES:
        raise ValueError(f"dtype must be one of {tuple(_C_TYPES)}; got {dtype!r}")
    if lattice not in LATTICES:
        raise ValueError(f"unknown lattice {lattice!r}; choose from {tuple(LATTICES)}")
    _check_dimension(variant, LATTICES[lattice].d)
    tile_x, tile_y = block
    return "\n".join(
        [
            cuda_preamble(LATTICES[lattice], _C_TYPES[dtype]),
            f"#define TILE_X {tile_x}",
            f"#define TILE_Y {tile_y}",
            (KERNEL_DIR / "common.cuh").read_text(),
            (KERNEL_DIR / _SOURCES[variant]).read_text(),
        ]
    )


@functools.cache
def compile_module(
    variant: str,
    dtype: str,
    block: tuple[int, int] = DEFAULT_BLOCK,
    lattice: str = D2Q9.name,
):
    """Compile (or fetch from CuPy's cache) the module for one variant."""
    cp = import_cupy()
    source = build_kernel_source(variant, dtype, block, lattice)
    return cp.RawModule(code=source, options=_COMPILE_OPTIONS)


RULES_BLOCK = 128


@functools.cache
def compile_rules_module(dtype: str):
    """The link-rule pre-pass kernels for one precision."""
    cp = import_cupy()
    source = f"typedef {_C_TYPES[dtype]} real_t;\n" + (KERNEL_DIR / "link_rules.cu").read_text()
    return cp.RawModule(code=source, options=_COMPILE_OPTIONS)


class _RulePass:
    """Device copy of compiled link rules for one layout, plus the launches.

    Rules differ a lot in length (Bouzidi: 2 terms, a D3Q19 Zou-He outlet
    population: ~20), so they are grouped by their number of terms and each
    group is stored trimmed to that length, term-major (slot[t][r]), so that
    neighbouring threads load neighbouring addresses. Every group evaluates
    into its own slice of one buffer; a single launch then writes all values.
    """

    def __init__(self, params: SimulationParams, layout: str, cp):
        compiled = compile_rules(
            params.lattice, params.link_rules, layout, params.solid, params.wall_velocity
        )
        values = compiled.values
        lengths = np.count_nonzero(values.coeff, axis=1)
        order = np.argsort(lengths, kind="stable")
        self._groups = []
        start = 0
        for length in np.unique(lengths):
            rows = order[lengths[order] == length]
            # Move each row's non-zero terms to the front, then keep `length` columns.
            nonzero_first = np.argsort(values.coeff[rows] == 0, axis=1, kind="stable")
            take = nonzero_first[:, : max(int(length), 1)]
            slots = np.take_along_axis(values.slots[rows], take, axis=1)
            coeff = np.take_along_axis(values.coeff[rows], take, axis=1)
            self._groups.append(
                (
                    start,
                    len(rows),
                    cp.asarray(np.ascontiguousarray(slots.T)),
                    cp.asarray(np.ascontiguousarray(coeff.T), dtype=params.dtype),
                    cp.asarray(values.constant[rows], dtype=params.dtype),
                )
            )
            start += len(rows)
        self._target = cp.asarray(compiled.target[order])
        self._buffer = cp.empty(len(order), dtype=params.dtype)
        module = compile_rules_module(params.dtype)
        self._evaluate = module.get_function("rules_evaluate")
        self._write = module.get_function("rules_write")
        self._n = np.int32(len(order))

    def __call__(self, populations) -> None:
        for start, count, slots, coeff, constant in self._groups:
            self._evaluate(
                (math.ceil(count / RULES_BLOCK),),
                (RULES_BLOCK,),
                (
                    populations,
                    slots,
                    coeff,
                    constant,
                    self._buffer[start:],
                    np.int32(count),
                    np.int32(slots.shape[0]),
                ),
            )
        self._write(
            (math.ceil(int(self._n) / RULES_BLOCK),),
            (RULES_BLOCK,),
            (populations, self._target, self._buffer, self._n),
        )


class CudaSolver(Solver):
    """Fused stream-bounce-collide on the GPU.

    Args:
        params: Simulation description.
        variant: ``"naive"``, ``"tiled"`` (2D only) or ``"aa"``.
        block: Threads per block (x, y). For ``tiled`` this is also the tile size.
    """

    def __init__(
        self,
        params: SimulationParams,
        variant: str = "naive",
        block: tuple[int, int] = DEFAULT_BLOCK,
    ):
        _check_variant(variant)
        _check_dimension(variant, params.d)
        bx, by = (int(b) for b in block)
        if bx <= 0 or by <= 0 or bx * by > 1024:
            raise ValueError(f"block must be positive with at most 1024 threads; got {block}")
        super().__init__(params)
        cp = import_cupy()
        self._cp = cp
        self.variant = variant
        self.step_multiple = 2 if variant == "aa" else 1

        module = compile_module(variant, params.dtype, (bx, by), params.lattice.name)
        real = np.dtype(params.dtype).type
        omega_plus, omega_minus = params.relaxation_rates
        nz = params.nz or 1
        fx, fy, fz = (*params.force, 0.0)[:3]
        # Scalars must carry exact C types: int for sizes, real_t for the rest.
        self._scalars = (
            np.int32(params.nx),
            np.int32(params.ny),
            np.int32(nz),
            real(omega_plus),
            real(omega_minus),
            real(fx),
            real(fy),
            real(fz),
        )
        self._opposite = list(params.lattice.opposite)
        self._block = (bx, by, 1)
        self._grid = (math.ceil(params.nx / bx), math.ceil(params.ny / by), nz)
        kinds = params.solid.astype(np.uint8)  # 0 fluid, 1 solid, 2 fluid with a local force
        self._ibm = self._field = None
        self._kernel_field = cp.zeros(1, dtype=params.dtype)  # placeholder pointer when unused
        if params.immersed is not None:
            from lbm.ibm import ImmersedForcing

            self._ibm = ImmersedForcing(params, cp)
            kinds.reshape(-1)[self._ibm.nodes_host] = 2
            self._field = cp.zeros((params.d, *params.shape), dtype=params.dtype)
            self._kernel_field = self._field
        self._solid = cp.asarray(kinds)
        self._wall = cp.ascontiguousarray(cp.asarray(params.wall_velocity, dtype=params.dtype))
        if variant == "aa":
            self._odd = module.get_function("aa_odd")
            self._even = module.get_function("aa_even")
        else:
            self._kernel = module.get_function(f"stream_collide_{variant}")
        self._rule_passes = {}
        if params.link_rules is not None and len(params.link_rules):
            layouts = ("aa_odd", "aa_even") if variant == "aa" else ("pull",)
            self._rule_passes = {layout: _RulePass(params, layout, cp) for layout in layouts}
        self.initialize()

    @property
    def xp(self):
        return self._cp

    @property
    def lattice_bytes(self) -> int:
        """Device memory held by population buffers."""
        buffers = 1 if self.variant == "aa" else 2
        p = self.params
        return buffers * p.lattice.q * math.prod(p.shape) * np.dtype(p.dtype).itemsize

    def initialize(self, rho=1.0, ux=0.0, uy=0.0, uz=0.0) -> None:
        feq = self._initial_populations(rho, ux, uy, uz)
        if self.variant == "aa":
            # Post-even-step layout: slot opp(i) holds f*_i.
            self._f = self._cp.ascontiguousarray(feq[self._opposite])
        else:
            self._f = self._cp.ascontiguousarray(feq)
            # Both buffers start fully defined; solid nodes are never overwritten.
            self._f_next = self._f.copy()
        self.time_step = 0

    def _advance(self, steps: int) -> None:
        grid, block, scalars = self._grid, self._block, self._scalars
        field = self._kernel_field
        if self.variant == "aa":
            for _ in range(steps // 2):
                self._pre_pass("aa_odd")
                self._odd(grid, block, (self._f, self._solid, self._wall, field, *scalars))
                self._pre_pass("aa_even")
                self._even(grid, block, (self._f, self._solid, field, *scalars))
            return
        for _ in range(steps):
            self._pre_pass("pull")
            self._kernel(
                grid, block, (self._f, self._f_next, self._solid, self._wall, field, *scalars)
            )
            self._f, self._f_next = self._f_next, self._f

    def _pre_pass(self, layout: str) -> None:
        """Sparse work before a kernel launch: link rules, then the IBM force field."""
        if self._rule_passes:
            self._rule_passes[layout](self._f)
        if self._ibm is not None:
            local = self._ibm.force_field(self._f.reshape(-1), layout)
            self._field.reshape(self.params.d, -1)[:, self._ibm.nodes] = local

    @property
    def state_layout(self) -> str:
        # Between run() calls the AA buffer is in its post-even-step layout.
        return "aa_odd" if self.variant == "aa" else "pull"

    def _state_buffer(self):
        return self._f

    def _device_populations(self):
        if self.variant == "aa":
            return self._f[self._opposite]
        return self._f

    def synchronize(self) -> None:
        """Block until all queued kernels have finished (for timing)."""
        self._cp.cuda.get_current_stream().synchronize()

    def kernel_resources(self) -> dict[str, dict[str, int]]:
        """Per-thread registers, local (spilled) memory and per-block shared memory.

        ``local_bytes`` > 0 would mean the compiler spilled per-thread arrays such
        as ``f[Q]`` to slow local memory; the constexpr lattice accessors exist to
        keep it at zero.
        """
        kernels = (self._odd, self._even) if self.variant == "aa" else (self._kernel,)
        return {
            k.name: {
                "registers": k.num_regs,
                "local_bytes": k.local_size_bytes,
                "shared_bytes": k.shared_size_bytes,
            }
            for k in kernels
        }
