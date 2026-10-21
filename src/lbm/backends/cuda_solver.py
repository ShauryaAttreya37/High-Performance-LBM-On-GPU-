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
3. Kernels are launched on a 2D grid of 2D blocks; each thread owns one node.

Variants: ``naive`` (two lattices, direct global loads), ``tiled`` (two
lattices, shared-memory staging) and ``aa`` (one lattice, AA pattern). All
three call the same ``collide_node`` device function, so they can only differ
in memory access, which is what the benchmark measures.
"""

from __future__ import annotations

import functools
import math
from pathlib import Path

import numpy as np

from lbm.backends import import_cupy
from lbm.lattice import OPPOSITE_TUPLE, Q, cuda_preamble
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
_OPPOSITE = list(OPPOSITE_TUPLE)


def _check_variant(variant: str) -> None:
    if variant not in VARIANTS:
        raise ValueError(f"unknown CUDA variant {variant!r}; choose from {VARIANTS}")


def build_kernel_source(variant: str, dtype: str, block: tuple[int, int] = DEFAULT_BLOCK) -> str:
    """Complete CUDA source for one kernel variant and precision."""
    _check_variant(variant)
    if dtype not in _C_TYPES:
        raise ValueError(f"dtype must be one of {tuple(_C_TYPES)}; got {dtype!r}")
    tile_x, tile_y = block
    return "\n".join(
        [
            cuda_preamble(_C_TYPES[dtype]),
            f"#define TILE_X {tile_x}",
            f"#define TILE_Y {tile_y}",
            (KERNEL_DIR / "common.cuh").read_text(),
            (KERNEL_DIR / _SOURCES[variant]).read_text(),
        ]
    )


@functools.cache
def compile_module(variant: str, dtype: str, block: tuple[int, int] = DEFAULT_BLOCK):
    """Compile (or fetch from CuPy's cache) the module for one variant."""
    cp = import_cupy()
    return cp.RawModule(code=build_kernel_source(variant, dtype, block), options=_COMPILE_OPTIONS)


class CudaSolver(Solver):
    """Fused stream-bounce-collide on the GPU.

    Args:
        params: Simulation description.
        variant: ``"naive"``, ``"tiled"`` or ``"aa"``.
        block: Threads per block (x, y). For ``tiled`` this is also the tile size.
    """

    def __init__(
        self,
        params: SimulationParams,
        variant: str = "naive",
        block: tuple[int, int] = DEFAULT_BLOCK,
    ):
        _check_variant(variant)
        bx, by = (int(b) for b in block)
        if bx <= 0 or by <= 0 or bx * by > 1024:
            raise ValueError(f"block must be positive with at most 1024 threads; got {block}")
        super().__init__(params)
        cp = import_cupy()
        self._cp = cp
        self.variant = variant
        self.step_multiple = 2 if variant == "aa" else 1

        module = compile_module(variant, params.dtype, (bx, by))
        real = np.dtype(params.dtype).type
        omega_plus, omega_minus = params.relaxation_rates
        fx, fy = params.force
        # Scalars must carry exact C types: int for nx/ny, real_t for the rest.
        self._scalars = (
            np.int32(params.nx),
            np.int32(params.ny),
            real(omega_plus),
            real(omega_minus),
            real(fx),
            real(fy),
        )
        self._block = (bx, by)
        self._grid = (math.ceil(params.nx / bx), math.ceil(params.ny / by))
        self._solid = cp.asarray(params.solid, dtype=cp.uint8)
        self._wall = cp.ascontiguousarray(cp.asarray(params.wall_velocity, dtype=params.dtype))
        if variant == "aa":
            self._odd = module.get_function("aa_odd")
            self._even = module.get_function("aa_even")
        else:
            self._kernel = module.get_function(f"stream_collide_{variant}")
        self.initialize()

    @property
    def xp(self):
        return self._cp

    @property
    def lattice_bytes(self) -> int:
        """Device memory held by population buffers."""
        buffers = 1 if self.variant == "aa" else 2
        return buffers * Q * self.params.nx * self.params.ny * np.dtype(self.params.dtype).itemsize

    def initialize(self, rho=1.0, ux=0.0, uy=0.0) -> None:
        feq = self._initial_populations(rho, ux, uy)
        if self.variant == "aa":
            # Post-even-step layout: slot opp(i) holds f*_i.
            self._f = self._cp.ascontiguousarray(feq[_OPPOSITE])
        else:
            self._f = self._cp.ascontiguousarray(feq)
            # Both buffers start fully defined; solid nodes are never overwritten.
            self._f_next = self._f.copy()
        self.time_step = 0

    def _advance(self, steps: int) -> None:
        grid, block, scalars = self._grid, self._block, self._scalars
        if self.variant == "aa":
            for _ in range(steps // 2):
                self._odd(grid, block, (self._f, self._solid, self._wall, *scalars))
                self._even(grid, block, (self._f, self._solid, *scalars))
            return
        for _ in range(steps):
            self._kernel(grid, block, (self._f, self._f_next, self._solid, self._wall, *scalars))
            self._f, self._f_next = self._f_next, self._f

    def _device_populations(self):
        if self.variant == "aa":
            return self._f[_OPPOSITE]
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
