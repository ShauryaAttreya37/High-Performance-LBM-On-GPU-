"""Backend selection.

==============  ====================================================================
``numpy``       ReferenceSolver on the CPU. Correctness oracle; what CI runs.
``cupy``        Same ReferenceSolver code on GPU arrays. Unfused: every step is a
                chain of small kernels, each streaming the whole lattice through
                global memory. Shows what a GPU gives you "for free".
``cuda-naive``  One fused stream-bounce-collide kernel, two lattices (ping-pong).
``cuda-tiled``  Same, but neighbours are staged through shared memory.
``cuda-aa``     AA access pattern: one lattice updated in place, half the memory.
==============  ====================================================================
"""

from __future__ import annotations

import functools

import numpy as np

from lbm.params import SimulationParams
from lbm.solver import ReferenceSolver, Solver

BACKENDS = ("numpy", "cupy", "cuda-naive", "cuda-tiled", "cuda-aa")


class BackendUnavailable(RuntimeError):
    """The requested backend needs CuPy and a CUDA device, and one is missing."""


def import_cupy():
    """Import CuPy and check that a CUDA device is present."""
    try:
        import cupy
    except ImportError as exc:
        raise BackendUnavailable(
            "CuPy is not installed; install the GPU extra with `uv sync --extra gpu`"
        ) from exc
    try:
        count = cupy.cuda.runtime.getDeviceCount()
    except cupy.cuda.runtime.CUDARuntimeError as exc:
        raise BackendUnavailable(f"no usable CUDA device: {exc}") from exc
    if count == 0:
        raise BackendUnavailable("no CUDA device found")
    return cupy


@functools.cache
def gpu_available() -> bool:
    """True when CuPy imports and at least one CUDA device is visible."""
    try:
        import_cupy()
    except BackendUnavailable:
        return False
    return True


def available_backends() -> tuple[str, ...]:
    """Backends usable on this machine."""
    return BACKENDS if gpu_available() else ("numpy",)


def make_solver(params: SimulationParams, backend: str = "numpy", **options) -> Solver:
    """Create a solver for ``params`` on the chosen backend.

    Args:
        params: Simulation description.
        backend: One of ``BACKENDS``.
        **options: Passed to the CUDA solver (e.g. ``block=(32, 8)``).

    Raises:
        ValueError: Unknown backend name.
        BackendUnavailable: GPU backend requested without CuPy or a device.
    """
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend {backend!r}; choose from {BACKENDS}")
    if backend == "numpy":
        return ReferenceSolver(params, xp=np)
    cupy = import_cupy()
    if backend == "cupy":
        return ReferenceSolver(params, xp=cupy)
    from lbm.backends.cuda_solver import CudaSolver

    return CudaSolver(params, variant=backend.removeprefix("cuda-"), **options)
