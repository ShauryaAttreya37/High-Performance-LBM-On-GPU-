"""Small builders shared by several test modules."""

import numpy as np
import pytest

from lbm.equilibrium import equilibrium
from lbm.lattice import LATTICES

SHAPES = {2: (5, 7), 3: (3, 4, 5)}
"""Small, non-square field shapes per dimension ((y, x) and (z, y, x))."""

ALL_LATTICES = pytest.mark.parametrize("vs", list(LATTICES.values()), ids=list(LATTICES))
"""Parametrise a test over every supported velocity set, as argument ``vs``."""


def random_macroscopic(vs, seed=0, dtype=np.float64, scale=0.05, shape=None):
    """Random density deviation and a d-component velocity field."""
    shape = SHAPES[vs.d] if shape is None else shape
    rng = np.random.default_rng(seed)
    drho = (scale * rng.standard_normal(shape)).astype(dtype)
    u = tuple((scale * rng.standard_normal(shape)).astype(dtype) for _ in range(vs.d))
    return drho, u


def random_populations(vs, seed=0, dtype=np.float64, noise=1e-3, shape=None):
    """Equilibrium of random fields plus a random non-equilibrium part."""
    drho, u = random_macroscopic(vs, seed, dtype, shape=shape)
    rng = np.random.default_rng(seed + 1000)
    f = equilibrium(vs, drho, u)
    return f + (noise * rng.standard_normal(f.shape)).astype(dtype)


def first_moments(vs, f):
    """sum_i c_ia f_i for each axis a, computed independently of the code under test."""
    return tuple(np.tensordot(vs.c_array[:, a].astype(f.dtype), f, axes=1) for a in range(vs.d))


GPU_BACKENDS = ("cupy", "cuda-naive", "cuda-tiled", "cuda-aa")


def all_backends():
    """pytest parameters: ``numpy`` always, GPU backends marked ``gpu``.

    ``cupy`` (the unfused reference on the GPU) launches ~100 tiny kernels per
    step, so on the small validation grids it is launch-bound and slower than
    NumPy; it is additionally marked ``slow``.
    """

    def marks(backend):
        return [pytest.mark.gpu, pytest.mark.slow] if backend == "cupy" else [pytest.mark.gpu]

    return [pytest.param("numpy"), *(pytest.param(b, marks=marks(b)) for b in GPU_BACKENDS)]


def compile_to_ptx(source, arch="compute_86"):
    """Compile CUDA source with NVRTC alone (no GPU or driver context needed).

    Skips the calling test when CuPy's NVRTC bindings are not installed.
    """
    nvrtc = pytest.importorskip("cupy_backends.cuda.libs.nvrtc")
    program = nvrtc.createProgram(source, "kernel.cu", [], [])
    try:
        nvrtc.compileProgram(program, [f"-arch={arch}", "--std=c++17"])
    except Exception as exc:
        raise AssertionError(f"NVRTC failed:\n{nvrtc.getProgramLog(program)}") from exc
    ptx = nvrtc.getPTX(program)
    return ptx.decode() if isinstance(ptx, bytes) else ptx
