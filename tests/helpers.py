"""Small builders shared by several test modules."""

import numpy as np

from lbm.equilibrium import equilibrium


def random_macroscopic(shape=(5, 7), seed=0, dtype=np.float64, scale=0.05):
    """Random density deviation and velocity fields of a given shape."""
    rng = np.random.default_rng(seed)
    drho, ux, uy = (scale * rng.standard_normal(shape).astype(dtype) for _ in range(3))
    return drho, ux, uy


def random_populations(shape=(5, 7), seed=0, dtype=np.float64, noise=1e-3):
    """Equilibrium of random fields plus a random non-equilibrium part."""
    drho, ux, uy = random_macroscopic(shape, seed, dtype)
    rng = np.random.default_rng(seed + 1000)
    f = equilibrium(drho, ux, uy)
    return f + (noise * rng.standard_normal(f.shape)).astype(dtype)


GPU_BACKENDS = ("cupy", "cuda-naive", "cuda-tiled", "cuda-aa")


def all_backends():
    """pytest parameters: ``numpy`` always, GPU backends marked ``gpu``.

    ``cupy`` (the unfused reference on the GPU) launches ~100 tiny kernels per
    step, so on the small validation grids it is launch-bound and slower than
    NumPy; it is additionally marked ``slow``.
    """
    import pytest

    def marks(backend):
        return [pytest.mark.gpu, pytest.mark.slow] if backend == "cupy" else [pytest.mark.gpu]

    return [pytest.param("numpy"), *(pytest.param(b, marks=marks(b)) for b in GPU_BACKENDS)]


def compile_to_ptx(source, arch="compute_86"):
    """Compile CUDA source with NVRTC alone (no GPU or driver context needed).

    Skips the calling test when CuPy's NVRTC bindings are not installed.
    """
    import pytest

    nvrtc = pytest.importorskip("cupy_backends.cuda.libs.nvrtc")
    program = nvrtc.createProgram(source, "kernel.cu", [], [])
    try:
        nvrtc.compileProgram(program, [f"-arch={arch}", "--std=c++17"])
    except Exception as exc:
        raise AssertionError(f"NVRTC failed:\n{nvrtc.getProgramLog(program)}") from exc
    ptx = nvrtc.getPTX(program)
    return ptx.decode() if isinstance(ptx, bytes) else ptx
