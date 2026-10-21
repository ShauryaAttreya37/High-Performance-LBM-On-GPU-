"""CUDA backends: source assembly (CPU) and bit-level agreement with the NumPy reference (GPU)."""

import numpy as np
import pytest

from lbm import SimulationParams, lattice, make_solver
from lbm.backends.cuda_solver import VARIANTS, CudaSolver, build_kernel_source
from lbm.boundaries import channel_walls

GPU_BACKENDS = ["cupy", "cuda-naive", "cuda-tiled", "cuda-aa"]
EQUIVALENCE_STEPS = 40


def mixed_geometry(nx=45, ny=29):
    """Channel with a moving top wall and an interior obstacle.

    The size is deliberately not a multiple of the 32 x 8 block, so partial
    blocks, halo wrap-around and every kind of wall link are all exercised.
    """
    solid = channel_walls(nx, ny)
    solid[10:16, 12:19] = True
    wall = np.zeros((2, ny, nx))
    wall[0, -1, :] = 0.05
    return solid, wall


def mixed_params(dtype, collision):
    solid, wall = mixed_geometry()
    return SimulationParams(
        nx=45,
        ny=29,
        tau=0.7,
        collision=collision,
        force=(1e-5, -2e-6),
        solid=solid,
        wall_velocity=wall,
        dtype=dtype,
    )


def random_state(params, seed=0):
    rng = np.random.default_rng(seed)
    shape = (params.ny, params.nx)
    return 1 + 0.01 * rng.standard_normal(shape), *(0.02 * rng.standard_normal((2, *shape)))


# ---------------------------------------------------------------- CPU-only checks


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize(("dtype", "c_type"), [("float32", "float"), ("float64", "double")])
def test_kernel_source_contains_preamble_physics_and_kernel(variant, dtype, c_type):
    source = build_kernel_source(variant, dtype, block=(16, 4))
    assert f"typedef {c_type} real_t;" in source
    assert "#define TILE_X 16" in source and "#define TILE_Y 4" in source
    assert "collide_node" in source
    kernels = ("aa_even", "aa_odd") if variant == "aa" else (f"stream_collide_{variant}",)
    for name in kernels:
        assert f'extern "C" __global__ void {name}(' in source


def test_unknown_variant_is_rejected_before_touching_the_gpu():
    params = SimulationParams(nx=4, ny=4, tau=0.8)
    with pytest.raises(ValueError, match="variant"):
        CudaSolver(params, variant="warp-specialised")
    with pytest.raises(ValueError, match="variant"):
        build_kernel_source("bogus", "float32")


def test_unsupported_kernel_dtype_is_rejected():
    with pytest.raises(ValueError, match="dtype"):
        build_kernel_source("naive", "float16")


# ---------------------------------------------------------------- GPU checks


@pytest.mark.gpu
@pytest.mark.parametrize("vs", list(lattice.LATTICES.values()), ids=list(lattice.LATTICES))
@pytest.mark.parametrize("c_type", ["float", "double"])
def test_generated_constants_evaluate_correctly_on_device(vs, c_type):
    import cupy as cp

    source = (
        lattice.cuda_preamble(vs, c_type)
        + r"""
    extern "C" __global__ void dump(int* ci, real_t* w) {
        const int i = threadIdx.x;
        ci[i] = cx(i); ci[Q + i] = cy(i); ci[2 * Q + i] = cz(i); ci[3 * Q + i] = opp(i);
        w[i] = weight(i);
    }"""
    )
    kernel = cp.RawModule(code=source).get_function("dump")
    ci = cp.zeros(4 * vs.q, dtype=cp.int32)
    w = cp.zeros(vs.q, dtype=cp.float32 if c_type == "float" else cp.float64)
    kernel((1,), (vs.q,), (ci, w))
    ci = ci.get().reshape(4, vs.q)
    for axis in range(3):
        np.testing.assert_array_equal(ci[axis], vs.component(axis))
    np.testing.assert_array_equal(ci[3], vs.opposite)
    np.testing.assert_allclose(w.get(), vs.w_array, rtol=1e-7 if c_type == "float" else 1e-16)


@pytest.mark.gpu
@pytest.mark.parametrize("collision", ["bgk", "trt"])
@pytest.mark.parametrize("backend", GPU_BACKENDS)
def test_float64_matches_reference_to_round_off(backend, collision):
    params = mixed_params("float64", collision)
    reference = make_solver(params, backend="numpy")
    gpu = make_solver(params, backend=backend)
    for solver in (reference, gpu):
        solver.initialize(*random_state(params))
        solver.run(EQUIVALENCE_STEPS)
    fluid = params.fluid
    diff = np.abs(reference.populations()[:, fluid] - gpu.populations()[:, fluid]).max()
    # Measured ~1e-17 (a few ulps of |f| ~ 1.6e-2); 1e-15 leaves ~100x margin for FMA
    # contraction and summation-order differences between compilers and GPUs.
    assert diff < 1e-15


@pytest.mark.gpu
@pytest.mark.parametrize("backend", GPU_BACKENDS)
def test_float32_matches_float32_reference(backend):
    params = mixed_params("float32", "trt")
    reference = make_solver(params, backend="numpy")
    gpu = make_solver(params, backend=backend)
    for solver in (reference, gpu):
        solver.initialize(*random_state(params))
        solver.run(EQUIVALENCE_STEPS)
    fluid = params.fluid
    diff = np.abs(reference.populations()[:, fluid] - gpu.populations()[:, fluid]).max()
    # Measured ~8e-9 (about 4 float32 ulps of |f| ~ 1.6e-2).
    assert diff < 1e-7


@pytest.mark.gpu
@pytest.mark.parametrize("backend", GPU_BACKENDS)
def test_macroscopic_fields_match_reference(backend):
    params = mixed_params("float64", "trt")
    reference = make_solver(params, backend="numpy")
    gpu = make_solver(params, backend=backend)
    for solver in (reference, gpu):
        solver.initialize(*random_state(params))
        solver.run(EQUIVALENCE_STEPS)
    ref, out = reference.macroscopic(), gpu.macroscopic()
    for name in ("rho", "ux", "uy"):
        np.testing.assert_allclose(getattr(out, name), getattr(ref, name), atol=1e-14)


@pytest.mark.gpu
def test_aa_backend_advances_in_pairs_only():
    solver = make_solver(SimulationParams(nx=8, ny=8, tau=0.8), backend="cuda-aa")
    with pytest.raises(ValueError, match="multiples of 2"):
        solver.run(3)
    solver.run(4)
    assert solver.time_step == 4


@pytest.mark.gpu
def test_aa_backend_uses_half_the_lattice_memory():
    params = SimulationParams(nx=64, ny=32, tau=0.8)
    naive = make_solver(params, backend="cuda-naive")
    aa = make_solver(params, backend="cuda-aa")
    assert aa.lattice_bytes * 2 == naive.lattice_bytes == 2 * 9 * 64 * 32 * 4


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("backend", ["cuda-naive", "cuda-tiled", "cuda-aa"])
def test_kernels_keep_populations_in_registers(backend, dtype):
    solver = make_solver(SimulationParams(nx=8, ny=8, tau=0.8, dtype=dtype), backend=backend)
    for name, resources in solver.kernel_resources().items():
        assert resources["local_bytes"] == 0, f"{name} spills to local memory"


@pytest.mark.gpu
def test_invalid_block_is_rejected():
    with pytest.raises(ValueError, match="block"):
        make_solver(SimulationParams(nx=8, ny=8, tau=0.8), backend="cuda-naive", block=(64, 32))
