"""CUDA backends: source assembly and compilation (CPU), agreement with the reference (GPU).

Every GPU check runs on a 2D geometry and a 3D one: the same kernel sources
serve both, so a bug in the generic indexing would show up in only one.
"""

import numpy as np
import pytest

from lbm import SimulationParams, lattice, make_solver
from lbm.backends.cuda_solver import TWO_D_ONLY, VARIANTS, CudaSolver, build_kernel_source
from lbm.boundaries import channel_walls
from tests.helpers import compile_to_ptx

GPU_BACKENDS = ["cupy", "cuda-naive", "cuda-tiled", "cuda-aa"]
GPU_BACKENDS_3D = [b for b in GPU_BACKENDS if b.removeprefix("cuda-") not in TWO_D_ONLY]
EQUIVALENCE_STEPS = 40
DIMENSIONS = [2, 3]


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


def mixed_geometry_3d(nx=24, ny=20, nz=16):
    """Box walls in y and z, periodic in x, a moving top plate and an obstacle.

    Non-cubic and not a multiple of the 32 x 8 block, so partial blocks and
    wrap-around on every axis are exercised; the plate moves in x and z so
    every component of the wall term is used.
    """
    solid = channel_walls(nx, ny, nz)
    solid[0, :, :] = True
    solid[-1, :, :] = True
    solid[6:10, 7:12, 9:14] = True
    wall = np.zeros((3, nz, ny, nx))
    wall[0, :, -1, :] = 0.05
    wall[2, :, -1, :] = -0.02
    return solid, wall


def mixed_params(dtype, collision, d=2):
    if d == 2:
        solid, wall = mixed_geometry()
        sizes, force = {"nx": 45, "ny": 29}, (1e-5, -2e-6)
    else:
        solid, wall = mixed_geometry_3d()
        sizes, force = {"nx": 24, "ny": 20, "nz": 16}, (1e-5, -2e-6, 3e-6)
    return SimulationParams(
        **sizes,
        tau=0.7,
        collision=collision,
        force=force,
        solid=solid,
        wall_velocity=wall,
        dtype=dtype,
    )


def random_state(params, seed=0):
    rng = np.random.default_rng(seed)
    shape = params.shape
    rho = 1 + 0.01 * rng.standard_normal(shape)
    u = 0.02 * rng.standard_normal((params.d, *shape))
    return rho, *u


def run_pair(params, backend):
    """The NumPy reference and ``backend`` after the same steps from the same random state."""
    reference = make_solver(params, backend="numpy")
    gpu = make_solver(params, backend=backend)
    for solver in (reference, gpu):
        solver.initialize(*random_state(params))
        solver.run(EQUIVALENCE_STEPS)
    return reference, gpu


def gpu_cases(backends_2d=GPU_BACKENDS, backends_3d=GPU_BACKENDS_3D):
    """(backend, d) pairs: every backend in 2D, the 3D-capable ones in 3D."""
    return [(b, 2) for b in backends_2d] + [(b, 3) for b in backends_3d]


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


def test_unknown_lattice_is_rejected():
    with pytest.raises(ValueError, match="lattice"):
        build_kernel_source("naive", "float32", lattice="D3Q27")


@pytest.mark.parametrize("variant", [v for v in VARIANTS if v not in TWO_D_ONLY])
def test_3d_source_uses_the_d3q19_preamble(variant):
    source = build_kernel_source(variant, "float32", lattice="D3Q19")
    assert "#define DIM 3" in source and "#define Q 19" in source


@pytest.mark.parametrize("variant", TWO_D_ONLY)
def test_2d_only_variants_refuse_3d(variant):
    with pytest.raises(ValueError, match="2D only"):
        build_kernel_source(variant, "float32", lattice="D3Q19")
    with pytest.raises(ValueError, match="2D only"):
        CudaSolver(SimulationParams(nx=8, ny=8, nz=8, tau=0.8), variant=variant)


COMPILE_CASES = [
    (v, lat)
    for lat in ("D2Q9", "D3Q19")
    for v in VARIANTS
    if not (lat == "D3Q19" and v in TWO_D_ONLY)
]


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize(("variant", "lattice_name"), COMPILE_CASES)
def test_every_kernel_compiles_with_nvrtc(variant, lattice_name, dtype):
    """NVRTC needs no GPU, so CI catches kernel syntax errors without one."""
    ptx = compile_to_ptx(build_kernel_source(variant, dtype, lattice=lattice_name))
    kernels = ("aa_even", "aa_odd") if variant == "aa" else (f"stream_collide_{variant}",)
    for name in kernels:
        assert f".entry {name}(" in ptx


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
@pytest.mark.parametrize(("backend", "d"), gpu_cases())
def test_float64_matches_reference_to_round_off(backend, d, collision):
    params = mixed_params("float64", collision, d)
    reference, gpu = run_pair(params, backend)
    fluid = params.fluid
    diff = np.abs(reference.populations()[:, fluid] - gpu.populations()[:, fluid]).max()
    # Measured ~1e-17 (a few ulps of |f| ~ 1.6e-2); 1e-15 leaves ~100x margin for FMA
    # contraction and summation-order differences between compilers and GPUs.
    assert diff < 1e-15


@pytest.mark.gpu
@pytest.mark.parametrize(("backend", "d"), gpu_cases())
def test_float32_matches_float32_reference(backend, d):
    params = mixed_params("float32", "trt", d)
    reference, gpu = run_pair(params, backend)
    fluid = params.fluid
    diff = np.abs(reference.populations()[:, fluid] - gpu.populations()[:, fluid]).max()
    # Measured ~8e-9 (about 4 float32 ulps of |f| ~ 1.6e-2).
    assert diff < 1e-7


@pytest.mark.gpu
@pytest.mark.parametrize(("backend", "d"), gpu_cases())
def test_macroscopic_fields_match_reference(backend, d):
    params = mixed_params("float64", "trt", d)
    reference, gpu = run_pair(params, backend)
    ref, out = reference.macroscopic(), gpu.macroscopic()
    np.testing.assert_allclose(out.rho, ref.rho, atol=1e-14)
    for axis in range(d):
        np.testing.assert_allclose(out.u[axis], ref.u[axis], atol=1e-14)


@pytest.mark.gpu
@pytest.mark.parametrize("backend", ["cuda-naive", "cuda-aa"])
def test_closed_3d_box_with_moving_lid_conserves_mass(backend):
    from lbm.boundaries import cavity_walls

    solid, wall = cavity_walls(n=14, lid_velocity=0.05, dimensions=3)
    params = SimulationParams(
        nx=16, ny=16, nz=16, tau=0.7, solid=solid, wall_velocity=wall, dtype="float64"
    )
    solver = make_solver(params, backend=backend)
    mass0 = solver.macroscopic().total_mass
    solver.run(400)
    assert solver.macroscopic().total_mass == pytest.approx(mass0, rel=1e-13)


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
def test_3d_lattice_memory_counts_nineteen_populations_per_node():
    params = SimulationParams(nx=16, ny=12, nz=8, tau=0.8)
    assert make_solver(params, backend="cuda-aa").lattice_bytes == 19 * 16 * 12 * 8 * 4


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize(("backend", "d"), gpu_cases(GPU_BACKENDS[1:], GPU_BACKENDS_3D[1:]))
def test_kernels_keep_populations_in_registers(backend, d, dtype):
    sizes = {"nx": 8, "ny": 8} | ({"nz": 8} if d == 3 else {})
    solver = make_solver(SimulationParams(**sizes, tau=0.8, dtype=dtype), backend=backend)
    for name, resources in solver.kernel_resources().items():
        assert resources["local_bytes"] == 0, f"{name} spills to local memory"


@pytest.mark.gpu
def test_invalid_block_is_rejected():
    with pytest.raises(ValueError, match="block"):
        make_solver(SimulationParams(nx=8, ny=8, tau=0.8), backend="cuda-naive", block=(64, 32))
