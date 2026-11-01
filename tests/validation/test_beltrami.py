"""ABC (Beltrami) flow: viscosity and second-order convergence of the D3Q19 bulk scheme.

Measured (tau = 0.8, float64, NumPy and cuda-aa agree to 4 digits):
TRT L2 errors 6.46e-3, 2.85e-3, 1.62e-3, 7.14e-4, 4.03e-4 at N = 16, 24, 32, 48, 64
(orders 2.02, 1.96, 2.03, 1.98); decay-rate error 1.5e-3 (TRT), 4.1e-3 (BGK) at N = 32.
float32 on the GPU gives the same error as float64 at N = 32.
"""

import pytest

from lbm.cases.beltrami import run_beltrami
from lbm.cases.taylor_green import convergence_order

GPU_3D = [pytest.param(b, marks=pytest.mark.gpu) for b in ("cuda-naive", "cuda-aa")]


@pytest.mark.parametrize("backend", ["numpy", *GPU_3D])
@pytest.mark.parametrize("collision", ["bgk", "trt"])
def test_decay_rate_matches_lattice_viscosity(backend, collision):
    result = run_beltrami(n=32, collision=collision, backend=backend)
    assert result.decay_rate_error < 1e-2


def test_velocity_error_converges_at_second_order_on_cpu():
    results = [run_beltrami(n=n) for n in (16, 24, 32)]
    orders = convergence_order(results)
    assert all(1.9 < order < 2.1 for order in orders), orders


@pytest.mark.gpu
@pytest.mark.parametrize("backend", ["cuda-naive", "cuda-aa"])
def test_velocity_error_converges_at_second_order_on_gpu(backend):
    results = [run_beltrami(n=n, backend=backend) for n in (32, 48, 64)]
    orders = convergence_order(results)
    assert all(1.9 < order < 2.1 for order in orders), orders


@pytest.mark.gpu
def test_float32_matches_float64_accuracy():
    single = run_beltrami(n=32, backend="cuda-aa", dtype="float32")
    double = run_beltrami(n=32, backend="cuda-aa", dtype="float64")
    assert single.l2_error == pytest.approx(double.l2_error, rel=1e-2)
