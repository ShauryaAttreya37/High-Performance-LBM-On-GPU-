"""Square and rectangular duct flow: bounce-back walls in two directions at once (D3Q19).

Measured (tau = 1, TRT, float64, NumPy and cuda-aa identical): L2 errors
4.01e-3, 8.76e-4, 2.09e-4, 5.16e-5 at H = 8, 16, 32, 64 (orders 2.19, 2.06, 2.02).
BGK: 5.91e-3 ... 1.12e-4, orders 1.80, 1.94, 1.98. 12 x 20 duct: 8.8e-4.
"""

import math

import numpy as np
import pytest

from lbm.cases.duct_flow import exact_profile, run_duct_flow


def orders(heights, errors):
    pairs = list(zip(heights, errors, strict=True))
    return [
        math.log(e0 / e1) / math.log(h1 / h0)
        for (h0, e0), (h1, e1) in zip(pairs, pairs[1:], strict=False)
    ]


def test_series_reduces_to_the_plate_parabola_for_a_wide_duct():
    height, width, force, nu = 16.0, 400.0, 1e-5, 0.1
    y = np.arange(16) + 0.5
    u = exact_profile(y, np.array([0.5 * width]), height, width, force, nu)[0]
    np.testing.assert_allclose(u, force / (2 * nu) * y * (height - y), rtol=1e-9)


def test_series_vanishes_on_the_walls_and_is_symmetric():
    height = width = 10.0
    s = np.linspace(0.0, 10.0, 21)
    u = exact_profile(s, s, height, width, 1e-5, 0.1)
    assert np.abs(u[0]).max() < 1e-12 and np.abs(u[:, 0]).max() < 1e-12
    np.testing.assert_allclose(u, u.T, atol=1e-15)


@pytest.mark.slow
def test_trt_converges_at_second_order_on_cpu():
    heights = [8, 16, 32]
    errors = [run_duct_flow(height=h).l2_error for h in heights]
    # The coarsest pair is slightly super-convergent (2.19): the leading error term
    # still competes with higher-order corner corrections at H = 8.
    assert all(1.9 < o < 2.3 for o in orders(heights, errors)), errors


@pytest.mark.gpu
@pytest.mark.parametrize("backend", ["cuda-naive", "cuda-aa"])
def test_trt_converges_at_second_order_on_gpu(backend):
    heights = [16, 32, 64]
    errors = [run_duct_flow(height=h, backend=backend).l2_error for h in heights]
    assert all(1.95 < o < 2.1 for o in orders(heights, errors)), errors


@pytest.mark.gpu
@pytest.mark.parametrize("backend", ["cuda-naive", "cuda-aa"])
def test_gpu_matches_cpu_result(backend):
    cpu = run_duct_flow(height=12, width=20)
    gpu = run_duct_flow(height=12, width=20, backend=backend)
    assert gpu.steps == cpu.steps
    np.testing.assert_allclose(gpu.u, cpu.u, rtol=1e-11, atol=1e-16)
    assert cpu.l2_error < 1e-3


@pytest.mark.gpu
def test_float32_reaches_the_same_accuracy():
    result = run_duct_flow(height=16, backend="cuda-aa", dtype="float32")
    assert result.l2_error < 1e-3
