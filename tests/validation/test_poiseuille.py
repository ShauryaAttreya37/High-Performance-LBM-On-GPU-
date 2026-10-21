"""Poiseuille flow: wall placement of halfway bounce-back under BGK and TRT.

Thresholds come from measured errors (H = 16, float64):
TRT ~3e-13 for every tau; BGK 2.8e-3 at tau = 0.8, 2.3e-2 at tau = 1.5, and
~3e-13 at tau = 1/2 + sqrt(3)/4; BGK convergence order 2.000.
"""

import math

import numpy as np
import pytest

from lbm.cases.poiseuille import BGK_EXACT_TAU, run_poiseuille
from tests.helpers import all_backends

ROUND_OFF = 1e-10


@pytest.mark.parametrize("backend", all_backends())
@pytest.mark.parametrize("tau", [0.6, 2.0])
def test_trt_with_magic_parameter_is_exact_for_any_tau(backend, tau):
    assert run_poiseuille(tau=tau, collision="trt", backend=backend).l2_error < ROUND_OFF


@pytest.mark.parametrize("backend", all_backends())
def test_bgk_is_exact_only_at_its_magic_tau(backend):
    assert run_poiseuille(tau=BGK_EXACT_TAU, collision="bgk", backend=backend).l2_error < ROUND_OFF


def test_bgk_wall_slip_grows_with_tau():
    errors = [run_poiseuille(tau=tau, collision="bgk").l2_error for tau in (1.2, 1.5, 2.0)]
    assert errors == sorted(errors)
    assert errors[1] > 1e-2, "BGK at tau = 1.5 misplaces the wall by a visible amount"


@pytest.mark.parametrize("tau", [0.6, 1.5])
def test_bgk_error_is_a_uniform_slip_given_by_the_magic_parameter(tau):
    # Closed form for halfway bounce-back with TRT/BGK in Poiseuille flow:
    # u_num(y) = u_exact(y) + u_slip,  u_slip = (2/3) (Lambda - 3/16) F / (rho nu),
    # with Lambda = (tau - 1/2)^2 for BGK. It vanishes exactly at Lambda = 3/16.
    height, max_velocity = 16, 0.01
    result = run_poiseuille(height=height, tau=tau, collision="bgk", max_velocity=max_velocity)
    nu = (tau - 0.5) / 3.0
    force = 8.0 * nu * max_velocity / height**2
    slip = (2.0 / 3.0) * ((tau - 0.5) ** 2 - 3.0 / 16.0) * force / nu
    np.testing.assert_allclose(result.u - result.u_exact, slip, rtol=1e-9, atol=1e-15)


def test_bgk_slip_error_converges_at_second_order():
    errors = [run_poiseuille(height=h, tau=1.5, collision="bgk").l2_error for h in (8, 16, 32)]
    orders = [math.log2(coarse / fine) for coarse, fine in zip(errors, errors[1:], strict=False)]
    assert all(1.95 < order < 2.05 for order in orders), orders


@pytest.mark.gpu
@pytest.mark.parametrize("backend", ["cuda-naive", "cuda-aa"])
def test_float32_trt_stays_at_single_precision_level(backend):
    # Measured ~1.6e-5: float32 round-off accumulated over ~10^4 steps of a small force.
    result = run_poiseuille(tau=0.8, collision="trt", backend=backend, dtype="float32")
    assert result.l2_error < 1e-4
