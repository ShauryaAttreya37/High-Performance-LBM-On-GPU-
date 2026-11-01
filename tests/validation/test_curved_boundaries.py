"""Curved bodies: staircase vs Bouzidi vs IBM on problems with known answers.

Measured (2D square array of cylinders, c = 0.05, Re = 0.1, tau = 1, TRT,
cuda-aa; error of the drag against Sangani & Acrivos):

    L   R      staircase   Bouzidi    IBM
    32  4.04   +0.56 %     -0.72 %    +12.6 %
    48  6.06   +2.65 %     -0.18 %    +7.5 %
    64  8.07   +1.15 %     -0.10 %    +5.4 %
    96  12.1   +1.24 %     -0.034 %   +3.4 %
    192 24.2   +0.32 %     +0.001 %

3D simple cubic array of spheres, c = 0.01 (Hasimoto): Bouzidi -2.63, -0.95,
-0.44 % at R = 4.3, 6.4, 8.6; staircase -3.7, +0.03, +0.96 %.
3D pipe (Hagen-Poiseuille, tau = 0.8): Bouzidi L2 error 9.7e-3 (R = 4.3) to
1.9e-4 (R = 24.3), overall order 2.3; staircase 6.3e-2 to 8.6e-3, erratic.
"""

import math

import pytest

from lbm.cases.pipe_flow import run_pipe_flow
from lbm.cases.stokes_array import run_stokes_array

GPU_AA = pytest.param("cuda-aa", marks=pytest.mark.gpu)


@pytest.mark.parametrize("backend", ["numpy", GPU_AA])
def test_bouzidi_cylinder_array_drag_converges_to_theory(backend):
    coarse = run_stokes_array(2, 32, method="bouzidi", backend=backend)
    fine = run_stokes_array(2, 64, method="bouzidi", backend=backend)
    assert abs(coarse.error) < 1e-2 and abs(fine.error) < 2e-3
    assert abs(fine.error) < abs(coarse.error) / 4


@pytest.mark.parametrize("method", ["staircase", "bouzidi"])
def test_momentum_exchange_balances_the_driving_force(method):
    result = run_stokes_array(2, 32, method=method)
    # Converged to a 1e-7 change per check, so the balance holds to ~1e-7 as well.
    assert result.momentum_exchange == pytest.approx(1.0, abs=1e-6)


@pytest.mark.slow
def test_ibm_cylinder_array_is_first_order_and_too_draggy():
    errors = [run_stokes_array(2, size, method="ibm").error for size in (32, 64)]
    assert 0.08 < errors[0] < 0.17 and 0.03 < errors[1] < 0.08
    assert 1.8 < errors[0] / errors[1] < 3.0  # halving dx roughly halves the error


def test_ibm_marker_force_balances_the_driving_force():
    result = run_stokes_array(2, 32, method="ibm")
    assert result.momentum_exchange == pytest.approx(1.0, abs=1e-6)


@pytest.mark.gpu
def test_bouzidi_sphere_array_drag_converges_to_hasimoto():
    errors = [
        run_stokes_array(3, size, 0.01, "bouzidi", backend="cuda-aa").error for size in (32, 64)
    ]
    assert abs(errors[1]) < 6e-3 and abs(errors[1]) < abs(errors[0]) / 4


@pytest.mark.parametrize("backend", ["numpy", GPU_AA])
def test_bouzidi_pipe_beats_the_staircase(backend):
    bouzidi = run_pipe_flow(8.3, "bouzidi", backend=backend)
    staircase = run_pipe_flow(8.3, "staircase", backend=backend)
    assert bouzidi.l2_error < 4e-3 and bouzidi.l2_error < staircase.l2_error / 5


@pytest.mark.gpu
def test_bouzidi_pipe_error_falls_at_about_second_order():
    small, large = (run_pipe_flow(r, "bouzidi", backend="cuda-aa").l2_error for r in (4.3, 24.3))
    order = math.log(small / large) / math.log(24.3 / 4.3)
    assert 1.8 < order < 2.8
