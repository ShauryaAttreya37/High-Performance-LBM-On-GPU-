"""Schaefer-Turek cylinder benchmark (Bouzidi cylinder, Zou-He inlet/outlet).

Measured (cuda-aa, TRT):
2D-1, float64, u_max = 0.01875: N = 20: C_D 5.599, C_L 0.01076, dp 0.1158;
N = 40: C_D 5.589, C_L 0.01058, dp 0.1170 (benchmark: [5.57, 5.59],
[0.0104, 0.0110], [0.1172, 0.1176]). With u_max = 0.075 the drag is 2 % high
at every N: weak compressibility, error ~ Ma^2.
2D-2, float32, u_max = 0.05, start from the inflow profile: N = 40:
C_D,max 3.286, C_L,max 1.004, St 0.3014 (benchmark: [3.22, 3.24],
[0.99, 1.01], [0.295, 0.305]).
"""

import pytest

from lbm.cases.cylinder_benchmark import build, run_periodic, run_steady

pytestmark = [pytest.mark.gpu, pytest.mark.slow]


def test_steady_2d1_drag_lift_and_pressure_drop():
    r = run_steady(20, backend="cuda-aa")
    assert r.cd == pytest.approx(5.58, rel=0.01)
    assert 0.0104 < r.cl < 0.0110
    assert r.dp == pytest.approx(0.1174, rel=0.02)


def test_periodic_2d2_shedding_frequency_and_lift():
    r = run_periodic(20)
    assert 0.295 < r.strouhal < 0.305
    assert r.cl_max == pytest.approx(1.0, rel=0.08)
    assert r.cd_max == pytest.approx(3.23, rel=0.03)


def test_build_rejects_bad_inputs():
    with pytest.raises(ValueError, match="test"):
        build("3D-1", 20)
    with pytest.raises(ValueError, match="multiple of 10"):
        build("2D-1", 25)
