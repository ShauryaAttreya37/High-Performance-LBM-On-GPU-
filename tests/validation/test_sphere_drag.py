"""Sphere in uniform flow (3D): Bouzidi vs IBM vs staircase against drag correlations.

Measured: see ``lbm.cases.sphere_drag``. Re = 20, N = 10: Bouzidi 2.769,
staircase 2.829, IBM 3.112 (Clift-Grace-Weber 2.735, Schiller-Naumann 2.61).
"""

import pytest

from lbm.cases.sphere_drag import clift_grace_weber, run_sphere_drag, schiller_naumann

pytestmark = [pytest.mark.gpu, pytest.mark.slow]


def test_correlations_agree_at_re_100():
    assert schiller_naumann(100) == pytest.approx(clift_grace_weber(100), rel=0.01)


def test_bouzidi_sphere_drag_at_re_20():
    r = run_sphere_drag(10, 20.0, "bouzidi")
    assert r.cd == pytest.approx(r.reference_cgw, rel=0.03)


def test_ibm_sphere_is_too_draggy_at_coarse_resolution():
    ibm = run_sphere_drag(10, 20.0, "ibm")
    assert 0.10 < ibm.cd / ibm.reference_cgw - 1 < 0.25
