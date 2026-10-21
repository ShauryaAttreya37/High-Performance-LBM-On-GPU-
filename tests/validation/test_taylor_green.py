"""Taylor-Green vortex: viscosity and second-order convergence of the bulk scheme.

Measured (tau = 0.8, TRT, float64): decay-rate error 3.9e-3 at N = 32; L2 errors
1.58e-2, 3.92e-3, 9.80e-4 at N = 16, 32, 64, i.e. order 2.02 and 2.00.
"""

import pytest

from lbm.cases.taylor_green import convergence_order, run_taylor_green
from tests.helpers import all_backends


@pytest.mark.parametrize("backend", all_backends())
@pytest.mark.parametrize("collision", ["bgk", "trt"])
def test_decay_rate_matches_lattice_viscosity(backend, collision):
    result = run_taylor_green(n=32, collision=collision, backend=backend)
    assert result.decay_rate_error < 1e-2


@pytest.mark.parametrize("backend", all_backends())
def test_velocity_error_converges_at_second_order(backend):
    results = [run_taylor_green(n=n, backend=backend) for n in (16, 32, 64)]
    orders = convergence_order(results)
    assert all(1.9 < order < 2.1 for order in orders), orders
