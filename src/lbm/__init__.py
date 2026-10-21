"""D2Q9 lattice Boltzmann solver with a NumPy reference and hand-written CUDA kernels.

Typical use::

    from lbm import SimulationParams, make_solver

    params = SimulationParams(nx=256, ny=256, tau=0.6, collision="trt")
    solver = make_solver(params, backend="cuda-aa")
    solver.initialize(rho=1.0, ux=0.0, uy=0.0)
    solver.run(1000)
    fields = solver.macroscopic()
"""

from lbm.backends import BACKENDS, BackendUnavailable, available_backends, make_solver
from lbm.params import SimulationParams, StabilityWarning
from lbm.solver import SimulationDiverged
from lbm.units import UnitConverter

__version__ = "0.1.0"

__all__ = [
    "BACKENDS",
    "BackendUnavailable",
    "SimulationDiverged",
    "SimulationParams",
    "StabilityWarning",
    "UnitConverter",
    "available_backends",
    "make_solver",
]
