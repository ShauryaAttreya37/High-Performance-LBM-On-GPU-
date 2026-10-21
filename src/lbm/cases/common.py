"""Helpers shared by the validation cases."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

from lbm.backends import BACKENDS
from lbm.solver import Solver

OUTPUT_DIR = Path("outputs")


def run_until_steady(
    solver: Solver,
    observe: Callable[[Solver], np.ndarray],
    *,
    interval: int,
    tol: float,
    max_steps: int,
) -> int:
    """Advance in chunks of ``interval`` until ``observe`` changes by less than ``tol``.

    The change is the largest absolute difference of the observed array between
    two consecutive checks. Returns the number of steps taken.

    Raises:
        RuntimeError: No steady state within ``max_steps``.
    """
    previous = observe(solver)
    change = np.inf
    while solver.time_step < max_steps:
        solver.run(interval, check_every=interval)
        current = observe(solver)
        change = float(np.max(np.abs(current - previous)))
        if change < tol:
            return solver.time_step
        previous = current
    raise RuntimeError(
        f"no steady state after {max_steps} steps: last change {change:.3e} >= tol {tol:.1e}"
    )


def even(steps: float) -> int:
    """Round to the nearest even integer >= 2 (the AA backend advances in pairs)."""
    return max(2, 2 * round(steps / 2))


def output_path(name: str) -> Path:
    OUTPUT_DIR.mkdir(exist_ok=True)
    return OUTPUT_DIR / name


def add_backend_arguments(parser) -> None:
    parser.add_argument("--backend", default="numpy", choices=BACKENDS)
    parser.add_argument("--dtype", default="float64", choices=("float32", "float64"))
    parser.add_argument("--collision", default="trt", choices=("bgk", "trt"))
    parser.add_argument("--plot", action="store_true", help="save a figure to outputs/")
