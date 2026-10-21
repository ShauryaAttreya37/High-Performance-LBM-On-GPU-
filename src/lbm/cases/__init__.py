"""Validation cases with known answers.

Each module exposes a ``run_*`` function (used by the test suite) and a CLI::

    python -m lbm.cases.poiseuille        --backend cuda-aa --plot
    python -m lbm.cases.taylor_green      --backend cuda-aa --plot
    python -m lbm.cases.lid_driven_cavity --re 1000 --n 256 --backend cuda-aa --plot

Plots are written to ``outputs/`` (not version-controlled).
"""
