"""Validation cases with known answers.

Each module exposes a ``run_*`` function (used by the test suite) and a CLI::

    python -m lbm.cases.poiseuille        --backend cuda-aa --plot
    python -m lbm.cases.taylor_green      --backend cuda-aa --plot
    python -m lbm.cases.lid_driven_cavity --re 1000 --n 256 --backend cuda-aa --plot
    python -m lbm.cases.beltrami          --backend cuda-aa --sizes 16 32 64   (3D)
    python -m lbm.cases.duct_flow         --backend cuda-aa --heights 8 16 32  (3D)
    python -m lbm.cases.stokes_array      --backend cuda-aa --dim 2           (curved bodies)
    python -m lbm.cases.pipe_flow         --backend cuda-aa                   (3D, curved wall)
    python -m lbm.cases.cylinder_benchmark --test 2D-1 --backend cuda-aa      (Schaefer-Turek)
    python -m lbm.cases.sphere_drag       --backend cuda-aa                   (3D)

Plots are written to ``outputs/`` (not version-controlled).
"""
