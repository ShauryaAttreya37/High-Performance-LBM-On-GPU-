"""Publication figures with Makie, driven from Python.

Python runs the solver (or reuses cached results) and hands NumPy arrays to
the Makie code in ``figures/lbm_figures.jl`` through juliacall; text is set
with LaTeX math (MathTeXEngine, Computer Modern). Figures go to
``outputs/figures/`` as PNG and PDF.

Setup (once): ``uv sync --group figures``. The first run installs CairoMakie
and LaTeXStrings into juliacall's Julia environment and precompiles them,
which takes several minutes. It uses the ``julia`` on PATH when there is one.

Usage::

    uv run --group figures python figures/make_figures.py              # all figures
    uv run --group figures python figures/make_figures.py cylinder     # one figure
    uv run --group figures python figures/make_figures.py --refresh    # re-run the GPU fields

The convergence figures plot errors measured with the validation cases on
2026-10-27 (RTX 3050); the commands that reproduce each table are given next
to it. The flow fields are computed here and cached in
``outputs/figures/data/``.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "figures"
DATA = OUT / "data"
JULIA_PACKAGES = {
    "CairoMakie": "13f3f980-e62b-5c42-98c6-ff1f3baf88f0",
    "LaTeXStrings": "b964fa9f-0449-5b57-a5c2-d3ea65f4040f",
}

# ---------------------------------------------------------------- measured tables

# python -m lbm.cases.stokes_array --backend cuda-aa --sizes 32 48 64 96 128 192
# (|error| in % against Sangani & Acrivos; radius R in nodes)
CYLINDER_ARRAY = {
    "staircase": (
        [4.04, 6.06, 8.07, 12.1, 16.15, 24.2],
        [0.562, 2.654, 1.154, 1.244, 0.449, 0.315],
    ),
    "Bouzidi": ([4.04, 6.06, 8.07, 12.1, 16.15, 24.2], [0.723, 0.178, 0.101, 0.034, 0.038, 0.001]),
    "IBM": ([4.04, 6.06, 8.07, 12.1], [12.588, 7.533, 5.357, 3.384]),
}
# python -m lbm.cases.pipe_flow --backend cuda-aa --radii 4.3 6.3 8.3 12.3 16.3 24.3
PIPE = {
    "staircase": (
        [4.3, 6.3, 8.3, 12.3, 16.3, 24.3],
        [6.304e-2, 9.867e-2, 3.362e-2, 1.576e-2, 1.383e-2, 8.626e-3],
    ),
    "Bouzidi": (
        [4.3, 6.3, 8.3, 12.3, 16.3, 24.3],
        [9.658e-3, 1.025e-2, 2.497e-3, 7.107e-4, 5.875e-4, 1.873e-4],
    ),
}
# python -m lbm.cases.stokes_array --dim 3 --fraction 0.01 --backend cuda-aa --sizes 32 48 64
SPHERE_ARRAY = {
    "staircase": ([4.28, 6.42, 8.55], [3.740, 0.026, 0.955]),
    "Bouzidi": ([4.28, 6.42, 8.55], [2.626, 0.948, 0.443]),
}
# python -m lbm.cases.cylinder_benchmark --test 2D-1 --sizes 40 --u-max <u>
COMPRESSIBILITY = ([0.075, 0.0375, 0.01875], [5.694, 5.609, 5.589])
# python -m lbm.cases.beltrami --backend cuda-aa --sizes 16 24 32 48 64 96 128
ABC = (
    [16, 24, 32, 48, 64, 96, 128],
    [6.458e-3, 2.849e-3, 1.623e-3, 7.137e-4, 4.033e-4, 1.787e-4, 1.022e-4],
)
# python -m lbm.cases.duct_flow --backend cuda-aa --heights 8 16 32 64
DUCT = ([8, 16, 32, 64], [4.008e-3, 8.758e-4, 2.093e-4, 5.164e-5])


# ---------------------------------------------------------------- flow fields (GPU, cached)


def cached(name: str, compute, refresh: bool) -> dict:
    path = DATA / f"{name}.npz"
    if path.exists() and not refresh:
        return dict(np.load(path))
    DATA.mkdir(parents=True, exist_ok=True)
    data = compute()
    np.savez_compressed(path, **data)
    return data


def cylinder_fields() -> dict:
    """Schaefer-Turek 2D-2 at N = 40: final vorticity and the last 25 D/U of C_D, C_L."""
    from lbm import make_solver
    from lbm.cases.cylinder_benchmark import DIAMETER, Observer, build
    from lbm.diagnostics import vorticity

    g = build("2D-2", 40, dtype="float32")
    solver = make_solver(g.params, backend="cuda-aa")
    g.start(solver)
    observe = Observer(g)
    t_conv = g.n / g.u_mean
    every = 2 * round(t_conv / 40 / 2)
    solver.run(every * round(150 * t_conv / every), every=every, callback=observe)
    fields = solver.macroscopic()
    omega = vorticity(fields.ux, fields.uy) * g.n / g.u_mean
    omega[g.params.solid] = np.nan
    omega[:, :2] = omega[:, -2:] = np.nan  # gradients there take an inlet/outlet neighbour
    history = np.asarray(observe.history)
    t = history[:, 0] / t_conv
    tail = t > t[-1] - 25
    ny, nx = omega.shape
    dx = DIAMETER / g.n
    return {
        "x": (np.arange(nx) - 1) * dx,
        "y": (np.arange(ny) - 0.5) * dx,
        "omega": omega,
        "t": t[tail],
        "cd": history[tail, 1],
        "cl": history[tail, 2],
    }


def sphere_fields() -> dict:
    """Sphere at Re = 100, N = 16: speed and u_x in the mid-plane after 60 D/U."""
    from lbm import SimulationParams, make_solver
    from lbm.geometry import sphere, voxelize, wall_links
    from lbm.lattice import CS2, D3Q19
    from lbm.link_rules import LinkRules, bouzidi_rules
    from lbm.open_boundaries import open_channel

    n, reynolds, u = 16, 100.0, 0.08
    width, length = 10 * n, 20 * n + 3
    shape = (width, width, length)
    body = sphere(1 + 5 * n, 0.5 * (width - 1), 0.5 * (width - 1), 0.5 * n)
    solid = voxelize(body, shape)
    inflow = np.zeros((3, width, width))
    inflow[0] = u
    solid, rules = open_channel(D3Q19, solid, inflow)
    links = wall_links(D3Q19, body, solid)
    params = SimulationParams(
        nx=length,
        ny=width,
        nz=width,
        tau=u * n / reynolds / CS2 + 0.5,
        collision="trt",
        solid=solid,
        link_rules=LinkRules.concat(rules, bouzidi_rules(D3Q19, links, solid)),
    )
    solver = make_solver(params, backend="cuda-aa")
    solver.initialize(1.0, np.where(solid, 0.0, u), 0.0, 0.0)
    solver.run(2 * round(60 * n / u / 2))
    f = solver.macroscopic()
    mid = width // 2
    inside = solid[mid]
    speed = np.sqrt(f.ux[mid] ** 2 + f.uy[mid] ** 2 + f.uz[mid] ** 2) / u
    x0, x1, y0, y1 = 2 * n, 14 * n, mid - 3 * n, mid + 3 * n
    window = (slice(y0, y1), slice(x0, x1))
    return {
        "x": (np.arange(x0, x1) - body.center[0]) / n,
        "y": (np.arange(y0, y1) - body.center[1]) / n,
        "speed": np.where(inside, np.nan, speed)[window],
        "ux": np.where(inside, np.nan, f.ux[mid] / u)[window],
    }


SOCIAL = ROOT / "outputs" / "social"
LOOP_PERIODS, LOOP_FRAMES = 3, 240
"""The loop covers exactly three shedding periods, so its last frame runs
into its first; at 24 fps that is 10 s."""
LOOP_WINDOW = 1.0
"""Length [m] of channel shown, from the inlet: the cylinder and ~8 D of wake."""
# python benchmarks/bench_mlups.py (2D, float32, 2048^2, cuda-aa / cuda-naive)
HEADLINE_MLUPS, HEADLINE_BANDWIDTH = 2130, 98.5


def wake_frames() -> dict:
    """Schaefer-Turek 2D-2 at N = 80: vorticity over three shedding periods.

    After 150 D/U the shedding is periodic. The lift period is measured from
    C_L over the last quarter, then frames are taken at evenly spaced steps
    across LOOP_PERIODS periods (each rounded to the solver's step multiple,
    so the rounding never accumulates).
    """
    from lbm import make_solver
    from lbm.cases.cylinder_benchmark import DIAMETER, Observer, build, lift_period
    from lbm.diagnostics import vorticity

    g = build("2D-2", 80, dtype="float32")
    solver = make_solver(g.params, backend="cuda-aa")
    g.start(solver)
    observe = Observer(g)
    t_conv = g.n / g.u_mean
    every = 2 * round(t_conv / 40 / 2)
    solver.run(every * round(150 * t_conv / every), every=every, callback=observe)
    history = np.asarray(observe.history)
    tail = history[len(history) * 3 // 4 :]
    period = lift_period(tail[:, 0], tail[:, 2])

    dx = DIAMETER / g.n
    columns = slice(2, 2 + round(LOOP_WINDOW / dx))  # columns 0, 1 border the inlet
    m = solver.step_multiple
    frames, done = [], 0
    for k in range(LOOP_FRAMES):
        target = m * round(k * LOOP_PERIODS * period / LOOP_FRAMES / m)
        solver.run(target - done)
        done = target
        f = solver.macroscopic()
        omega = vorticity(f.ux, f.uy) * g.n / g.u_mean
        omega[g.params.solid] = np.nan
        frames.append(omega[:, columns].astype(np.float32))
    ny, nx = g.params.solid.shape
    return {
        "x": (np.arange(nx)[columns] - 1) * dx,
        "y": (np.arange(ny) - 0.5) * dx,
        "omega": np.stack(frames),
        "strouhal": g.n / (period * g.u_mean),
        "cd_max": tail[:, 1].max(),
        "grid": np.array([nx, ny]),
    }


# ---------------------------------------------------------------- Julia side


def julia():
    """juliacall's Main with lbm_figures.jl loaded (installs the Julia packages once)."""
    if "PYTHON_JULIAPKG_EXE" not in os.environ and shutil.which("julia"):
        bindir = subprocess.run(
            ["julia", "--startup-file=no", "-e", "print(Sys.BINDIR)"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        os.environ["PYTHON_JULIAPKG_EXE"] = str(Path(bindir) / "julia")
    import juliapkg

    for name, uuid in JULIA_PACKAGES.items():
        juliapkg.add(name, uuid)
    from juliacall import Main as jl

    jl.include(str(Path(__file__).with_name("lbm_figures.jl")))
    return jl


def as_vectors(table: dict) -> dict:
    return {name: (np.asarray(r, float), np.asarray(e, float)) for name, (r, e) in table.items()}


def main(argv=None) -> None:
    figures = ("curved", "compressibility", "d3q19", "cylinder", "sphere")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "which",
        nargs="*",
        metavar="FIGURE",
        help=f"any of {figures} or 'social' (hero PNG + MP4 loop); default all but social",
    )
    parser.add_argument("--refresh", action="store_true", help="recompute cached flow fields")
    args = parser.parse_args(argv)
    unknown = set(args.which) - {*figures, "social"}
    if unknown:
        parser.error(f"unknown figure(s) {sorted(unknown)}; choose from {(*figures, 'social')}")
    OUT.mkdir(parents=True, exist_ok=True)
    jl = julia()
    stem = lambda name: str(OUT / name)  # noqa: E731
    for name in args.which or figures:
        if name == "curved":
            out = jl.curved_wall_convergence(
                stem("curved_wall_convergence"),
                as_vectors(CYLINDER_ARRAY),
                as_vectors(PIPE),
                as_vectors(SPHERE_ARRAY),
            )
        elif name == "compressibility":
            u, cd = (np.asarray(v, float) for v in COMPRESSIBILITY)
            out = jl.compressibility(stem("cylinder_compressibility"), u, cd, 5.57, 5.59)
        elif name == "d3q19":
            out = jl.d3q19_validation(
                stem("d3q19_validation"), *(np.asarray(v, float) for v in (*ABC, *DUCT))
            )
        elif name == "cylinder":
            d = cached("cylinder_2d2_N40", cylinder_fields, args.refresh)
            out = jl.cylinder_wake(
                stem("cylinder_2d2_wake"),
                d["x"],
                d["y"],
                np.ascontiguousarray(d["omega"].T),
                d["t"],
                d["cd"],
                d["cl"],
                (3.22, 3.24),
                0.99,
            )
        elif name == "social":
            from lbm.cases.cylinder_benchmark import CENTRE, DIAMETER

            d = cached("wake_loop_2d2_N80", wake_frames, args.refresh)
            SOCIAL.mkdir(parents=True, exist_ok=True)
            stats = {
                "mlups": HEADLINE_MLUPS,
                "bandwidth": HEADLINE_BANDWIDTH,
                "strouhal": float(d["strouhal"]),
                "grid": tuple(int(v) for v in d["grid"]),
                "cylinder": (*CENTRE, DIAMETER / 2),  # same x, y coordinates as the map [m]
            }
            # Julia indexes each frame as omega[k, x, y].
            omega = np.ascontiguousarray(d["omega"].transpose(0, 2, 1))
            out = jl.social_figures(str(SOCIAL / "wake"), d["x"], d["y"], omega, stats)
            print(f"wrote {out}_hero.png and {out}_loop.mp4", flush=True)
            continue
        else:
            d = cached("sphere_re100_N16", sphere_fields, args.refresh)
            out = jl.sphere_wake(
                stem("sphere_re100_wake"),
                d["x"],
                d["y"],
                np.ascontiguousarray(d["speed"].T),
                np.ascontiguousarray(d["ux"].T),
            )
        print(f"wrote {out}.png and {out}.pdf", flush=True)


if __name__ == "__main__":
    main()
