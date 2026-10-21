# High-Performance Lattice Boltzmann on the GPU

A 2D lattice Boltzmann (D2Q9) fluid solver written from scratch: a readable
NumPy reference implementation, plus hand-written CUDA C++ kernels compiled at
runtime and driven from Python through CuPy. No CFD frameworks, no wrappers
around existing solvers.

The long-term goal is drag prediction around arbitrary bodies with an
immersed boundary method, validated against wind-tunnel data for
10² ≤ Re ≤ 10⁵. This repository is being built in stages; **stage 1, the
validated and benchmarked core solver, is complete.**

## Highlights (stage 1)

* **98.5 % of measured memory bandwidth.** The fused kernels run at
  ~2 130 million lattice updates per second (MLUPS) on an RTX 3050 6 GB
  Laptop GPU: 153.8 GB/s effective vs a 156.2 GB/s device-copy peak.
  That is 37× the same algorithm written as unfused CuPy array code.
* **Three kernel strategies, measured against each other:** a fused
  two-lattice kernel, a shared-memory tiled kernel, and the in-place AA
  pattern (half the memory). The tiled kernel, a classic 2008-era
  optimisation, is *slower* on modern caches (84 % vs 96 % of peak at 4096²).
* **GPU results identical to the CPU reference to a few ulps** (max population
  difference ~1e-17 in float64, ~1e-8 in float32) on a geometry with walls, a
  moving lid, an obstacle and a body force.
* **Exact where theory says exact.** Poiseuille flow with TRT (Λ = 3/16)
  matches the analytical profile to 3e-13 for every relaxation time. BGK's
  wall slip follows the closed form u_slip = (2/3)(Λ − 3/16)·F/(ρν) to
  machine precision.
* **Second-order convergence** (Taylor–Green vortex: observed order 2.00) and
  agreement with Ghia et al. (1982) lid-driven cavity data to 0.5 % (Re = 100)
  and 0.9 % (Re = 1000).

## Method

| Component | Choice |
|-----------|--------|
| Lattice | D2Q9, populations stored shifted (f − w) for float32 accuracy |
| Collision | BGK and TRT (two-relaxation-time, magic parameter Λ) in one operator |
| Forcing | Guo et al. (2002), split into symmetric/antisymmetric parts for TRT |
| Streaming | pull scheme, periodic by default |
| Walls | halfway bounce-back on solid nodes, Ladd moving-wall correction |
| Viscosity | ν = c_s² (τ − ½), c_s² = 1/3 (lattice units) |
| Safety | τ ≤ ½ rejected, τ < 0.51 and Ma > 0.17 warned, divergence watchdog |

One time step, identical in every backend:

1. pull-stream `f_i(x) = f*_i(x − c_i)`
2. bounce-back where `x − c_i` is solid
3. moments `ρ = 1 + Σ f̃_i`, `ρu = Σ c_i f_i + F/2`
4. TRT/BGK collision with the Guo source, stored at fluid nodes

## Backends

| Backend | What it is | float32 MLUPS (2048²) |
|---------|------------|------------------------|
| `numpy` | CPU reference; the correctness oracle and what CI runs | ~6 (1024²) |
| `cupy` | the same reference code on GPU arrays (unfused, ~100 kernels/step) | 57 |
| `cuda-naive` | one fused stream-bounce-collide kernel, two lattices | 2 137 |
| `cuda-tiled` | as above, neighbours staged through shared memory | 2 015 |
| `cuda-aa` | AA access pattern: one lattice updated in place | 2 123 |

The kernels live in `src/lbm/backends/kernels/`. Lattice constants are
generated into each kernel from the Python definitions as `constexpr`
functions, so every per-thread array stays in registers (verified: 0 bytes of
local memory in all kernels).

In float64 the same kernels reach ~280–310 MLUPS: on this consumer GPU FP64
runs at 1/64 of the FP32 rate, so double precision turns LBM from
memory-bound to compute-bound. Validation results are identical in float32
and float64 to four digits, so float32 is the production precision.

## Validation

| Case | Checks | Result |
|------|--------|--------|
| Poiseuille flow | walls, forcing, wall placement vs τ | TRT exact to 3e-13; BGK slip law exact; order 2.000 |
| Taylor–Green vortex | viscosity, bulk accuracy | decay-rate error 0.4 % (N = 32); order 2.00 |
| Lid-driven cavity, Re = 100 | moving walls, mass conservation | max centreline error 0.0054 U (n = 64) |
| Lid-driven cavity, Re = 1000 | stability near τ → ½ | max centreline error 0.0086 U (n = 256) |
| Backend equivalence | all GPU kernels vs NumPy | ~1e-17 (float64), ~1e-8 (float32) |

At Re = 1000 on coarse grids plain BGK either diverges (τ ≈ 0.51) or never
settles to the steady state, while TRT converges. That is the τ → ½ failure
mode the later high-Reynolds stage has to address.

## Getting started

Requirements: Python ≥ 3.12, [uv](https://docs.astral.sh/uv/). For the GPU
backends: an NVIDIA GPU and driver supporting CUDA 13. No system CUDA toolkit
is needed; NVRTC is installed from PyPI.

```bash
uv sync --extra gpu        # or plain `uv sync` for CPU only
uv run pytest -m "not slow"
```

Run the validation cases (figures are written to `outputs/`):

```bash
uv run python -m lbm.cases.poiseuille --plot
uv run python -m lbm.cases.taylor_green --plot
uv run python -m lbm.cases.lid_driven_cavity --re 1000 --n 256 --backend cuda-aa --dtype float32 --plot
```

Benchmark every backend on your GPU:

```bash
uv run python benchmarks/bench_mlups.py --plot
```

Use the solver directly:

```python
from lbm import SimulationParams, make_solver
from lbm.boundaries import cavity_walls

solid, wall = cavity_walls(n=256, lid_velocity=0.1)
params = SimulationParams(
    nx=258, ny=258, tau=0.5768, collision="trt",
    solid=solid, wall_velocity=wall, dtype="float32",
)
solver = make_solver(params, backend="cuda-aa")
solver.run(200_000, every=10_000, callback=lambda s: print(s.time_step))
fields = solver.macroscopic()     # rho, ux, uy as NumPy arrays
```

## Tests

```bash
uv run pytest                 # everything, including slow GPU runs
uv run pytest -m "not slow"   # fast subset (~30 s with a GPU)
uv run pytest -m "not gpu"    # what a CPU-only machine can run
```

Tests marked `gpu` skip automatically when CuPy or a CUDA device is missing.
Every tolerance was set from a measured error with a stated margin.

## Project layout

```
src/lbm/
  lattice.py         D2Q9 constants; generates the CUDA preamble
  params.py          SimulationParams: validated, immutable run description
  units.py           physical <-> lattice unit conversion
  equilibrium.py     shifted second-order equilibrium
  collision.py       TRT/BGK collision
  forcing.py         Guo forcing
  streaming.py       pull streaming
  boundaries.py      bounce-back links, channel and cavity geometries
  diagnostics.py     macroscopic fields, vorticity, energy
  solver.py          Solver base class (run loop, watchdog), ReferenceSolver
  backends/          backend selection, CudaSolver, CUDA kernels
  cases/             Poiseuille, Taylor-Green, lid-driven cavity, Ghia data
benchmarks/          MLUPS and bandwidth benchmark
tests/               unit tests, validation tests, GPU equivalence tests
```

## Roadmap

1. ~~Core D2Q9 solver, NumPy reference + CUDA kernels, validated and benchmarked~~
2. Curved boundaries: interpolated bounce-back, immersed boundary method,
   momentum-exchange drag; cylinder C_d and Strouhal number for Re ≤ 200
3. High Reynolds numbers: regularised/cumulant collision and LES;
   Re = 10³–10⁵ drag sweep, with the 2D-vs-3D gap reported honestly
4. Real-time visualisation with stream tracers
5. Automated CI validation pipeline
6. D3Q19 in 3D

## References

* T. Krüger et al., *The Lattice Boltzmann Method: Principles and Practice*, Springer (2017).
* I. Ginzburg, F. Verhaeghe, D. d'Humières, Two-relaxation-time lattice Boltzmann
  scheme, Commun. Comput. Phys. 3 (2008) 427.
* Z. Guo, C. Zheng, B. Shi, Discrete lattice effects on the forcing term in the
  lattice Boltzmann method, Phys. Rev. E 65 (2002) 046308.
* A. J. C. Ladd, Numerical simulations of particulate suspensions via a
  discretized Boltzmann equation, J. Fluid Mech. 271 (1994) 285.
* U. Ghia, K. N. Ghia, C. T. Shin, High-Re solutions for incompressible flow
  using the Navier-Stokes equations and a multigrid method, J. Comput. Phys. 48
  (1982) 387.
* P. Bailey et al., Accelerating lattice Boltzmann fluid flow simulations using
  graphics processors, ICPP (2009).
* J. Tölke, Implementation of a lattice Boltzmann kernel using CUDA, Comput.
  Visual. Sci. 13 (2010) 29.
