# High-Performance Lattice Boltzmann on the GPU

A lattice Boltzmann fluid solver for 2D (D2Q9) and 3D (D3Q19), written from
scratch: a readable NumPy reference implementation, plus hand-written CUDA C++
kernels compiled at runtime and driven from Python through CuPy. One kernel
source serves both dimensions. No CFD frameworks, no wrappers around existing
solvers.

The long-term goal is drag prediction around arbitrary bodies with an
immersed boundary method, validated against wind-tunnel data for
10² ≤ Re ≤ 10⁵. This repository is being built in stages; **stage 1 (the
validated and benchmarked 2D core), stage 1b (one dimension-generic engine
for 2D and 3D) and stage 2 (curved bodies, open boundaries, forces) are
complete.**

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

## Highlights (stage 1b: 3D)

* **~1 000 MLUPS in 3D, 93–99.9 % of measured memory bandwidth** (D3Q19,
  float32, 64³–256³, two-lattice and AA kernels). 2D did not slow down:
  98.7 % of peak at 2048², as before.
* **One CUDA source for both dimensions.** A 2D lattice is compiled as a 3D
  one with nz = 1 and c_z ≡ 0. The z terms are compile-time zeros, so the 2D
  kernels compile to the same code as before. D3Q19 in double precision still
  keeps all 19 populations in registers (0 B local memory).
* **3D GPU results match the NumPy reference to ~5e-18** (float64) on a
  non-cubic box with walls, a moving plate, an obstacle and a 3D body force.
* **Exact 3D validation.** The decaying ABC (Beltrami) flow, an exact 3D
  Navier–Stokes solution, converges at order 2.0 from 16³ to 128³. Square-duct
  flow against its Fourier-series solution converges at order 2.0 with walls
  in two directions. The 3D plate channel with TRT is exact to 3e-13, as in 2D.

## Highlights (stage 2: curved bodies)

* **Schäfer–Turek cylinder benchmark inside the published intervals.**
  2D-1 (Re = 20, N = 40): C_D = 5.589, C_L = 0.01058. 2D-2 (Re = 100,
  N = 80): C_D,max = 3.238, C_L,max = 0.992, St = 0.302.
* **Two body representations, compared on the same cases.** Interpolated
  bounce-back (Bouzidi) converges at second order to the Stokes drag of a
  cylinder array (−0.72 % at R = 4, +0.001 % at R = 24). The immersed
  boundary method (direct forcing, multi-direct iterations) converges at
  first order (+12.6 % at R = 4, +3.4 % at R = 12). The plain staircase wall
  wanders between +0.3 % and +2.7 %, depending on how the circle cuts the grid.
* **One mechanism, no kernel changes.** Bouzidi, the Zou–He velocity inlet
  and pressure outlet, and the IBM's velocity prediction are all *affine
  link rules* over post-collision populations. They are compiled per memory
  layout (including both halves of the in-place AA scheme), and a sparse
  pre-pass applies them before each fused kernel launch. GPU = NumPy
  reference to 1e-17.
* **Bodies cost little.** A Bouzidi cylinder plus Zou–He inlet and outlet
  keeps 96 % of the empty-domain throughput in 2D (2 035 vs 2 122 MLUPS at
  1024²), 77–83 % in 3D. The IBM pre-pass is still plain CuPy array code and
  costs two thirds of the throughput; fusing it is future work.
* **Momentum-exchange forces** as a precompiled linear functional of the
  state: one sparse gather per evaluation. In steady periodic flow it
  matches the driving force to 1e-9.
* **Two silent errors found and explained.** First, a 2 % drag excess that
  did not shrink with resolution was weak compressibility (an O(Ma²) density
  rise upstream). Second, an exponential mass blow-up after 140k steps came
  from treating Zou–He corner nodes inconsistently.

## Method

| Component | Choice |
|-----------|--------|
| Lattice | D2Q9 and D3Q19, populations stored shifted (f − w) for float32 accuracy |
| Collision | BGK and TRT (two-relaxation-time, magic parameter Λ) in one operator |
| Forcing | Guo et al. (2002), split into symmetric/antisymmetric parts for TRT |
| Streaming | pull scheme, periodic by default |
| Walls | halfway bounce-back on solid nodes, Ladd moving-wall correction |
| Curved bodies | Bouzidi linear interpolated bounce-back, or direct-forcing IBM (Roma kernel) |
| Open boundaries | Zou–He velocity inlet and pressure outlet (Hecht–Harting form for D3Q19) |
| Forces | momentum exchange (Bouzidi/staircase), marker forces (IBM) |
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
| `cuda-tiled` | as above, neighbours staged through shared memory (2D only) | 2 015 |
| `cuda-aa` | AA access pattern: one lattice updated in place | 2 123 |

In 3D (D3Q19, float32), `cuda-naive` runs at 1 015 MLUPS and `cuda-aa` at
1 000 MLUPS at 128³, which is 97–99 % of the copy peak. Each update moves
152 bytes. The tiled variant stays 2D only: a 3D halo tile would load about
4× the useful data, and the variant already loses in 2D.

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
| 3D ABC (Beltrami) flow | D3Q19 bulk scheme, viscosity | order 2.0 (16³–128³); decay-rate error 0.15 % (32³) |
| 3D plate channel | D3Q19 walls and forcing | TRT exact to 3e-13; BGK slip equal to 2D |
| 3D square duct | walls in two directions | order 2.0 vs the Fourier series (H = 8–64) |
| 3D backend equivalence | naive and AA kernels vs NumPy | ~5e-18 (float64), ~3e-9 (float32) |
| Stokes cylinder array | curved-wall accuracy | Bouzidi −0.10 % (R = 8), IBM +5.4 %, staircase +1.2 % |
| Stokes sphere array (3D) | curved walls in 3D | Bouzidi −0.44 % at R = 8.6 vs Hasimoto |
| Pipe flow (3D) | curved wall, exact solution | Bouzidi 7–45× smaller error than staircase, order ≈ 2.3 |
| Schäfer–Turek 2D-1 / 2D-2 | inlet/outlet, forces, shedding | inside the benchmark intervals (see above) |
| Sphere, Re = 20 / 100 (3D) | 3D bodies in open flow | Bouzidi 2.77 / 1.13 vs Clift–Grace–Weber 2.74 / 1.09 |

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
uv run python -m lbm.cases.beltrami --backend cuda-aa --sizes 16 32 64 128
uv run python -m lbm.cases.duct_flow --backend cuda-aa --heights 8 16 32 64
uv run python -m lbm.cases.stokes_array --backend cuda-aa --sizes 32 64 96
uv run python -m lbm.cases.cylinder_benchmark --test 2D-1 --backend cuda-aa --sizes 20 40
uv run python -m lbm.cases.cylinder_benchmark --test 2D-2 --backend cuda-aa --dtype float32 --sizes 80 --u-max 0.025
uv run python -m lbm.cases.sphere_drag --backend cuda-aa
```

Benchmark every backend on your GPU:

```bash
uv run python benchmarks/bench_mlups.py --plot
uv run python benchmarks/bench_mlups.py --dim 3 --backends cuda-naive cuda-aa --plot
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

3D is the same API with `nz` set: arrays are indexed `[z, y, x]`, the force
has three components, and `wall_velocity` has shape `(3, nz, ny, nx)`:

```python
solid, wall = cavity_walls(n=128, lid_velocity=0.1, dimensions=3)
params = SimulationParams(nx=130, ny=130, nz=130, tau=0.56, collision="trt",
                          solid=solid, wall_velocity=wall, dtype="float32")
solver = make_solver(params, backend="cuda-aa")   # D3Q19 chosen automatically
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
  lattice.py         VelocitySet (D2Q9, D3Q19); generates the CUDA preamble
  params.py          SimulationParams: validated, immutable run description
  units.py           physical <-> lattice unit conversion
  equilibrium.py     shifted second-order equilibrium
  collision.py       TRT/BGK collision
  forcing.py         Guo forcing
  streaming.py       pull streaming
  boundaries.py      bounce-back links, channel and cavity geometries
  diagnostics.py     macroscopic fields, vorticity, energy
  geometry.py        shapes (circle, sphere, pipe), voxelisation, wall-link fractions q
  link_rules.py      affine boundary rules, compiled per memory layout; Bouzidi
  open_boundaries.py Zou-He velocity inlet and pressure outlet
  forces.py          momentum-exchange force probe
  ibm.py             immersed boundary method (direct forcing)
  solver.py          Solver base class (run loop, watchdog), ReferenceSolver
  backends/          backend selection, CudaSolver, CUDA kernels
  cases/             Poiseuille (2D/3D), Taylor-Green, ABC flow, duct flow,
                     lid-driven cavity, Ghia data, Stokes arrays, pipe flow,
                     Schaefer-Turek cylinder, sphere drag
benchmarks/          MLUPS and bandwidth benchmark
tests/               unit tests, validation tests, GPU equivalence tests
```

## Roadmap

1. ~~Core D2Q9 solver, NumPy reference + CUDA kernels, validated and benchmarked~~
1b. ~~Dimension-generic engine: D2Q9 and D3Q19 through one API and one kernel source~~
2. ~~Curved boundaries: interpolated bounce-back, immersed boundary method,
   momentum-exchange drag; cylinder C_d and Strouhal number for Re ≤ 200~~
3. High Reynolds numbers: regularised/cumulant collision and LES;
   Re = 10³–10⁵ drag sweep, with the 2D-vs-3D gap reported honestly
4. Real-time visualisation with stream tracers
5. Automated CI validation pipeline
6. 3D applications on the D3Q19 engine (3D cavity vs literature, 3D bodies)

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
* M. Bouzidi, M. Firdaouss, P. Lallemand, Momentum transfer of a Boltzmann-lattice
  fluid with boundaries, Phys. Fluids 13 (2001) 3452.
* Q. Zou, X. He, On pressure and velocity boundary conditions for the lattice
  Boltzmann BGK model, Phys. Fluids 9 (1997) 1591.
* M. Hecht, J. Harting, Implementation of on-site velocity boundary conditions
  for D3Q19 lattice Boltzmann simulations, J. Stat. Mech. (2010) P01018.
* M. Schäfer, S. Turek, Benchmark computations of laminar flow around a
  cylinder, Notes Numer. Fluid Mech. 52 (1996) 547.
* A. M. Roma, C. S. Peskin, M. J. Berger, An adaptive version of the immersed
  boundary method, J. Comput. Phys. 153 (1999) 509.
* Z. Wang, J. Fan, K. Luo, Combined multi-direct forcing and immersed boundary
  method for simulating flows with moving particles, Int. J. Multiphase Flow 34
  (2008) 283.
* A. S. Sangani, A. Acrivos, Slow flow past periodic arrays of cylinders with
  application to heat transfer, Int. J. Multiphase Flow 8 (1982) 193.
* H. Hasimoto, On the periodic fundamental solutions of the Stokes equations and
  their application to viscous flow past a cubic array of spheres, J. Fluid
  Mech. 5 (1959) 317.
