try; import KaimonSlate; catch; error("This is a Kaimon Slate notebook — running it as plain Julia needs the KaimonSlate runtime in this environment. Add it with `import Pkg; Pkg.add(\"KaimonSlate\")`, or open it in Kaimon Slate."); end; KaimonSlate.standalone!(@__MODULE__; dir=@__DIR__)

#%% md id=cavity_intro
@md raw"""
## Lid-driven cavity on the GPU (`cuda-aa`)

A square box of side **n** fluid nodes. The top wall (the *lid*) slides right at speed **U**; the other three walls stay still. Moving-wall bounce-back drags the fluid along, and a big primary vortex forms, with small counter-rotating eddies in the corners.

The Reynolds number fixes the viscosity: $\nu = U n / \mathrm{Re}$, and LBM relates viscosity to the relaxation time by $\tau = 3\nu + \tfrac12$.

We run until the flow stops changing, then compare velocity along the two centrelines with **Ghia, Ghia & Shin (1982)**, the standard reference solution for this problem.

The Python `lbm` package does the physics on the GPU. Julia drives the run, checks convergence, measures speed and makes the plots.
"""

#%% code id=setup
# The physics lives in the project's Python package `lbm`; PythonCall calls it from Julia.
# Two settings make this work (see notebook footer + the env's LocalPreferences.toml):
#   threads = 1,0  — PythonCall crashes if Python is called off Julia's main thread
#   PythonCall exe = <project>/.venv/bin/python  — use the project venv, not a Conda Python
using PythonCall

lbm = pyimport("lbm")
pyconvert(Vector{String}, lbm.available_backends())

#%% code id=cavity_params
# Physical setup. Re = 1000 is the hardest Ghia case: thin shear layers, tau close to 1/2.
const REYNOLDS     = 1000
const N_CAVITY     = 256      # fluid nodes per side
const LID_VELOCITY = 0.1      # lattice units; Mach = U*sqrt(3) ≈ 0.17
const DTYPE        = "float32"  # RTX 3050 runs float64 at 1/64 the float32 rate

# Steady when no velocity component (in units of U) moves more than STEADY_TOL
# over one lid-transit time n/U. Rounded to an even step count: cuda-aa advances in pairs.
const STEADY_TOL    = 1e-6
const MAX_STEPS     = 2_000_000
const CHECK_INTERVAL = 2 * round(Int, N_CAVITY / LID_VELOCITY / 2)

viscosity = LID_VELOCITY * N_CAVITY / REYNOLDS
tau = 3viscosity + 0.5
(; viscosity, tau, check_interval = CHECK_INTERVAL, nodes = (N_CAVITY + 2)^2)

#%% code id=cavity_solver
const lbm_boundaries = pyimport("lbm.boundaries")

# cavity_walls gives an (n+2)×(n+2) grid: a one-node solid frame, lid on top moving in +x.
solid, wall_velocity = lbm_boundaries.cavity_walls(N_CAVITY, LID_VELOCITY)

cavity_sim_params = lbm.SimulationParams(
    nx = N_CAVITY + 2, ny = N_CAVITY + 2,
    tau = tau, collision = "trt",
    solid = solid, wall_velocity = wall_velocity,
    dtype = DTYPE,
)
cavity = lbm.make_solver(cavity_sim_params, backend = "cuda-aa")
cavity.initialize()

(backend = "cuda-aa",
 grid = pyconvert(Tuple{Int,Int}, cavity_sim_params.shape),
 lattice_MiB = pyconvert(Int, cavity.lattice_bytes) / 2^20,
 step_multiple = pyconvert(Int, cavity.step_multiple))

#%% code id=cavity_run
# Host snapshot of (ux, uy)/U as Julia matrices indexed [y, x] (same order as NumPy).
function velocity_snapshot(solver)
    fields = solver.macroscopic()
    ux = pyconvert(Matrix{Float64}, fields.ux) ./ LID_VELOCITY
    uy = pyconvert(Matrix{Float64}, fields.uy) ./ LID_VELOCITY
    return ux, uy
end

# Advance one lid-transit time at a time until the velocity field stops changing.
# Records the change per check so we can plot convergence afterwards.
function run_to_steady!(solver; interval, tol, max_steps)
    history = (step = Int[], change = Float64[], seconds = Float64[])
    previous = velocity_snapshot(solver)
    gpu_seconds = 0.0
    while pyconvert(Int, solver.time_step) < max_steps
        # macroscopic() copies to host, which waits for the GPU, so the timing is honest.
        gpu_seconds += @elapsed solver.run(interval, check_every = interval)
        current = velocity_snapshot(solver)
        change = max(maximum(abs.(current[1] .- previous[1])), maximum(abs.(current[2] .- previous[2])))
        push!(history.step, pyconvert(Int, solver.time_step))
        push!(history.change, change)
        push!(history.seconds, gpu_seconds)
        change < tol && return (; history, converged = true)
        previous = current
    end
    return (; history, converged = false)
end

cavity.initialize()  # rerunning this cell restarts from rest
cavity_wall_time = @elapsed cavity_run_result = run_to_steady!(cavity;
    interval = CHECK_INTERVAL, tol = STEADY_TOL, max_steps = MAX_STEPS)

steps_taken = last(cavity_run_result.history.step)
gpu_time    = last(cavity_run_result.history.seconds)
(converged = cavity_run_result.converged,
 steps = steps_taken,
 final_change = last(cavity_run_result.history.change),
 gpu_seconds = round(gpu_time, digits = 2),
 wall_seconds = round(cavity_wall_time, digits = 2),
 MLUPS = round((N_CAVITY + 2)^2 * steps_taken / gpu_time / 1e6, digits = 1))

#%% code id=cavity_ghia
ghia = pyimport("lbm.cases.ghia1982").ghia_centerlines(REYNOLDS)
ghia_y, ghia_u = pyconvert(Vector{Float64}, ghia.y), pyconvert(Vector{Float64}, ghia.u)
ghia_x, ghia_v = pyconvert(Vector{Float64}, ghia.x), pyconvert(Vector{Float64}, ghia.v)

# Drop the solid frame; what's left is the n×n fluid region, indexed [y, x].
ux_steady, uy_steady = map(a -> a[2:end-1, 2:end-1], velocity_snapshot(cavity))

# Value on the middle line along `dim` (n even → average of the two middle lines).
function centreline(field, dim)
    n = size(field, dim)
    mid(i) = vec(selectdim(field, dim, i))
    isodd(n) ? mid(n ÷ 2 + 1) : (mid(n ÷ 2) .+ mid(n ÷ 2 + 1)) ./ 2
end

# Node centres sit at (i - 1/2)/n; add the walls, where the no-slip / lid values are known.
cell_centres = ((1:N_CAVITY) .- 0.5) ./ N_CAVITY
line_y = [0.0; cell_centres; 1.0]
line_u = [0.0; centreline(ux_steady, 2); 1.0]   # u along x = 0.5 (fix x, vary y)
line_x = [0.0; cell_centres; 1.0]
line_v = [0.0; centreline(uy_steady, 1); 0.0]   # v along y = 0.5 (fix y, vary x)

# Piecewise-linear interpolation onto Ghia's sample points.
function interp(xs, ys, q)
    i = clamp(searchsortedlast(xs, q), 1, length(xs) - 1)
    t = (q - xs[i]) / (xs[i+1] - xs[i])
    return (1 - t) * ys[i] + t * ys[i+1]
end
u_error = [interp(line_y, line_u, y) for y in ghia_y] .- ghia_u
v_error = [interp(line_x, line_v, x) for x in ghia_x] .- ghia_v

(max_abs_u_error = maximum(abs, u_error),
 max_abs_v_error = maximum(abs, v_error),
 rms_u_error = sqrt(sum(abs2, u_error) / length(u_error)),
 rms_v_error = sqrt(sum(abs2, v_error) / length(v_error)))

#%% code id=cavity_plot_u
# u along the vertical centreline, drawn the way Ghia does: u on the horizontal axis, y up.
echart(
    series(:line, line_u, line_y; name = "LBM cuda-aa, n = $N_CAVITY", showSymbol = false),
    series(:scatter, ghia_u, ghia_y; name = "Ghia et al. 1982", symbolSize = 8);
    title = "u / U on x = 0.5, Re = $REYNOLDS",
    xAxis = (type = :value, name = "u / U", min = -0.5, max = 1.0),
    yAxis = (type = :value, name = "y", min = 0, max = 1),
    height = 420,
)

#%% code id=cavity_plot_v
echart(
    series(:line, line_x, line_v; name = "LBM cuda-aa, n = $N_CAVITY", showSymbol = false),
    series(:scatter, ghia_x, ghia_v; name = "Ghia et al. 1982", symbolSize = 8);
    title = "v / U on y = 0.5, Re = $REYNOLDS",
    xAxis = (type = :value, name = "x", min = 0, max = 1),
    yAxis = (type = :value, name = "v / U"),
    height = 420,
)

#%% code id=cavity_vorticity
# ω_z = ∂v/∂x − ∂u/∂y by central differences, in units of U/L (grid spacing 1/n).
# Fields are [y, x]; the result loses one node on each edge.
function vorticity(ux, uy, n)
    dvdx = (uy[2:end-1, 3:end] .- uy[2:end-1, 1:end-2]) ./ 2
    dudy = (ux[3:end, 2:end-1] .- ux[1:end-2, 2:end-1]) ./ 2
    return (dvdx .- dudy) .* n
end

omega_steady = vorticity(ux_steady, uy_steady, N_CAVITY)

# Average k×k blocks so the browser draws fewer cells (k = 2: ~16k instead of ~65k).
block_mean(a, k) = [sum(@view a[(i-1)*k+1:i*k, (j-1)*k+1:j*k]) / k^2
                    for i in 1:size(a, 1) ÷ k, j in 1:size(a, 2) ÷ k]

# Heatmap of an [y, x] matrix with y pointing up (ECharts category y axes grow upward).
# Colours clip at ±clip: the lid corners are singular and would wash out everything else.
const OMEGA_CLIP = 5.0
function vorticity_heatmap(omega; title, block = 2, clip = OMEGA_CLIP)
    w = block_mean(omega, block)
    ny, nx = size(w)
    cells = [[j - 1, i - 1, round(w[i, j], digits = 3)] for i in 1:ny for j in 1:nx]
    echart(; title = title,
        xAxis = (type = :category, data = round.(((1:nx) .- 0.5) ./ nx, digits = 2), name = "x"),
        yAxis = (type = :category, data = round.(((1:ny) .- 0.5) ./ ny, digits = 2), name = "y"),
        visualMap = (min = -clip, max = clip, calculable = true, orient = :vertical,
                     right = 0, top = :center,
                     inRange = (color = ["#2166ac", "#67a9cf", "#f7f7f7", "#ef8a62", "#b2182b"],)),
        grid = (left = 50, right = 90, top = 50, bottom = 40),
        series = [(type = :heatmap, data = cells, progressive = 0)],
        height = 520)
end

vorticity_heatmap(omega_steady; title = "Vorticity ω (U/L), Re = $REYNOLDS, steady state")

#%% code id=cavity_convergence
# How fast the flow settled: largest velocity change per lid-transit, on a log scale.
echart(:line, cavity_run_result.history.step, cavity_run_result.history.change;
    name = "max |Δu| / U per check", showSymbol = false,
    title = "Convergence to steady state ($(round(gpu_time, digits = 1)) s on GPU)",
    xAxis = (type = :value, name = "time step"),
    yAxis = (type = :log, name = "max |Δu| / U"),
    markLine = (data = [(yAxis = STEADY_TOL, name = "tol")], symbol = :none,
                label = (formatter = "tol = $STEADY_TOL",)),
    height = 380)

#%% code id=spinup_frames
# Spin-up movie: start a fresh solver from rest and grab the vorticity every half lid-transit.
# A separate solver, so the steady-state `cavity` above is left untouched.
const FRAME_EVERY = CHECK_INTERVAL ÷ 2      # 1280 steps = half a lid-transit (even, for cuda-aa)
const N_FRAMES    = 200                     # 256k steps: covers the whole approach to steady state

spinup = lbm.make_solver(cavity_sim_params, backend = "cuda-aa")
spinup.initialize()

function capture_vorticity_frames(solver; every, nframes)
    frames, steps = Matrix{Float32}[], Int[]
    for _ in 1:nframes
        solver.run(every, check_every = every)
        ux, uy = map(a -> a[2:end-1, 2:end-1], velocity_snapshot(solver))
        omega = block_mean(vorticity(ux, uy, N_CAVITY), 2)       # 127×127, keeps the movie small
        push!(frames, Float32.(reverse(clamp.(omega, -OMEGA_CLIP, OMEGA_CLIP), dims = 1)))  # lid on top
        push!(steps, pyconvert(Int, solver.time_step))
    end
    return frames, steps
end

spinup_seconds = @elapsed spinup_frames, spinup_steps = capture_vorticity_frames(spinup;
    every = FRAME_EVERY, nframes = N_FRAMES)

(frames = length(spinup_frames), frame_size = size(first(spinup_frames)),
 last_step = last(spinup_steps), seconds = round(spinup_seconds, digits = 1))

#%% code id=spinup_movie
# Plays in the browser (WebGL); Julia isn't involved during playback.
# Frames are pre-clipped to ±OMEGA_CLIP, so :symmetric gives a fixed ±5 diverging scale.
animate(spinup_frames;
    fps = 20, clim = :symmetric,
    x = range(0, 1, length = size(first(spinup_frames), 2)),
    y = range(1, 0, length = size(first(spinup_frames), 1)),   # row 1 is the lid
    title = "Vorticity spin-up, Re = $REYNOLDS, one frame = $FRAME_EVERY steps (half a lid-transit)",
    autoplay = true, height = 520)

#%% md id=spinup_notes
@md"""
**What to watch in the movie** (blue = clockwise, red = anticlockwise; colours clip at ±5 U/L):

1. **First ~10 frames (≤ 13k steps):** the moving lid sheds strong clockwise vorticity that rolls up near the top-right corner. The strongest blue spot moves down and left, from (0.85, 0.85) at frame 1 to (0.70, 0.69) at frame 10.
2. **Frames 20 → 80 (25k → 100k steps):** red, anticlockwise eddies grow in both bottom corners. Their peak vorticity rises from ≈ 0.15 to ≈ 1.06 U/L, then stays there.
3. **After frame ~80:** the picture is essentially frozen. The largest frame-to-frame change falls from 6 U/L early on to 0.005 at frame 80 and 1e-4 by frame 160. This is the steady state compared with Ghia above.

The capture took ~13 s for 256k GPU steps plus 200 vorticity snapshots.

*If the player shows "0 / 0", the browser tab was in the background while the cell rendered. Bring the tab to the front and it starts.*
"""

#%% md id=cavity_summary
@md"""
### What the cavity run shows

- **Accuracy.** Both centreline profiles land on Ghia's points. The largest deviation is under 0.9 % of the lid speed (see `cavity_ghia`), and that's in float32 at τ ≈ 0.577, close to the stability edge at ½.
- **Physics.** The vorticity map shows the primary vortex (ω ≈ −2 U/L at the centre, turning clockwise), the thin shear layer under the lid, and counter-rotating eddies (ω > 0) in both bottom corners, which are expected at Re = 1000.
- **Speed.** About 1.5 billion lattice updates per second on a laptop RTX 3050 with the AA-pattern kernel (one in-place population buffer). Reaching steady state takes about 312k steps and about 14 s.
- **Convergence.** The change per lid-transit falls by about 6 orders of magnitude, from ~1 to below 1e-6. It isn't monotone: 30 of 121 checks went up while the vortex was still oscillating into place. float32 is good enough to reach a 1e-6 tolerance.
"""

#%% md id=d3_intro
@md"""
## 3D engine: D3Q19 in Julia + CUDA.jl

The 2D GPU kernels above are CUDA C called from Python. For 3D, the GPU kernels are written **in Julia** with CUDA.jl, which compiles Julia functions straight to GPU code (no nvcc needed). The Python package stays the **reference**: its NumPy D3Q19 solver checks the Julia engine number by number.

**The scheme is the same as the 2D engine, one dimension up:**

- **D3Q19 lattice:** 19 velocities per node (rest, 6 faces, 12 edges). Weights 1/3, 1/18, 1/36.
- **Shifted storage:** we store f̃ᵢ = fᵢ − wᵢ, so float32 doesn't waste digits on the large constant part.
- **TRT collision with Guo forcing:** BGK is the special case ω⁻ = ω⁺.
- **Halfway bounce-back on solid nodes:** includes a moving-wall term 6 wᵢ (cᵢ·u_w), which drives the lid.
- **AA pattern:** one in-place population array, updated by two alternating kernels.
  - *Even step:* collide in place and write to the reversed slots.
  - *Odd step:* pull from the neighbours, collide, push to the neighbours.
  - Between step pairs the array holds post-collision values in reversed slots: `F[x,y,z,opp(i)] = f*ᵢ`.

**Memory layout:** `F[x, y, z, i]`, a Julia column-major array, so x is fastest in memory. Neighbouring GPU threads take neighbouring x, so their loads and stores coalesce. As raw bytes this is identical to NumPy's `f[i, z, y, x]`, which makes comparing with Python a reshape, not a transpose.

**Validation plan:**
1. Same initial state as the Python reference, then compare every population after 20 steps (float64). The test uses an awkward grid with moving walls, an obstacle and a body force.
2. 3D Taylor–Green decay, which has to converge at second order as the grid is refined.
3. 3D lid-driven cavity against the CuPy reference.
4. Speed (MLUPS) and achieved memory bandwidth, plus how big a box fits in 6 GB.
"""

#%% code id=d3_gpu
using CUDA
# Full unrolling without closures: `@nexprs 19 k -> ex` writes out ex for k = 1..19, and
# `@ntuple 19 k -> ex` builds the tuple (ex₁, …, ex₁₉), with k replaced by the literal 1..19.
using Base.Cartesian: @nexprs, @ntuple

CUDA.functional() || error("CUDA.jl cannot use the GPU: ", CUDA.functional(true))
(device = CUDA.name(CUDA.device()),
 memory_GiB = round(CUDA.totalmem(CUDA.device()) / 2^30, digits = 2),
 driver = CUDA.driver_version(), runtime = CUDA.runtime_version(),
 capability = CUDA.capability(CUDA.device()))

#%% code id=d3_lattice
# D3Q19 constants, numbered exactly like lbm.lattice.D3Q19 (Julia index k = Python index + 1):
# rest, the 6 face links (+x −x +y −y +z −z), then 12 edge links, each next to its opposite.
#
# They are FUNCTIONS of k, like cx(i), cy(i), ... in the CUDA C preamble, not global tuples.
# Slate drops `const` from globals (so cells can be re-run), and a GPU kernel cannot read a
# non-const global: its type is unknown at compile time. A function body is always constant,
# and with k a literal (from the unrolled loops) each call folds to an immediate value.
Q19 = 19   # host-side only; kernels use the literal 19 (Val(19), @nexprs 19)
@inline d3_cx(k)  = (0, 1, -1, 0,  0, 0,  0, 1, -1,  1, -1, 1, -1,  1, -1, 0,  0,  0,  0)[k]
@inline d3_cy(k)  = (0, 0,  0, 1, -1, 0,  0, 1, -1, -1,  1, 0,  0,  0,  0, 1, -1,  1, -1)[k]
@inline d3_cz(k)  = (0, 0,  0, 0,  0, 1, -1, 0,  0,  0,  0, 1, -1, -1,  1, 1, -1, -1,  1)[k]
@inline d3_w(k)   = k == 1 ? 1/3 : (k <= 7 ? 1/18 : 1/36)
@inline d3_opp(k) = (1, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14, 17, 16, 19, 18)[k]

# Host-side tuples for checks and moment sums.
D3_CX, D3_CY, D3_CZ = ntuple(d3_cx, Q19), ntuple(d3_cy, Q19), ntuple(d3_cz, Q19)
D3_W, D3_OPP = ntuple(d3_w, Q19), ntuple(d3_opp, Q19)

# Single source of truth: refuse to run if these ever drift from the Python lattice.
let py = pyimport("lbm.lattice").D3Q19
    c = pyconvert(Vector{Tuple{Int,Int,Int}}, py.c)
    @assert Tuple(first.(c)) == D3_CX && Tuple(getindex.(c, 2)) == D3_CY && Tuple(last.(c)) == D3_CZ
    @assert all(isapprox.(pyconvert(Vector{Float64}, py.w), collect(D3_W); atol = 1e-15))
    @assert Tuple(pyconvert(Vector{Int}, py.opposite) .+ 1) == D3_OPP
end
@assert all(D3_CX[D3_OPP[k]] == -D3_CX[k] && D3_CY[D3_OPP[k]] == -D3_CY[k] &&
            D3_CZ[D3_OPP[k]] == -D3_CZ[k] for k in 1:Q19)
(q = Q19, weight_sum = sum(D3_W), matches_python = true)

#%% code id=d3_collide
# How the unrolling works (and why it matters on the GPU):
#   `@ntuple 19 k -> g(k, …)` writes out the tuple (g(1, …), g(2, …), …, g(19, …)). Each g is
#   @inline and receives a LITERAL k, so d3_cx(k), d3_w(k), f[d3_opp(k)] all fold to constants
#   and the 19 populations live in registers. The obvious `ntuple(Val(19)) do k … end` builds a
#   closure that Julia doesn't inline when the body is large; k is then a runtime value, f[k]
#   needs a runtime index, and the whole tuple is pushed out to slow local memory.

# c * a for a lattice component c ∈ {−1, 0, 1}: folds to a, −a or nothing (no multiplications).
@inline mulc(c::Int, a) = c == 0 ? zero(a) : (c == 1 ? a : -a)
@inline cdot(k, ax, ay, az) = mulc(d3_cx(k), ax) + mulc(d3_cy(k), ay) + mulc(d3_cz(k), az)

# (δrho, jx, jy, jz) = (Σ f̃ᵢ, Σ cᵢ f̃ᵢ).
@inline function d3_moments(f::NTuple{19,T}) where {T}
    drho = zero(T); jx = zero(T); jy = zero(T); jz = zero(T)
    @nexprs 19 k -> begin
        drho += f[k]
        jx += mulc(d3_cx(k), f[k]); jy += mulc(d3_cy(k), f[k]); jz += mulc(d3_cz(k), f[k])
    end
    return drho, jx, jy, jz
end

# Node-local TRT + Guo collision, the same operator as kernels/common.cuh and
# lbm.collision.collide (shifted storage f̃ = f − w, so rho = 1 + Σ f̃):
#   u    = (Σ cᵢ f̃ᵢ + F/2) / rho
#   nᵢ   = f̃ᵢ − f̃ᵢᵉᑫ(rho, u)
#   f*ᵢ  = f̃ᵢ − ω⁺/2 (nᵢ + n_opp) − ω⁻/2 (nᵢ − n_opp) + Sᵢ
#   Sᵢ   = (1 − ω⁺/2) wᵢ [9 (c·u)(c·F) − 3 u·F] + (1 − ω⁻/2) 3 wᵢ (c·F)
# TRT only needs the pair sums nᵢ ± n_opp. The equilibrium splits into a part even in c (the
# same for i and opp) and an odd part (flips sign), so with cu = 3 cᵢ·u:
#   nᵢ + n_opp = (f̃ᵢ + f̃_opp) − 2wᵢ [δrho + rho (cu²/2 − 3|u|²/2)]
#   nᵢ − n_opp = (f̃ᵢ − f̃_opp) − 2wᵢ rho cu
# so no separate 19-value nᵢ tuple has to be kept alive.

# Relaxed population for direction k (k a literal: see the note at the top).
@inline function d3_relax(k, f::NTuple{19,T}, drho, rho, ux, uy, uz, usq, hp, hm, ap, am,
                          Fx, Fy, Fz, u_dot_F) where {T}
    w = T(d3_w(k))
    c_dot_u = cdot(k, ux, uy, uz)
    cu = T(3) * c_dot_u
    f_opp = f[d3_opp(k)]
    n_plus  = (f[k] + f_opp) - T(2) * w * (drho + rho * (T(0.5) * cu * cu - usq))
    n_minus = (f[k] - f_opp) - T(2) * w * rho * cu
    c_dot_F = cdot(k, Fx, Fy, Fz)
    source = w * (ap * (T(9) * c_dot_u * c_dot_F - T(3) * u_dot_F) + am * (T(3) * c_dot_F))
    return f[k] - hp * n_plus - hm * n_minus + source
end

@inline function d3_collide(f::NTuple{19,T}, ωp::T, ωm::T, Fx::T, Fy::T, Fz::T) where {T}
    drho, jx, jy, jz = d3_moments(f)
    rho = one(T) + drho
    ux = (jx + T(0.5) * Fx) / rho
    uy = (jy + T(0.5) * Fy) / rho
    uz = (jz + T(0.5) * Fz) / rho
    usq = T(1.5) * (ux * ux + uy * uy + uz * uz)
    hp, hm = T(0.5) * ωp, T(0.5) * ωm
    ap, am = one(T) - hp, one(T) - hm
    u_dot_F = ux * Fx + uy * Fy + uz * Fz
    return @ntuple 19 k -> d3_relax(k, f, drho, rho, ux, uy, uz, usq, hp, hm, ap, am, Fx, Fy, Fz, u_dot_F)
end

# Shifted equilibrium f̃ᵉᑫᵢ = wᵢ [δrho + rho (cu + cu²/2 − 3|u|²/2)], cu = 3 cᵢ·u.
@inline function d3_feq(k, drho::T, rho::T, ux::T, uy::T, uz::T, usq::T) where {T}
    cu = T(3) * cdot(k, ux, uy, uz)
    return T(d3_w(k)) * (drho + rho * (cu + T(0.5) * cu * cu - usq))
end

@inline function d3_equilibrium(drho::T, ux::T, uy::T, uz::T) where {T}
    rho = one(T) + drho
    usq = T(1.5) * (ux * ux + uy * uy + uz * uz)
    return @ntuple 19 k -> d3_feq(k, drho, rho, ux, uy, uz, usq)
end

# Sanity on the CPU: collision conserves mass exactly, and changes momentum by exactly the
# force impulse F (half from relaxing the −F/2 offset, half from the Guo source term).
let f = d3_equilibrium(0.01, 0.02, -0.01, 0.03) .+ ntuple(k -> 1e-3 * sin(k), 19)
    g = d3_collide(f, 1 / 0.8, 1 / 0.9, 1e-4, 0.0, -2e-4)
    (mass_change = sum(g) - sum(f),
     x_momentum_change = sum(D3_CX .* g) - sum(D3_CX .* f),   # should equal Fx = 1e-4
     z_momentum_change = sum(D3_CZ .* g) - sum(D3_CZ .* f))   # should equal Fz = −2e-4
end

#%% code id=d3_kernels
# Periodic neighbour index (1-based), valid for v in [1 − n, 2n].
@inline wrap1(v, n) = v < 1 ? v + n : (v > n ? v - n : v)

# Moving-wall bounce-back term 6 wₖ (cₖ · u_w), u_w read at solid node (x, y, z).
@inline function d3_wall_term(k, wall::AbstractArray{T}, x, y, z) where {T}
    @inbounds T(6 * d3_w(k)) * cdot(k, wall[x, y, z, 1], wall[x, y, z, 2], wall[x, y, z, 3])
end

# One thread per node: x from the thread index (coalesced), y and z from the block grid.
@inline function node_index()
    x = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    return x, blockIdx().y, blockIdx().z
end

# Pull of direction k in the odd step: fₖ(x) = F[opp(k)](x − cₖ), or, if x − cₖ is solid,
# the bounce-back value F[k](x) (which holds f*_opp(k)(x)) plus the moving-wall term.
@inline function d3_pull(k, F, solid, wall, x, y, z, nx, ny, nz)
    xs, ys, zs = wrap1(x - d3_cx(k), nx), wrap1(y - d3_cy(k), ny), wrap1(z - d3_cz(k), nz)
    @inbounds return solid[xs, ys, zs] ? F[x, y, z, k] + d3_wall_term(k, wall, xs, ys, zs) :
                                         F[xs, ys, zs, d3_opp(k)]
end

# Even step: purely node-local. Read F[k], collide, write f*ₖ into the reversed slot opp(k).
function d3_aa_even!(F, solid, ωp, ωm, Fx, Fy, Fz)
    x, y, z = node_index()
    @inbounds if x <= size(F, 1) && !solid[x, y, z]
        f = @ntuple 19 k -> F[x, y, z, k]
        g = d3_collide(f, ωp, ωm, Fx, Fy, Fz)
        @nexprs 19 k -> (F[x, y, z, d3_opp(k)] = g[k])
    end
    return nothing
end

# Odd step: pull fₖ(x) = F[opp(k)](x − cₖ), collide, push f*ₖ to F[k](x + cₖ).
# Bounce-back: a pull from a solid node takes F[k](x) plus the wall term (d3_pull); a push into
# a solid node goes back into F[opp(k)](x) for the next even step.
# Each slot is read and written by exactly one thread, so the in-place update is race-free.
function d3_aa_odd!(F, solid, wall, ωp, ωm, Fx, Fy, Fz)
    x, y, z = node_index()
    nx, ny, nz = size(F, 1), size(F, 2), size(F, 3)
    @inbounds if x <= nx && !solid[x, y, z]
        f = @ntuple 19 k -> d3_pull(k, F, solid, wall, x, y, z, nx, ny, nz)
        g = d3_collide(f, ωp, ωm, Fx, Fy, Fz)
        @nexprs 19 k -> begin
            xd, yd, zd = wrap1(x + d3_cx(k), nx), wrap1(y + d3_cy(k), ny), wrap1(z + d3_cz(k), nz)
            if solid[xd, yd, zd]
                F[x, y, z, d3_opp(k)] = g[k] + d3_wall_term(d3_opp(k), wall, xd, yd, zd)
            else
                F[xd, yd, zd, k] = g[k]
            end
        end
    end
    return nothing
end

# Writes the AA "post-even" layout F[opp(k)] = f̃ᵉᑫₖ from macroscopic fields.
function d3_init_kernel!(F, drho, ux, uy, uz)
    x, y, z = node_index()
    @inbounds if x <= size(F, 1)
        feq = d3_equilibrium(drho[x, y, z], ux[x, y, z], uy[x, y, z], uz[x, y, z])
        @nexprs 19 k -> (F[x, y, z, d3_opp(k)] = feq[k])
    end
    return nothing
end
:kernels_defined

#%% code id=d3_engine
const D3_THREADS = 128   # threads per block along x

# Engine state. F holds the AA "post-even" layout F[x,y,z,opp(k)] = f̃*ₖ between step pairs.
mutable struct D3Q19Sim{T<:AbstractFloat}
    F::CuArray{T,4}         # (nx, ny, nz, 19) shifted populations
    solid::CuArray{Bool,3}  # (nx, ny, nz)
    wall::CuArray{T,4}      # (nx, ny, nz, 3) wall velocity, zero on fluid nodes
    ωp::T
    ωm::T
    force::NTuple{3,T}
    time_step::Int
end

grid_size(sim::D3Q19Sim) = size(sim.solid)
lattice_bytes(sim::D3Q19Sim) = sizeof(sim.F)

"""
    D3Q19Sim(nx, ny, nz; tau, collision=:trt, magic=3/16, force=(0,0,0),
             solid=falses(nx,ny,nz), wall=nothing, T=Float32)

Same parameters and meaning as Python's `SimulationParams` (arrays in Julia [x, y, z] order).
`wall = nothing` means every wall is at rest. Starts at rest with rho = 1; call
`d3_initialize!` for another initial state.

Big arrays are built on the GPU, never as full-size host copies: this laptop has ~12 GB of RAM
shared with Slate, Kaimon and the browser, and a 256³ Float64 field alone is 134 MB.
"""
function D3Q19Sim(nx, ny, nz; tau, collision = :trt, magic = 3 / 16, force = (0.0, 0.0, 0.0),
                  solid = falses(nx, ny, nz), wall = nothing, T = Float32)
    tau > 0.5 || throw(ArgumentError("tau must be > 1/2 for positive viscosity; got $tau"))
    collision in (:bgk, :trt) || throw(ArgumentError("collision must be :bgk or :trt; got $collision"))
    size(solid) == (nx, ny, nz) || throw(DimensionMismatch("solid must be $((nx, ny, nz)); got $(size(solid))"))
    wall === nothing || size(wall) == (nx, ny, nz, 3) ||
        throw(DimensionMismatch("wall must be $((nx, ny, nz, 3)); got $(size(wall))"))
    length(force) == 3 || throw(ArgumentError("force needs 3 components; got $force"))
    ωp = 1 / tau
    ωm = collision == :bgk ? ωp : 1 / (0.5 + magic / (tau - 0.5))   # TRT magic parameter Λ
    solid_d = CuArray{Bool}(solid)
    wall_d = wall === nothing ? CUDA.zeros(T, nx, ny, nz, 3) :
             CuArray{T}(wall) .* solid_d                    # wall velocity only matters on solids
    sim = D3Q19Sim{T}(CUDA.zeros(T, nx, ny, nz, Q19), solid_d, wall_d,
                      T(ωp), T(ωm), T.(Tuple(force)), 0)
    return d3_initialize!(sim)
end

function launch_config(sim::D3Q19Sim)
    nx, ny, nz = grid_size(sim)
    threads = min(D3_THREADS, nx)
    return (threads = threads, blocks = (cld(nx, threads), ny, nz))
end

"""
    d3_initialize!(sim; rho=1, ux=0, uy=0, uz=0)

Equilibrium state from scalars or (nx, ny, nz) arrays (host or GPU). The stored state is
post-collision and its moments report rho u = Σ c f* − F/2, so the equilibrium is built with
u + F/(2 rho): the reported velocity at t = 0 is then the requested one (same as Python
`_initial_populations`). Fields go straight to the GPU; scalars are filled there.
"""
function d3_initialize!(sim::D3Q19Sim{T}; rho = 1.0, ux = 0.0, uy = 0.0, uz = 0.0) where {T}
    shape = grid_size(sim)
    device_field(a) = a isa Number ? CUDA.fill(T(a), shape) :
        (size(a) == shape ? CuArray{T}(a) : throw(DimensionMismatch("field must be $shape; got $(size(a))")))
    rho_d = device_field(rho)
    u_d = [device_field(u) .+ T(0.5) .* sim.force[a] ./ rho_d for (a, u) in enumerate((ux, uy, uz))]
    (; threads, blocks) = launch_config(sim)
    @cuda threads = threads blocks = blocks d3_init_kernel!(sim.F, rho_d .- one(T), u_d...)
    sim.time_step = 0
    return sim
end

"""
    d3_step!(sim, steps)

Advance `steps` time steps (even: one odd + one even AA kernel per pair). Returns immediately;
the GPU works asynchronously until something reads the result (or `CUDA.synchronize()`).
"""
function d3_step!(sim::D3Q19Sim, steps::Integer)
    iseven(steps) && steps >= 0 || throw(ArgumentError("the AA pattern advances in pairs; got steps=$steps"))
    (; threads, blocks) = launch_config(sim)
    Fx, Fy, Fz = sim.force
    for _ in 1:steps÷2
        @cuda threads = threads blocks = blocks d3_aa_odd!(sim.F, sim.solid, sim.wall, sim.ωp, sim.ωm, Fx, Fy, Fz)
        @cuda threads = threads blocks = blocks d3_aa_even!(sim.F, sim.solid, sim.ωp, sim.ωm, Fx, Fy, Fz)
    end
    sim.time_step += steps
    return sim
end

"Host copy of post-collision populations in natural order, shape (nx, ny, nz, 19)."
d3_populations(sim::D3Q19Sim) = Array(sim.F)[:, :, :, collect(D3_OPP)]

# Moments by one matrix product over the whole lattice: column 1 sums the populations (δrho),
# columns 2–4 give momentum. Slots are reversed (F[opp(k)] = f*ₖ) and c_opp = −c, so the
# momentum Σₖ cₖ f*ₖ equals −Σⱼ cⱼ F[j]: hence the minus signs.
const D3_MOMENT_MATRIX = hcat(ones(Q19), -collect(D3_CX), -collect(D3_CY), -collect(D3_CZ))

"""
    d3_macroscopic(sim) -> (; rho, ux, uy, uz)

Host (nx, ny, nz) fields from post-collision moments: rho u = Σ c f* − F/2.
Solid nodes report rho = 1, u = 0.
"""
function d3_macroscopic(sim::D3Q19Sim{T}) where {T}
    m = reshape(sim.F, :, Q19) * CuArray{T}(D3_MOMENT_MATRIX)
    fluid = .!vec(sim.solid)
    rho = ifelse.(fluid, one(T) .+ m[:, 1], one(T))
    u = ntuple(a -> ifelse.(fluid, (m[:, a+1] .- T(0.5) * sim.force[a]) ./ (one(T) .+ m[:, 1]), zero(T)), 3)
    host(a) = reshape(Array(a), grid_size(sim))
    return (rho = host(rho), ux = host(u[1]), uy = host(u[2]), uz = host(u[3]))
end

"Throw if any fluid node has non-finite populations or rho ≤ 0 (the Python watchdog's rule)."
function d3_check_health(sim::D3Q19Sim)
    drho = vec(sum(sim.F; dims = 4))
    bad = (.!isfinite.(drho) .| (drho .<= -1)) .& .!vec(sim.solid)
    any(bad) && error("D3Q19 simulation diverged by step $(sim.time_step) at node ",
                      Tuple(CartesianIndices(grid_size(sim))[findfirst(Array(bad))]))
    return sim
end

"Advance with a divergence check every `check_every` steps."
function d3_run!(sim::D3Q19Sim, steps::Integer; check_every = 1000)
    done = 0
    while done < steps
        chunk = min(check_every, steps - done)
        d3_step!(sim, chunk)
        d3_check_health(sim)
        done += chunk
    end
    return sim
end
:engine_defined

#%% code id=d3_check_python
# Validation 1: same initial state + same scheme ⇒ same numbers.
# A deliberately awkward float64 case, so any index or sign slip shows up:
#   non-cubic 12×10×8 grid, periodic in x and z, plates at y = 1 and y = ny,
#   the top plate moving in x and z, a solid block in the middle, a body force in all 3 axes,
#   TRT, and a smooth non-separable initial density + velocity field.
const np = pyimport("numpy")
to_numpy(a) = np.ascontiguousarray(np.asarray(a).T)        # Julia [x,y,z,…] → NumPy [… ,z,y,x]
from_numpy(p) = pyconvert(Array{Float64}, np.asarray(p).T)  # NumPy [q,z,y,x] → Julia [x,y,z,q]

eq_n = (12, 10, 8)
eq_solid = falses(eq_n...)
eq_solid[:, [1, end], :] .= true            # plates
eq_solid[5:7, 4:6, 3:5] .= true             # obstacle
eq_wall = zeros(eq_n..., 3)
eq_wall[:, end, :, 1] .= 0.05               # top plate slides in +x …
eq_wall[:, end, :, 3] .= 0.02               # … and +z
eq_force = (1e-5, -2e-5, 3e-5)

eq_x, eq_y, eq_z = (reshape(1:n, ntuple(d -> d == a ? n : 1, 3)) for (a, n) in enumerate(eq_n))
eq_rho = 1 .+ 0.01 .* sin.(0.7 .* eq_x .+ 0.3 .* eq_y) .* cos.(0.5 .* eq_z)
eq_ux  = 0.02 .* sin.(0.4 .* eq_y .+ 0.9 .* eq_z) .+ 0 .* eq_x
eq_uy  = 0.01 .* cos.(0.8 .* eq_x) .* sin.(0.6 .* eq_z) .+ 0 .* eq_y
eq_uz  = -0.015 .* sin.(0.5 .* eq_x .+ 0.2 .* eq_y) .+ 0 .* eq_z

eq_ref = lbm.make_solver(lbm.SimulationParams(
        nx = eq_n[1], ny = eq_n[2], nz = eq_n[3], tau = 0.8, collision = "trt",
        force = eq_force, solid = to_numpy(eq_solid), wall_velocity = to_numpy(eq_wall),
        dtype = "float64"), backend = "numpy")
eq_ref.initialize(rho = to_numpy(eq_rho), ux = to_numpy(eq_ux), uy = to_numpy(eq_uy), uz = to_numpy(eq_uz))

eq_sim = D3Q19Sim(eq_n...; tau = 0.8, collision = :trt, force = eq_force,
                  solid = eq_solid, wall = eq_wall, T = Float64)
d3_initialize!(eq_sim; rho = eq_rho, ux = eq_ux, uy = eq_uy, uz = eq_uz)

max_diff() = maximum(abs, d3_populations(eq_sim) .- from_numpy(eq_ref.populations()))
eq_initial = max_diff()

eq_steps = 20
eq_ref.run(eq_steps, check_every = eq_steps)
d3_step!(eq_sim, eq_steps)
eq_after = max_diff()

eq_fields = d3_macroscopic(eq_sim)
eq_ref_fields = eq_ref.macroscopic()
eq_u_diff = maximum(abs, eq_fields.uz .- pyconvert(Array{Float64}, np.asarray(eq_ref_fields.u[2]).T))

(initial_max_diff = eq_initial, after_20_steps_max_diff = eq_after, uz_max_diff = eq_u_diff,
 max_population = maximum(abs, d3_populations(eq_sim)),
 verdict = eq_after < 1e-13 ? "matches the Python reference to rounding" : "MISMATCH")

#%% code id=d3_taylor_green
# Validation 2: 3D Taylor–Green vortex in a periodic box,
#   u = U sin(kx) cos(ky) cos(kz),  v = −U cos(kx) sin(ky) cos(kz),  w = 0,   k = 2π/n.
# At low Reynolds number every velocity component has |k|² = 3k², so kinetic energy decays as
# E(t) = E₀ exp(−6 ν k² t). We run until E should have fallen by e^{-1}, measure the average
# decay rate, and compare it with 6νk². The error should fall like 1/n² (second-order accuracy).
# Same setup as the earlier NumPy spike (outputs/spikes/d3q19_spike.py, two-lattice pull
# scheme), which gives 7.956 / 1.850 / 0.844 / 0.502 % at 16 / 32 / 48 / 64³: an independent
# cross-check of this engine.
const TG_TAU, TG_U = 0.8, 0.01

function taylor_green_fields(n)
    k = 2π / n
    x, y, z = (reshape(0:n-1, ntuple(d -> d == a ? n : 1, 3)) for a in 1:3)
    ux = TG_U .* sin.(k .* x) .* cos.(k .* y) .* cos.(k .* z)
    uy = -TG_U .* cos.(k .* x) .* sin.(k .* y) .* cos.(k .* z)
    return ux, uy, zero(ux)
end

kinetic_energy(m) = sum(0.5 .* m.rho .* (m.ux .^ 2 .+ m.uy .^ 2 .+ m.uz .^ 2))

function taylor_green_decay(n; T = Float64)
    nu = (TG_TAU - 0.5) / 3
    k = 2π / n
    exact_rate = 6nu * k^2
    steps = 2round(Int, 0.5 / exact_rate)          # E falls by ≈ e^{-1}; even for AA
    sim = D3Q19Sim(n, n, n; tau = TG_TAU, collision = :bgk, T = T)
    ux, uy, uz = taylor_green_fields(n)
    d3_initialize!(sim; ux = ux, uy = uy, uz = uz)
    e0 = kinetic_energy(d3_macroscopic(sim))
    d3_run!(sim, steps; check_every = steps)
    rate = -log(kinetic_energy(d3_macroscopic(sim)) / e0) / steps
    return (n = n, steps = steps, rate = rate, exact = exact_rate, error_pct = 100 * (rate / exact_rate - 1))
end

tg_sizes = (16, 32, 48, 64)
tg64 = [taylor_green_decay(n; T = Float64) for n in tg_sizes]
tg32 = [taylor_green_decay(n; T = Float32) for n in tg_sizes]

# Observed order between consecutive grids: p = log(e₁/e₂) / log(n₂/n₁).
order(r) = [log(abs(r[i].error_pct / r[i+1].error_pct)) / log(r[i+1].n / r[i].n) for i in 1:length(r)-1]

slate_table([(n = a.n, steps = a.steps,
              error_float64_pct = round(a.error_pct, digits = 3),
              error_float32_pct = round(b.error_pct, digits = 3),
              observed_order = i == 1 ? missing : round(order(tg64)[i-1], digits = 2))
             for (i, (a, b)) in enumerate(zip(tg64, tg32))])

#%% code id=d3_cavity_check
# Validation 3: 3D lid-driven cube, Julia engine vs the Python CuPy reference (same GPU).
# Cube of 64 fluid nodes per side in a one-node solid frame; the top (y) face slides in +x.
const C3_N, C3_U, C3_RE = 64, 0.1, 100
c3_tau = 3 * (C3_U * C3_N / C3_RE) + 0.5

c3_solid_py, c3_wall_py = lbm_boundaries.cavity_walls(C3_N, C3_U, dimensions = 3)
c3_solid = pyconvert(Array{Bool}, np.asarray(c3_solid_py).T)   # (nx, ny, nz)
c3_wall  = from_numpy(c3_wall_py)                                # (nx, ny, nz, 3)
c3_side = C3_N + 2

c3_ref = lbm.make_solver(lbm.SimulationParams(
        nx = c3_side, ny = c3_side, nz = c3_side, tau = c3_tau, collision = "trt",
        solid = c3_solid_py, wall_velocity = c3_wall_py, dtype = "float32"), backend = "cupy")
c3_sim = D3Q19Sim(c3_side, c3_side, c3_side; tau = c3_tau, collision = :trt,
                  solid = c3_solid, wall = c3_wall, T = Float32)

c3_check_steps = 2000   # ≈ 3 lid-transits: the vortex is still forming, so the fields are far from trivial
c3_ref_seconds = @elapsed (c3_ref.run(c3_check_steps, check_every = c3_check_steps); c3_ref.macroscopic())
c3_jl_seconds  = @elapsed (d3_step!(c3_sim, c3_check_steps); CUDA.synchronize())

c3_jl = d3_macroscopic(c3_sim)
c3_py = c3_ref.macroscopic()
# Python tuples are indexed from 0 through PythonCall: u[0], u[1], u[2] = ux, uy, uz.
c3_diff = maximum(maximum(abs, getproperty(c3_jl, name) .- pyconvert(Array{Float64}, np.asarray(c3_py.u[a-1]).T))
                  for (a, name) in enumerate((:ux, :uy, :uz))) / C3_U
c3_nodes = c3_side^3

(tau = c3_tau, max_velocity_diff_over_U = c3_diff,
 max_speed_over_U = maximum(sqrt.(c3_jl.ux .^ 2 .+ c3_jl.uy .^ 2 .+ c3_jl.uz .^ 2)) / C3_U,
 cupy_reference_MLUPS = round(c3_nodes * c3_check_steps / c3_ref_seconds / 1e6, digits = 1),
 julia_engine_MLUPS = round(c3_nodes * c3_check_steps / c3_jl_seconds / 1e6, digits = 1))

#%% code id=d3_cavity_steady
# Carry the cube on to steady state with the Julia engine alone (same rule as the 2D run:
# no velocity component, in units of U, moves more than the tolerance over one lid-transit).
function d3_run_to_steady!(sim; lid_velocity, interval, tol, max_steps)
    history = (step = Int[], change = Float64[])
    velocity(m) = (m.ux, m.uy, m.uz) ./ lid_velocity
    previous = velocity(d3_macroscopic(sim))
    while sim.time_step < max_steps
        d3_run!(sim, interval; check_every = interval)
        current = velocity(d3_macroscopic(sim))
        change = maximum(maximum(abs, c .- p) for (c, p) in zip(current, previous))
        push!(history.step, sim.time_step); push!(history.change, change)
        change < tol && return (; history, converged = true)
        previous = current
    end
    return (; history, converged = false)
end

c3_interval = 2round(Int, C3_N / C3_U / 2)    # one lid-transit, even
c3_seconds = @elapsed c3_result = d3_run_to_steady!(c3_sim; lid_velocity = C3_U,
    interval = c3_interval, tol = 1e-6, max_steps = 400_000)
c3_steady = d3_macroscopic(c3_sim)

(converged = c3_result.converged, steps = c3_sim.time_step,
 final_change = last(c3_result.history.change), seconds = round(c3_seconds, digits = 1),
 MLUPS = round(c3_nodes * (c3_sim.time_step - c3_check_steps) / c3_seconds / 1e6, digits = 1),
 mass_drift = sum(c3_steady.rho[.!c3_solid]) / count(.!c3_solid) - 1)

#%% code id=d3_cavity_plots
# Mid-plane z = 0.5 of the cube (n even ⇒ average the two middle planes), fluid region only.
interior3(a) = a[2:end-1, 2:end-1, 2:end-1]
middle(a, dim) = (n = size(a, dim); (selectdim(a, dim, n ÷ 2) .+ selectdim(a, dim, n ÷ 2 + 1)) ./ 2)

c3_ux_mid = middle(interior3(c3_steady.ux), 3) ./ C3_U    # (x, y)
c3_uy_mid = middle(interior3(c3_steady.uy), 3) ./ C3_U
c3_coords = ((1:C3_N) .- 0.5) ./ C3_N
c3_u_line = [0.0; vec(middle(c3_ux_mid, 1)); 1.0]   # u on x = 0.5, as a function of y
c3_v_line = [0.0; vec(middle(c3_uy_mid, 2)); 0.0]   # v on y = 0.5, as a function of x

ghia100 = pyimport("lbm.cases.ghia1982").ghia_centerlines(100)

c3_profile_chart = echart(
    series(:line, c3_u_line, [0.0; c3_coords; 1.0]; name = "3D cube, mid-plane (Julia D3Q19)", showSymbol = false),
    series(:scatter, pyconvert(Vector{Float64}, ghia100.u), pyconvert(Vector{Float64}, ghia100.y);
           name = "2D Ghia Re = 100 (for contrast, not a 3D reference)", symbolSize = 7);
    title = "u / U on x = 0.5, z = 0.5, Re = $C3_RE",
    xAxis = (type = :value, name = "u / U", min = -0.4, max = 1.0),
    yAxis = (type = :value, name = "y", min = 0, max = 1), height = 420)

#%% code id=d3_cavity_vorticity
# ω_z on the mid-plane (the 2D helpers want [y, x] matrices, hence the transposes).
c3_omega_mid = vorticity(permutedims(c3_ux_mid), permutedims(c3_uy_mid), C3_N)
vorticity_heatmap(c3_omega_mid; block = 1,
    title = "3D cube, mid-plane z = 0.5: vorticity ω_z (U/L), Re = $C3_RE")

#%% code id=d3_benchmark
# Validation 4: speed. LBM is memory-bound: each node update must read and write 19 floats,
# so the ceiling is (memory bandwidth) / (2 · 19 · 4 B = 152 B). We measure the real bandwidth
# with a plain device-to-device copy and report how close the kernels get to it.
const BYTES_PER_UPDATE = 2 * Q19 * sizeof(Float32)

function copy_bandwidth_GBs(; bytes = 2^29, repeats = 10)
    src = CUDA.ones(Float32, bytes ÷ 4); dst = similar(src)
    copyto!(dst, src)                                      # warm-up
    t = CUDA.@elapsed for _ in 1:repeats; copyto!(dst, src); end
    CUDA.unsafe_free!(src); CUDA.unsafe_free!(dst)
    return 2 * bytes * repeats / t / 1e9                   # read + write
end

# Periodic box, TRT, uniform flow. The cost per update doesn't depend on the flow, and a scalar
# initial state is filled on the GPU, so even 256³ needs no big host arrays.
function benchmark_d3(n; target_updates = 3e9)
    sim = D3Q19Sim(n, n, n; tau = 0.8, collision = :trt, T = Float32)
    d3_initialize!(sim; ux = 0.05, uy = 0.02, uz = -0.01)
    d3_step!(sim, 2); CUDA.synchronize()                   # warm-up
    steps = max(2, 2round(Int, target_updates / n^3 / 2))
    t = CUDA.@elapsed d3_step!(sim, steps)
    d3_check_health(sim)
    mlups = n^3 * steps / t / 1e6
    result = (n = n, steps = steps, MLUPS = round(mlups, digits = 0),
              GBs = round(mlups * 1e6 * BYTES_PER_UPDATE / 1e9, digits = 1),
              lattice_MiB = round(lattice_bytes(sim) / 2^20, digits = 0))
    CUDA.unsafe_free!(sim.F); CUDA.unsafe_free!(sim.wall); CUDA.unsafe_free!(sim.solid)
    return result
end

peak_GBs = copy_bandwidth_GBs()
d3_bench = [benchmark_d3(n) for n in (64, 96, 128, 160, 192, 256)]

# Register use and spills of the odd kernel (the heavier one).
d3_odd_kernel = let s = D3Q19Sim(8, 8, 8; tau = 0.8, T = Float32)
    @cuda launch = false d3_aa_odd!(s.F, s.solid, s.wall, s.ωp, s.ωm, s.force...)
end

slate_table([(; r..., pct_of_copy_peak = round(100 * r.GBs / peak_GBs, digits = 0)) for r in d3_bench])

#%% code id=d3_memory
# Kernel resources and memory budget.
# Per node: 19 float32 populations (76 B) + solid flag (1 B) + wall velocity (3 × 4 B = 12 B).
const D3_BYTES_PER_NODE = Q19 * 4 + 1 + 3 * 4
GC.gc(); CUDA.reclaim()                     # hand cached blocks back so "free" is honest
d3_free = CUDA.free_memory()
d3_max_side = floor(Int, (0.9 * d3_free / D3_BYTES_PER_NODE)^(1 / 3))

(odd_kernel_registers = CUDA.registers(d3_odd_kernel),
 odd_kernel_local_bytes = CUDA.memory(d3_odd_kernel).local,   # > 0 would mean register spills
 copy_peak_GBs = round(peak_GBs, digits = 1),
 free_GiB = round(d3_free / 2^30, digits = 2),
 bytes_per_node = D3_BYTES_PER_NODE,
 largest_cube = "$(d3_max_side)³ ≈ $(round(Int, d3_max_side^3 / 1e6)) M nodes")

#%% md id=d3_summary
@md"""
### What the 3D engine shows

| Check | Result |
|---|---|
| Same state, same scheme as the Python NumPy reference (float64, 20 steps, walls + moving plate + obstacle + force) | max population difference **4e-18** (populations ~7e-3): equal to machine precision |
| 3D Taylor–Green decay, 16 / 32 / 48 / 64³ | error 7.956 / 1.850 / 0.844 / 0.502 %, observed order ≈ 2 (2.1, 1.94, 1.8); **identical** to the independent NumPy spike; float32 = float64 |
| 3D lid-driven cube, 64³, Re = 100, vs the CuPy reference after 2000 steps | max velocity difference **4.5e-7 U** (float32 rounding) |
| Same cube run to steady state | 11.6k steps, ~5 s; mass drift −1.2e-7 |
| Speed, float32, 64³–256³ | **84–97 % of the copy bandwidth measured in the same run**: 700–790 MLUPS when the copy peak was 123 GB/s, 870–1000 MLUPS when it was 157 GB/s |
| CuPy reference on the same cube | 21–27 MLUPS: the fused Julia kernels are ~30× faster |
| Memory | 89 B per node, so ~380³ ≈ 55 M nodes fit in the free 5 GB |

**Why the speed moves between runs:** the laptop GPU changes its memory clock with its power state, so the copy peak itself varies (123–157 GB/s here). The fraction of that peak the kernels reach is the stable number, which is why the benchmark measures the copy peak in the same run.

**Two GPU-programming lessons came out of the speed work:**

1. **Unrolling must really unroll.** The first version built the 19 values with `ntuple(Val(19)) do k … end`. The kernel compiled with 232–368 B of *local memory* per thread, meaning values that should have sat in registers were spilled to slow memory, and it reached only 35–55 % of the copy peak. The likely cause: the closure wasn't inlined, so `k` was a runtime value and `f[k]` needed a runtime index. Writing the tuple out literally with `Base.Cartesian.@ntuple 19 k -> …` makes every `k` a constant. Local memory dropped to 0 bytes and speed rose ~1.7×.
2. **Write the physics so it needs fewer live values.** TRT only needs nᵢ ± n_opp, and those have closed forms in fᵢ ± f_opp. So the collision never stores a separate 19-value non-equilibrium tuple.

**Slate specifics:** Slate drops `const` from globals so cells can be re-run, and a GPU kernel can't read a non-const global. The lattice constants are therefore functions (`d3_cx(k)`, `d3_w(k)`, …), which is also how the CUDA C preamble does it.

**What's still 2D-only:** the CUDA C backends in the Python package. The Python reference (NumPy/CuPy) does run 3D, which is what made these checks possible.
"""

# ╔═╡ Slate.env · notebook packages (auto-maintained — manage via the package panel)
#   CUDA 6.4.2 052768ef-5323-5732-b1bb-66c8b64840ba
#   PythonCall 0.9.37 6099a3de-0909-46bc-b1f4-468b9a2dfc0d
# ╚═╡
# ╔═╡ Slate.config · per-notebook settings (Settings panel)
#   threads = 1,0
#   docid = 88089c58-e478-4b90-a012-8f55ecc4882e
# ╚═╡
