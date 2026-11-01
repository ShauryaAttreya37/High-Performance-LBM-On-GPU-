// Node-local physics shared by every kernel variant, for D2Q9 and D3Q19.
//
// The variants (naive, tiled, AA) differ only in *where* they read and write
// populations. Once the Q values of a node sit in registers, they all call
// collide_node(), so a physics bug cannot hide in one variant only.
//
// This file is concatenated after the generated preamble (real_t, DIM, Q,
// cx(), cy(), cz(), opp(), weight()) when CudaSolver compiles a kernel. The
// mathematics is identical, term for term, to lbm.collision.collide in Python:
//
//   rho  = 1 + sum_i f_i                      (shifted storage: f_i = f_i - w_i)
//   u    = (sum_i c_i f_i + F/2) / rho        (Guo half-force shift)
//   n_i  = f_i - feq_i(rho, u)                (non-equilibrium part)
//   f*_i = f_i - w+ (n_i + n_opp)/2 - w- (n_i - n_opp)/2 + S_i
//   S_i  = (1 - w+/2) w_i [9 (c.u)(c.F) - 3 u.F] + (1 - w-/2) 3 w_i (c.F)
//
// One source serves both dimensions. A 2D lattice is a 3D one with nz = 1 and
// cz(i) = 0 for every i: every z term below is either multiplied by a cz(i)
// the compiler knows is 0, or guarded by the compile-time constant THREE_D, so
// the D2Q9 kernels compile to the same code as a hand-written 2D version.
//
// Every loop over directions is fully unrolled, so `i` and `opp(i)` are
// compile-time constants and f[] lives in registers.

constexpr bool THREE_D = DIM == 3;

// Node kinds stored in the solid mask byte.
constexpr unsigned char FLUID = 0;
constexpr unsigned char SOLID = 1;
constexpr unsigned char FORCED = 2;  // fluid node that also receives a local force (IBM)

struct Relaxation {
    real_t omega_plus;   // 1 / tau, sets the viscosity
    real_t omega_minus;  // antisymmetric rate; equal to omega_plus for BGK
    real_t force_x;      // uniform body-force density
    real_t force_y;
    real_t force_z;      // 0 in 2D
};

// Shifted equilibrium of direction i: feq_i - w_i.
__device__ __forceinline__ real_t shifted_feq(int i, real_t drho, real_t rho,
                                              real_t ux, real_t uy, real_t uz, real_t usq)
{
    const real_t cu = (real_t)3 * (cx(i) * ux + cy(i) * uy + cz(i) * uz);
    return weight(i) * (drho + rho * (cu + (real_t)0.5 * cu * cu - usq));
}

// Guo source term S_i.
__device__ __forceinline__ real_t guo_source(int i, const Relaxation& r, real_t ux, real_t uy,
                                             real_t uz, real_t u_dot_f)
{
    const real_t c_dot_f = cx(i) * r.force_x + cy(i) * r.force_y + cz(i) * r.force_z;
    const real_t c_dot_u = cx(i) * ux + cy(i) * uy + cz(i) * uz;
    const real_t a_plus = (real_t)1 - (real_t)0.5 * r.omega_plus;
    const real_t a_minus = (real_t)1 - (real_t)0.5 * r.omega_minus;
    return weight(i) * (a_plus * ((real_t)9 * c_dot_u * c_dot_f - (real_t)3 * u_dot_f)
                        + a_minus * ((real_t)3 * c_dot_f));
}

__device__ __forceinline__ void collide_node(real_t f[Q], const Relaxation r)
{
    real_t drho = 0, jx = 0, jy = 0, jz = 0;
#pragma unroll
    for (int i = 0; i < Q; ++i) {
        drho += f[i];
        jx += cx(i) * f[i];
        jy += cy(i) * f[i];
        jz += cz(i) * f[i];
    }
    const real_t rho = (real_t)1 + drho;
    const real_t ux = (jx + (real_t)0.5 * r.force_x) / rho;
    const real_t uy = (jy + (real_t)0.5 * r.force_y) / rho;
    const real_t uz = THREE_D ? (jz + (real_t)0.5 * r.force_z) / rho : (real_t)0;
    const real_t usq = (real_t)1.5 * (ux * ux + uy * uy + uz * uz);
    const real_t u_dot_f = ux * r.force_x + uy * r.force_y + (THREE_D ? uz * r.force_z : (real_t)0);
    const real_t half_wp = (real_t)0.5 * r.omega_plus;
    const real_t half_wm = (real_t)0.5 * r.omega_minus;

    // Visit each opposite pair (i, opp(i)) once and update both members, so no
    // separate Q-value non-equilibrium array has to stay live: fewer registers,
    // which matters for D3Q19 in double precision. The rest direction is its
    // own opposite and relaxes with w+ only.
#pragma unroll
    for (int i = 0; i < Q; ++i) {
        const int j = opp(i);
        if (j < i) continue;
        const real_t n_i = f[i] - shifted_feq(i, drho, rho, ux, uy, uz, usq);
        if (j == i) {
            f[i] = f[i] - r.omega_plus * n_i + guo_source(i, r, ux, uy, uz, u_dot_f);
            continue;
        }
        const real_t n_j = f[j] - shifted_feq(j, drho, rho, ux, uy, uz, usq);
        const real_t sym = half_wp * (n_i + n_j);
        const real_t asym = half_wm * (n_i - n_j);
        f[i] = f[i] - sym - asym + guo_source(i, r, ux, uy, uz, u_dot_f);
        f[j] = f[j] - sym + asym + guo_source(j, r, ux, uy, uz, u_dot_f);
    }
}

// Add the local force density of a FORCED node (field holds DIM components of n_nodes values).
__device__ __forceinline__ void add_local_force(Relaxation& r, const real_t* __restrict__ field,
                                                size_t node, size_t n_nodes)
{
    r.force_x += field[node];
    r.force_y += field[n_nodes + node];
    if (THREE_D) r.force_z += field[2 * n_nodes + node];
}

// Moving-wall bounce-back momentum term 6 w_i rho_w (c_i . u_w), rho_w = 1,
// with the wall velocity read at solid node `s`. wall_u holds DIM components
// of n_nodes values each; components where c_i is 0 are never loaded.
__device__ __forceinline__ real_t wall_term(int i, const real_t* __restrict__ wall_u,
                                            size_t s, size_t n_nodes)
{
    real_t c_dot_u = (real_t)0;
    if (cx(i) != 0) c_dot_u += cx(i) * wall_u[s];
    if (cy(i) != 0) c_dot_u += cy(i) * wall_u[n_nodes + s];
    if (THREE_D && cz(i) != 0) c_dot_u += cz(i) * wall_u[2 * n_nodes + s];
    return (real_t)6 * weight(i) * c_dot_u;
}

// Periodic neighbour coordinate, cheap version: valid for v in [-n, 2n).
__device__ __forceinline__ int wrap(int v, int n)
{
    return v < 0 ? v + n : (v >= n ? v - n : v);
}

// Periodic coordinate for any v (used where offsets can exceed the domain size).
__device__ __forceinline__ int positive_mod(int v, int n)
{
    const int m = v % n;
    return m < 0 ? m + n : m;
}

// Grid position of one thread. Launch grid: (ceil(nx/bx), ceil(ny/by), nz).
struct Node {
    int x, y, z;
    size_t index;  // (z * ny + y) * nx + x
};

__device__ __forceinline__ bool locate(Node& n, int nx, int ny, int nz)
{
    n.x = blockIdx.x * blockDim.x + threadIdx.x;
    n.y = blockIdx.y * blockDim.y + threadIdx.y;
    n.z = THREE_D ? (int)blockIdx.z : 0;
    if (n.x >= nx || n.y >= ny || n.z >= nz) return false;
    n.index = ((size_t)n.z * ny + n.y) * nx + n.x;
    return true;
}

// Index of the periodic neighbour at node + S * c_i (S = -1 upstream, +1 downstream).
// Axes on which c_i has no component skip the wrap (folded at compile time).
template <int S>
__device__ __forceinline__ size_t neighbour(int i, const Node& n, int nx, int ny, int nz)
{
    const int x = cx(i) == 0 ? n.x : wrap(n.x + S * cx(i), nx);
    const int y = cy(i) == 0 ? n.y : wrap(n.y + S * cy(i), ny);
    const int z = (!THREE_D || cz(i) == 0) ? n.z : wrap(n.z + S * cz(i), nz);
    return ((size_t)z * ny + y) * nx + x;
}
