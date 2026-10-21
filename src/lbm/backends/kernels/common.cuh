// Node-local physics shared by every kernel variant.
//
// The variants (naive, tiled, AA) differ only in *where* they read and write
// populations. Once the nine values of a node sit in registers, they all call
// collide_node(), so a physics bug cannot hide in one variant only.
//
// This file is concatenated after the generated preamble (real_t, Q, cx(),
// cy(), opp(), weight()) when CudaSolver compiles a kernel. The mathematics is
// identical, term for term, to lbm.collision.collide in Python:
//
//   rho  = 1 + sum_i f_i                      (shifted storage: f_i = f_i - w_i)
//   u    = (sum_i c_i f_i + F/2) / rho        (Guo half-force shift)
//   n_i  = f_i - feq_i(rho, u)                (non-equilibrium part)
//   f*_i = f_i - w+ (n_i + n_opp)/2 - w- (n_i - n_opp)/2 + S_i
//   S_i  = (1 - w+/2) w_i [9 (c.u)(c.F) - 3 u.F] + (1 - w-/2) 3 w_i (c.F)
//
// Every loop over directions is fully unrolled, so `i` and `opp(i)` are
// compile-time constants and the arrays f[], neq[] live in registers.

struct Relaxation {
    real_t omega_plus;   // 1 / tau, sets the viscosity
    real_t omega_minus;  // antisymmetric rate; equal to omega_plus for BGK
    real_t force_x;      // uniform body-force density
    real_t force_y;
};

__device__ __forceinline__ void collide_node(real_t f[Q], const Relaxation r)
{
    real_t drho = 0, jx = 0, jy = 0;
#pragma unroll
    for (int i = 0; i < Q; ++i) {
        drho += f[i];
        jx += cx(i) * f[i];
        jy += cy(i) * f[i];
    }
    const real_t rho = (real_t)1 + drho;
    const real_t ux = (jx + (real_t)0.5 * r.force_x) / rho;
    const real_t uy = (jy + (real_t)0.5 * r.force_y) / rho;
    const real_t usq = (real_t)1.5 * (ux * ux + uy * uy);

    real_t neq[Q];
#pragma unroll
    for (int i = 0; i < Q; ++i) {
        const real_t cu = (real_t)3 * (cx(i) * ux + cy(i) * uy);
        const real_t feq = weight(i) * (drho + rho * (cu + (real_t)0.5 * cu * cu - usq));
        neq[i] = f[i] - feq;
    }

    const real_t half_wp = (real_t)0.5 * r.omega_plus;
    const real_t half_wm = (real_t)0.5 * r.omega_minus;
    const real_t a_plus = (real_t)1 - half_wp;
    const real_t a_minus = (real_t)1 - half_wm;
    const real_t u_dot_f = ux * r.force_x + uy * r.force_y;
#pragma unroll
    for (int i = 0; i < Q; ++i) {
        const real_t c_dot_f = cx(i) * r.force_x + cy(i) * r.force_y;
        const real_t c_dot_u = cx(i) * ux + cy(i) * uy;
        const real_t source = weight(i) * (a_plus * ((real_t)9 * c_dot_u * c_dot_f - (real_t)3 * u_dot_f)
                                           + a_minus * ((real_t)3 * c_dot_f));
        f[i] = f[i] - half_wp * (neq[i] + neq[opp(i)]) - half_wm * (neq[i] - neq[opp(i)]) + source;
    }
}

// Moving-wall bounce-back momentum term 6 w_i rho_w (c_i . u_w), rho_w = 1,
// with the wall velocity read at solid node `s`. Stationary walls give zero.
__device__ __forceinline__ real_t wall_term(int i, const real_t* __restrict__ wall_u,
                                            size_t s, size_t n_nodes)
{
    return (real_t)6 * weight(i) * (cx(i) * wall_u[s] + cy(i) * wall_u[n_nodes + s]);
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
