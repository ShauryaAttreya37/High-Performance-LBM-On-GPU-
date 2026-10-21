// AA access pattern (Bailey et al. 2009): one lattice, updated in place.
//
// Two kernels alternate. Write F[k](x) for slot k of node x in the single
// buffer. Between launches the buffer is in one of two layouts:
//
//   after an even step:  F[opp(i)](x) = f*_i(x)      post-collision, reversed slots
//   after an odd step:   F[i](x)      = f_i(x)       streamed, ready to collide
//
// even step: read F[i](x), collide, write f*_i to F[opp(i)](x).
//            Purely node-local; streaming is implicit in the slot reversal.
// odd step:  read f_i(x) = F[opp(i)](x - c_i)           (pull from neighbour)
//            collide
//            write f*_i to F[i](x + c_i)                (push to neighbour)
//
// Race freedom of the odd step. Thread x reads slot (opp(i), x - c_i) and
// writes slot (i, x + c_i). Take any slot (k, y):
//   it is read by the thread with opp(i) = k, x = y + c_i = y - c_k;
//   it is written by the thread with i = k,   x = y - c_i = y - c_k.
// Same thread. Every thread loads all nine values into registers before it
// stores anything, so updating the single buffer in place is safe.
//
// Bounce-back in the odd step:
//   load : x - c_i solid  ->  f_i(x) = F[i](x) + wall term   (F[i](x) holds f*_opp(i)(x))
//   store: x + c_i solid  ->  reflected value goes to F[opp(i)](x) with the
//          wall term for direction opp(i), ready for the next even step.
// Those slots would normally belong to the solid node's thread, which exits
// immediately, so again exactly one thread touches each slot.
//
// Memory: one buffer instead of two, and the even step's traffic is
// perfectly aligned and node-local.

extern "C" __global__ void aa_even(
    real_t* __restrict__ F,
    const unsigned char* __restrict__ solid,
    const int nx, const int ny,
    const real_t omega_plus, const real_t omega_minus,
    const real_t force_x, const real_t force_y)
{
    const Relaxation r{omega_plus, omega_minus, force_x, force_y};
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= nx || y >= ny) return;
    const size_t n_nodes = (size_t)nx * ny;
    const size_t node = (size_t)y * nx + x;
    if (solid[node]) return;

    real_t f[Q];
#pragma unroll
    for (int i = 0; i < Q; ++i) f[i] = F[i * n_nodes + node];

    collide_node(f, r);

#pragma unroll
    for (int i = 0; i < Q; ++i) F[opp(i) * n_nodes + node] = f[i];
}

extern "C" __global__ void aa_odd(
    real_t* __restrict__ F,
    const unsigned char* __restrict__ solid,
    const real_t* __restrict__ wall_u,
    const int nx, const int ny,
    const real_t omega_plus, const real_t omega_minus,
    const real_t force_x, const real_t force_y)
{
    const Relaxation r{omega_plus, omega_minus, force_x, force_y};
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= nx || y >= ny) return;
    const size_t n_nodes = (size_t)nx * ny;
    const size_t node = (size_t)y * nx + x;
    if (solid[node]) return;

    real_t f[Q];
#pragma unroll
    for (int i = 0; i < Q; ++i) {
        const size_t src = (size_t)wrap(y - cy(i), ny) * nx + wrap(x - cx(i), nx);
        if (solid[src]) {
            f[i] = F[i * n_nodes + node] + wall_term(i, wall_u, src, n_nodes);
        } else {
            f[i] = F[opp(i) * n_nodes + src];
        }
    }

    collide_node(f, r);

#pragma unroll
    for (int i = 0; i < Q; ++i) {
        const size_t dst = (size_t)wrap(y + cy(i), ny) * nx + wrap(x + cx(i), nx);
        if (solid[dst]) {
            F[opp(i) * n_nodes + node] = f[i] + wall_term(opp(i), wall_u, dst, n_nodes);
        } else {
            F[i * n_nodes + dst] = f[i];
        }
    }
}
