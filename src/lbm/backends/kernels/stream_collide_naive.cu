// Fused pull-stream + bounce-back + collide, two lattices (ping-pong).
//
// One thread per node. Global-memory traffic per fluid node and time step:
// 9 loads + 9 stores of real_t (72 bytes in float32) plus the solid-mask bytes,
// which are shared by neighbours and mostly hit in cache. This is the minimum
// for a two-lattice scheme, so the only question is how close to peak DRAM
// bandwidth the access pattern gets.
//
// Layout f[q][y][x] (structure of arrays). Threads of a warp have consecutive x,
// so for each q the warp reads 32 consecutive addresses (shifted by -cx), and
// writes 32 consecutive, aligned addresses. A one-element shift makes a load
// straddle two 32-byte sectors; the L1/L2 caches absorb most of that cost.

extern "C" __global__ void stream_collide_naive(
    const real_t* __restrict__ f_src,
    real_t* __restrict__ f_dst,
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
            // Halfway bounce-back: the population that left this node towards the wall.
            f[i] = f_src[opp(i) * n_nodes + node] + wall_term(i, wall_u, src, n_nodes);
        } else {
            f[i] = f_src[i * n_nodes + src];
        }
    }

    collide_node(f, r);

#pragma unroll
    for (int i = 0; i < Q; ++i) f_dst[i * n_nodes + node] = f[i];
}
