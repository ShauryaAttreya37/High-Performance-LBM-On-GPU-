// Fused pull-stream + bounce-back + collide, two lattices (ping-pong), 2D or 3D.
//
// One thread per node. Global-memory traffic per fluid node and time step:
// Q loads + Q stores of real_t (72 bytes for D2Q9, 152 for D3Q19, float32)
// plus the solid-mask bytes, which are shared by neighbours and mostly hit in
// cache. This is the minimum for a two-lattice scheme, so the only question is
// how close to peak DRAM bandwidth the access pattern gets.
//
// Layout f[q][z][y][x] (structure of arrays; 2D is nz = 1). Threads of a warp
// have consecutive x, so for each q the warp reads 32 consecutive addresses
// (shifted by -c_i) and writes 32 consecutive, aligned addresses. A
// one-element shift makes a load straddle two 32-byte sectors; the L1/L2
// caches absorb most of that cost.

extern "C" __global__ void stream_collide_naive(
    const real_t* __restrict__ f_src,
    real_t* __restrict__ f_dst,
    const unsigned char* __restrict__ solid,
    const real_t* __restrict__ wall_u,
    const real_t* __restrict__ force_field,
    const int nx, const int ny, const int nz,
    const real_t omega_plus, const real_t omega_minus,
    const real_t force_x, const real_t force_y, const real_t force_z)
{
    Relaxation r{omega_plus, omega_minus, force_x, force_y, force_z};
    Node n;
    if (!locate(n, nx, ny, nz)) return;
    const size_t n_nodes = (size_t)nx * ny * nz;
    const unsigned char kind = solid[n.index];
    if (kind == SOLID) return;
    if (kind == FORCED) add_local_force(r, force_field, n.index, n_nodes);

    real_t f[Q];
#pragma unroll
    for (int i = 0; i < Q; ++i) {
        const size_t src = neighbour<-1>(i, n, nx, ny, nz);
        if (solid[src] == SOLID) {
            // Halfway bounce-back: the population that left this node towards the wall.
            f[i] = f_src[opp(i) * n_nodes + n.index] + wall_term(i, wall_u, src, n_nodes);
        } else {
            f[i] = f_src[i * n_nodes + src];
        }
    }

    collide_node(f, r);

#pragma unroll
    for (int i = 0; i < Q; ++i) f_dst[i * n_nodes + n.index] = f[i];
}
