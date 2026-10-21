// Fused pull-stream + bounce-back + collide with shared-memory staging.
//
// The block (TILE_X x TILE_Y threads) first copies, for every direction, its
// tile of populations plus a one-node halo into shared memory, then each
// thread pulls its neighbours' values from shared memory instead of global.
//
// History: on early CUDA GPUs (no L1 cache for global loads) a load shifted by
// one element cost a second memory transaction, so staging through shared
// memory (Toelke 2008) roughly doubled LBM performance. Since Fermi, the L1/L2
// caches absorb those shifted loads, and the halo makes every block load
// (TILE_X+2)(TILE_Y+2)/(TILE_X TILE_Y) - 1 = 33 % more data (32 x 8 tile).
// This kernel exists to measure whether staging still pays off; the benchmark
// answers that, not this comment.
//
// TILE_X and TILE_Y are #defined by CudaSolver and must equal blockDim.

extern "C" __global__ void stream_collide_tiled(
    const real_t* __restrict__ f_src,
    real_t* __restrict__ f_dst,
    const unsigned char* __restrict__ solid,
    const real_t* __restrict__ wall_u,
    const int nx, const int ny,
    const real_t omega_plus, const real_t omega_minus,
    const real_t force_x, const real_t force_y)
{
    const Relaxation r{omega_plus, omega_minus, force_x, force_y};
    constexpr int HX = TILE_X + 2;  // tile width including the halo
    constexpr int HY = TILE_Y + 2;
    __shared__ real_t tile[Q][HY][HX];
    __shared__ unsigned char solid_tile[HY][HX];

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int x0 = blockIdx.x * TILE_X, y0 = blockIdx.y * TILE_Y;
    const size_t n_nodes = (size_t)nx * ny;

    // Cooperative load: every thread copies a strided share of the halo tile.
    // Out-of-domain threads still help load and still reach the barrier.
    // positive_mod handles domains smaller than a tile (offsets beyond nx).
    for (int k = ty * TILE_X + tx; k < HX * HY; k += TILE_X * TILE_Y) {
        const int hy = k / HX, hx = k % HX;
        const size_t g = (size_t)positive_mod(y0 + hy - 1, ny) * nx + positive_mod(x0 + hx - 1, nx);
        solid_tile[hy][hx] = solid[g];
#pragma unroll
        for (int i = 0; i < Q; ++i) tile[i][hy][hx] = f_src[i * n_nodes + g];
    }
    __syncthreads();

    const int x = x0 + tx, y = y0 + ty;
    if (x >= nx || y >= ny || solid_tile[ty + 1][tx + 1]) return;

    real_t f[Q];
#pragma unroll
    for (int i = 0; i < Q; ++i) {
        const int sx = tx + 1 - cx(i), sy = ty + 1 - cy(i);
        if (solid_tile[sy][sx]) {
            const size_t src = (size_t)wrap(y - cy(i), ny) * nx + wrap(x - cx(i), nx);
            f[i] = tile[opp(i)][ty + 1][tx + 1] + wall_term(i, wall_u, src, n_nodes);
        } else {
            f[i] = tile[i][sy][sx];
        }
    }

    collide_node(f, r);

    const size_t node = (size_t)y * nx + x;
#pragma unroll
    for (int i = 0; i < Q; ++i) f_dst[i * n_nodes + node] = f[i];
}
