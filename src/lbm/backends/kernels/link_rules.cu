// Link-rule pre-pass: evaluate affine boundary rules, then write them back.
//
// A rule r sets one population slot from a fixed combination of others:
//
//     F[target[r]] = constant[r] + sum_t coeff[r][t] * F[slot[r][t]]
//
// (see lbm.link_rules for what the rules are and how slots are chosen for
// each memory layout). Evaluation and write-back are two launches, so every
// rule reads the populations as they were before any rule wrote: rules never
// see each other's results, whatever the geometry. Both kernels touch only
// the sparse boundary links, so they cost little next to the main kernel.
//
// Needs only `real_t`; the lattice preamble is not involved.

extern "C" __global__ void rules_evaluate(
    const real_t* __restrict__ F,
    const long long* __restrict__ slot,
    const real_t* __restrict__ coeff,
    const real_t* __restrict__ constant,
    real_t* __restrict__ out,
    const int n_rules, const int n_terms)
{
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_rules) return;
    // Terms are stored term-major (slot[t * n_rules + r]): consecutive threads
    // read consecutive addresses of the slot and coefficient arrays.
    real_t value = constant[r];
    for (int t = 0; t < n_terms; ++t) {
        const size_t k = (size_t)t * n_rules + r;
        value += coeff[k] * F[slot[k]];
    }
    out[r] = value;
}

extern "C" __global__ void rules_write(
    real_t* __restrict__ F,
    const long long* __restrict__ target,
    const real_t* __restrict__ value,
    const int n_rules)
{
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < n_rules) F[target[r]] = value[r];
}
