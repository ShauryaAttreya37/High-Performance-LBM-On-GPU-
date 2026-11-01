"""Throughput (MLUPS) and achieved memory bandwidth of every backend.

LBM is memory-bound: each node update does a few hundred floating-point
operations but must move at least Q populations in and Q out. The honest
performance metric is therefore how close a kernel gets to the GPU's DRAM
bandwidth:

    MLUPS                = million lattice-node updates per second
    bytes per update     = 2 * Q * sizeof(real_t)     (float32: 72 B D2Q9, 152 B D3Q19)
    achieved bandwidth   = MLUPS * 1e6 * bytes per update
    peak bandwidth       = measured device-to-device copy (read + write)

A laptop GPU changes its memory clock with its power state, so the copy peak
itself moves between runs (by ~25 % on the RTX 3050). The peak is therefore
re-measured right before every kernel measurement, and the fraction of that
peak is the number to compare.

Grids small enough to fit in the L2 cache can exceed DRAM bandwidth; the
table flags them, since that number says nothing about large problems.

Usage::

    python benchmarks/bench_mlups.py
    python benchmarks/bench_mlups.py --sizes 512 1024 2048 --dtype float64 --plot
    python benchmarks/bench_mlups.py --backends numpy cupy cuda-aa --sizes 256 512
    python benchmarks/bench_mlups.py --dim 3 --backends cuda-naive cuda-aa --sizes 64 128 256

Results go to ``outputs/benchmarks/`` (CSV, and a PNG with ``--plot``).
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from lbm import SimulationParams, make_solver
from lbm.backends import BACKENDS, available_backends, import_cupy
from lbm.backends.cuda_solver import TWO_D_ONLY
from lbm.lattice import default_lattice

OUTPUT = Path("outputs/benchmarks")
TARGET_SECONDS = 1.0
"""Each measurement runs long enough to take about this long."""


@dataclass(frozen=True)
class Measurement:
    backend: str
    dtype: str
    dim: int
    size: int
    steps: int
    seconds: float
    mlups: float
    bandwidth_gbs: float
    peak_gbs: float
    fraction_of_peak: float
    fits_in_l2: bool


def peak_copy_bandwidth(nbytes: int = 512 * 2**20, repeats: int = 20) -> float:
    """Device-to-device copy bandwidth in GB/s (bytes read + bytes written)."""
    cp = import_cupy()
    src = cp.ones(nbytes // 4, dtype=cp.float32)
    dst = cp.empty_like(src)
    cp.copyto(dst, src)
    start, stop = cp.cuda.Event(), cp.cuda.Event()
    start.record()
    for _ in range(repeats):
        cp.copyto(dst, src)
    stop.record()
    stop.synchronize()
    seconds = cp.cuda.get_elapsed_time(start, stop) / 1e3
    return 2 * nbytes * repeats / seconds / 1e9


def l2_cache_bytes() -> int:
    cp = import_cupy()
    return int(cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)["l2CacheSize"])


def _release_gpu_memory(backend: str) -> None:
    """Return cached blocks to the driver so the next, larger grid can allocate."""
    if backend != "numpy":
        import_cupy().get_default_memory_pool().free_all_blocks()


def _out_of_memory_error(backends) -> type[Exception]:
    if any(b != "numpy" for b in backends):
        return import_cupy().cuda.memory.OutOfMemoryError
    return MemoryError


def _synchronize(solver) -> None:
    if solver.xp is not np:
        solver.xp.cuda.get_current_stream().synchronize()


def _timed_run(solver, steps: int) -> float:
    _synchronize(solver)
    start = time.perf_counter()
    solver.run(steps, check_every=steps)
    _synchronize(solver)
    return time.perf_counter() - start


def grid_params(size: int, dim: int, dtype: str) -> SimulationParams:
    """Fully periodic size^dim grid."""
    sizes = {"nx": size, "ny": size} | ({"nz": size} if dim == 3 else {})
    return SimulationParams(**sizes, tau=0.8, collision="trt", dtype=dtype)


def measure(backend: str, size: int, dim: int, dtype: str, l2_bytes: int) -> Measurement:
    """Benchmark one backend on a fully periodic size^dim grid."""
    params = grid_params(size, dim, dtype)
    solver = make_solver(params, backend=backend)
    rng = np.random.default_rng(0)
    velocity = (0.01 * rng.standard_normal(params.shape) for _ in range(dim))
    solver.initialize(1.0, *velocity)

    # Warm up (kernel compilation, caches), then size the timed run to ~TARGET_SECONDS.
    probe = 2 * solver.step_multiple
    _timed_run(solver, probe)
    per_step = _timed_run(solver, probe) / probe
    steps = solver.step_multiple * max(1, round(TARGET_SECONDS / per_step / solver.step_multiple))
    peak_gbs = peak_copy_bandwidth() if backend != "numpy" else 0.0
    seconds = _timed_run(solver, steps)

    del solver
    _release_gpu_memory(backend)

    q = params.lattice.q
    nodes = size**dim
    itemsize = np.dtype(dtype).itemsize
    mlups = nodes * steps / seconds / 1e6
    bandwidth = mlups * 1e6 * 2 * q * itemsize / 1e9
    return Measurement(
        backend=backend,
        dtype=dtype,
        dim=dim,
        size=size,
        steps=steps,
        seconds=seconds,
        mlups=mlups,
        bandwidth_gbs=bandwidth,
        peak_gbs=peak_gbs,
        fraction_of_peak=bandwidth / peak_gbs if peak_gbs else float("nan"),
        fits_in_l2=q * nodes * itemsize <= l2_bytes,
    )


def write_csv(results: list[Measurement], path: Path) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(results[0])))
        writer.writeheader()
        writer.writerows(asdict(r) for r in results)


def plot(results: list[Measurement], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_mlups, ax_bw) = plt.subplots(1, 2, figsize=(11, 4.2))
    for backend in dict.fromkeys(r.backend for r in results):
        rows = [r for r in results if r.backend == backend]
        sizes = [r.size for r in rows]
        ax_mlups.plot(sizes, [r.mlups for r in rows], "o-", label=backend)
        ax_bw.plot(sizes, [r.bandwidth_gbs for r in rows], "o-", label=backend)
    peaks = [r.peak_gbs for r in results if r.peak_gbs]
    if peaks:
        ax_bw.axhspan(min(peaks), max(peaks), color="k", alpha=0.12, label="copy peak (range)")
    dim = results[0].dim
    for ax in (ax_mlups, ax_bw):
        ax.set_xscale("log", base=2)
        ax.set_xlabel(f"grid size N ({' x '.join(['N'] * dim)} nodes)")
        ax.grid(alpha=0.3)
    ax_mlups.set(ylabel="MLUPS", title="Throughput")
    ax_mlups.set_yscale("log")
    ax_bw.set(ylabel="effective bandwidth [GB/s]", title="Memory bandwidth (memory roofline)")
    ax_bw.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main(argv=None) -> None:
    gpu_backends = [b for b in BACKENDS if b.startswith("cuda") or b == "cupy"]
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backends", nargs="+", default=gpu_backends, choices=BACKENDS)
    parser.add_argument("--dim", type=int, default=2, choices=(2, 3))
    parser.add_argument("--sizes", type=int, nargs="+", default=None)
    parser.add_argument("--dtype", default="float32", choices=("float32", "float64"))
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args(argv)
    if args.sizes is None:
        args.sizes = [256, 512, 1024, 2048, 4096] if args.dim == 2 else [64, 96, 128, 192, 256]
    if args.dim == 3:
        skipped = [b for b in args.backends if b.removeprefix("cuda-") in TWO_D_ONLY]
        if skipped:
            print(f"skipping 2D-only backends in 3D: {skipped}")
        args.backends = [b for b in args.backends if b not in skipped]

    unavailable = set(args.backends) - set(available_backends())
    if unavailable:
        raise SystemExit(f"backends not available on this machine: {sorted(unavailable)}")

    has_gpu = any(b != "numpy" for b in args.backends)
    l2 = l2_cache_bytes() if has_gpu else 0
    if has_gpu:
        cp = import_cupy()
        name = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)["name"].decode()
        print(f"{name}: L2 cache {l2 / 2**20:.1f} MiB, {default_lattice(args.dim).name}")

    for backend in (b for b in args.backends if b.startswith("cuda")):
        tiny = make_solver(grid_params(8, args.dim, args.dtype), backend)
        for name, res in tiny.kernel_resources().items():
            print(
                f"  {name:>22}: {res['registers']:3d} registers/thread, "
                f"{res['local_bytes']} B local (spill), {res['shared_bytes']} B shared/block"
            )

    out_of_memory = _out_of_memory_error(args.backends)
    results = []
    print(f"{'backend':>11} {'N':>6} {'MLUPS':>9} {'GB/s':>8} {'peak':>7} {'% peak':>7}  note")
    for backend in args.backends:
        for size in args.sizes:
            try:
                r = measure(backend, size, args.dim, args.dtype, l2)
            except out_of_memory:
                _release_gpu_memory(backend)
                print(f"{backend:>11} {size:6d}  skipped: out of device memory")
                continue
            results.append(r)
            note = "lattice fits in L2" if r.fits_in_l2 else ""
            print(
                f"{backend:>11} {size:6d} {r.mlups:9.1f} {r.bandwidth_gbs:8.1f} "
                f"{r.peak_gbs:7.1f} {100 * r.fraction_of_peak:6.1f}%  {note}"
            )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    tag = "_".join([f"{args.dim}d", args.dtype, *args.backends])
    csv_path = OUTPUT / f"mlups_{tag}.csv"
    write_csv(results, csv_path)
    print(f"results written to {csv_path}")
    if args.plot:
        png_path = OUTPUT / f"bandwidth_{tag}.png"
        plot(results, png_path)
        print(f"figure written to {png_path}")


if __name__ == "__main__":
    main()
