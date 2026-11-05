# Phase 1 results: blocked GEMM kernel and transpose-free matmul paths

Verdict: **SHIP.** Native (`-DNN_NATIVE=ON`) cnn_mnist is 2.17x and mlp_wide 3.52x faster than the pre-change code (bar: more than 5%). The portable build (`-DNN_NATIVE=OFF`) is faster on every guarded row (cnn_mnist 1.99x, mlp_wide 2.31x, mlp_adam 2.33x, cnn_large 1.76x, infer_nograd_b1 10.8x, infer_nograd_b256 1.15x). `ctest` passes in both build dirs (171/171), and so do `nn_tests` (4581 assertions) and `bench_tests` (39).

Reference: commit b7856c8 (main). New: branch `perf/phase1-matmul` at a340e2c (after the zero-initialisation fix for the edge-tile buffers). Both built Release.

## History

The first measurement (commit d8804a6) recorded DROP: the native build gained 2.17x / 3.60x, but the portable build used the scalar kernel and was slower than the old loop (mlp_wide 0.70x, cnn_large 0.86x, mlp_adam 0.70x, infer_nograd_b256 0.65x). Task 6 added an SSE2 microkernel (commit 5a66d52) used when AVX2/FMA is absent on x86-64. This document is the final measurement with that kernel and replaces the earlier one.

## What shipped

- `include/nn/kernels/gemm.hpp`, `src/kernels/gemm.cpp`: cache-blocked, packed, single-threaded `sgemm` with transpose flags. Microkernels: AVX2/FMA (only with `-DNN_NATIVE=ON`), SSE2 (plain x86-64), scalar (other targets).
- `Tensor::matmul` routes through `sgemm`; new op `matmul_bt`; Linear and conv2d backward use the transpose flags instead of copies.

## Method

Machine: i5-12450HX, cpus 0-7 are the P-core threads, `powersave` governor, no sudo. Protocol: `NN_NUM_THREADS=8 taskset -c 0-7 apps/bench --out <csv> --label pinned`, 15 runs per row. Old and new alternated three times each (old1, new1, old2, new2, old3, new3), native first, then portable, one bench at a time. Before the runs: load average 1.35 right after the builds (1 min), gnome-shell 5.9%, claude 4.3%, ptyxis 2.1%. CSVs: `docs/baseline/phase1/{old,new}-{native,portable}-run<k>.csv`. Tables use the median of three per side. Ratio = old/new total_ms (matmul: new/old GFLOP/s), above 1 is faster. "noise" below is the larger median spread of the two sides.

### Old-vs-old stability and the outlier

The stop gate was 10% pairwise between old runs on cnn_mnist or mlp_wide total_ms.

| set | runs (ms) | max/min |
|---|---|---|
| old native cnn_mnist | 17.491, 17.481, 17.850 | +2.1% |
| old native mlp_wide | 167.153, 167.578, **184.366** | **+10.3%** |
| old portable cnn_mnist | 18.209, 18.348, 17.890 | +2.6% |
| old portable mlp_wide | 201.502, 206.282, 194.437 | +6.1% |
| new native cnn_mnist | 8.023, 8.581, 8.055 | +6.9% |
| new native mlp_wide | 47.614, 48.321, 47.451 | +1.8% |
| new portable cnn_mnist | 9.151, 9.153, 9.530 | +4.1% |
| new portable mlp_wide | 87.087, 86.943, 90.079 | +3.6% |

Old native mlp_wide run3 exceeded the 10% gate by 0.3 points, on one run. The controller ruled to accept the median of three and continue. The median (167.578) is unaffected by the outlier, and the measured effects (2x to 3.5x) are far above the 5% bar, so no observed noise level changes the verdict.

## Native (medians of three)

Times in ms. fwd/bwd/opt are the forward, backward and optimizer parts.

| row | old fwd/bwd/opt | new fwd/bwd/opt | old total | new total | speedup | noise |
|---|---|---|---|---|---|---|
| xor | 0.005/0.007/0.000 | 0.004/0.004/0.000 | 0.012 | 0.009 | 1.31x | 0.126 (within noise) |
| mlp_wide | 65.600/100.161/1.515 | 20.480/25.555/1.462 | 167.578 | 47.614 | 3.52x | 0.008 |
| mlp_adam | 62.938/99.109/5.146 | 18.334/21.350/5.200 | 167.255 | 45.597 | 3.67x | 0.011 |
| cnn_mnist | 3.518/13.908/0.054 | 4.358/3.613/0.060 | 17.491 | 8.055 | 2.17x | 0.310 |
| cnn_large | 328.039/681.874/0.120 | 274.651/224.944/0.139 | 1010.310 | 501.187 | 2.02x | 0.032 |
| infer_nograd_b1 | 1.123/0/0 | 0.082/0/0 | 1.123 | 0.082 | 13.72x | 0.012 |
| infer_nograd_b256 | 25.560/0/0 | 21.924/0/0 | 25.560 | 21.924 | 1.17x | 0.013 |

Matmul sweep (square n x n, GFLOP/s; the single-core peak of about 140 is an estimate, not a measurement):

| n | old | new | speedup | new % of 140 | noise |
|---|---|---|---|---|---|
| 8 | 7.0 | 4.6 | 0.66x | 3% | 0.130 (within noise) |
| 16 | 14.9 | 26.0 | 1.75x | 19% | 0.055 |
| 32 | 23.3 | 49.2 | 2.11x | 35% | 0.045 |
| 64 | 42.7 | 69.7 | 1.63x | 50% | 0.015 |
| 128 | 34.7 | 96.8 | 2.79x | 69% | 0.004 |
| 256 | 38.2 | 107.7 | 2.82x | 77% | 0.008 |
| 512 | 30.6 | 106.4 | 3.47x | 76% | 0.013 |
| 1024 | 24.5 | 104.2 | 4.26x | 74% | 0.012 |
| 2048 | 14.5 | 104.6 | 7.22x | 75% | 0.020 |

matmul/8: an 8x8 product is too small for packing to pay off, and no workload uses it.

## Portable (medians of three)

| row | old fwd/bwd/opt | new fwd/bwd/opt | old total | new total | speedup | noise |
|---|---|---|---|---|---|---|
| xor | 0.005/0.007/0.000 | 0.004/0.004/0.000 | 0.012 | 0.009 | 1.38x | 0.062 |
| mlp_wide | 79.364/120.000/1.553 | 35.672/49.915/1.526 | 201.502 | 87.087 | 2.31x | 0.008 |
| mlp_adam | 74.890/115.937/5.148 | 32.980/45.858/5.150 | 195.897 | 84.034 | 2.33x | 0.015 |
| cnn_mnist | 3.825/14.280/0.052 | 4.681/4.443/0.059 | 18.209 | 9.153 | 1.99x | 0.258 |
| cnn_large | 376.033/746.214/0.116 | 320.868/316.194/0.132 | 1123.370 | 637.286 | 1.76x | 0.006 |
| infer_nograd_b1 | 1.163/0/0 | 0.108/0/0 | 1.163 | 0.108 | 10.80x | 0.009 |
| infer_nograd_b256 | 28.479/0/0 | 24.816/0/0 | 28.479 | 24.816 | 1.15x | 0.012 |

Worst guarded portable row: infer_nograd_b256 at 1.15x (faster).

| n | old GFLOP/s | new GFLOP/s | speedup | noise |
|---|---|---|---|---|
| 8 | 6.1 | 4.6 | 0.74x | 0.130 (within noise) |
| 16 | 11.3 | 16.4 | 1.45x | 0.055 |
| 32 | 15.7 | 22.6 | 1.43x | 0.010 |
| 64 | 23.8 | 33.9 | 1.43x | 0.046 |
| 128 | 23.6 | 36.5 | 1.55x | 0.007 |
| 256 | 22.8 | 38.6 | 1.69x | 0.016 |
| 512 | 21.3 | 38.9 | 1.83x | 0.005 |
| 1024 | 19.5 | 39.0 | 2.00x | 0.007 |
| 2048 | 10.6 | 38.3 | 3.60x | 0.005 |

matmul/8: same note as native, an 8x8 product is too small for packing to pay off and no workload uses it.

## Exit rule

- Native cnn_mnist (2.17x) and mlp_wide (3.52x) each improve by more than 5%: pass.
- Portable not slower on cnn_mnist, mlp_wide, mlp_adam, cnn_large, infer_nograd_b1, infer_nograd_b256: pass (1.15x to 10.8x faster).
- ctest in both build dirs: pass (171/171 each).

## Real MNIST (`taskset -c 0-7 train_mnist data/mnist`, unseeded, tolerance 98.0%)

| build | test accuracy | wall time | Phase 0 (accuracy, wall) |
|---|---|---|---|
| native | 98.25% | 0:38.79 | 98.11%, 1:42.46 |
| portable | 98.43% | 0:46.31 | 98.47%, 1:47.72 |

Raw output in `docs/baseline/phase1/mnist-{native,portable}.txt`. The native wall time is longer than the earlier Phase 1 run (0:33.13); the machine was busier at that point (load average 2.2 at start), so treat it as noisy.

## Known caveats

- **cnn_mnist forward is slower than before** (about 4.3 ms vs 3.5 ms native, 4.7 vs 3.8 portable). This is not GEMM: the forward matmuls take about 0.5 ms of it. It comes from glibc heap trim and regrow (minor page faults roughly 28k to 54k per `perf stat`); with `GLIBC_TUNABLES` pinning the allocator, the new forward was about 2.1 ms vs 3.6 ms old. A later allocation-reuse phase should address it. Backward more than compensates (13.9 to 3.6 ms), so the total is still 2.17x.
- **Non-x86 targets** still use the scalar microkernel (about 10 GFLOP/s), which is slower than the old loop (20 to 24). They are not covered by this guard (a NEON kernel would be the fix).
- `micro_sse2` is compiled only when AVX2/FMA is absent, so the native build's tests do not run it, and vice versa; each is exercised by its own build's `ctest`.
- cnn_mnist per-run spread is high (0.26 to 0.31) in all sets; xor (microseconds) is noise.
- A fresh `perf` profile belongs at the start of Phase 2.
