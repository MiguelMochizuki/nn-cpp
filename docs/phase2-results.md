# Phase 2 results: thread pool and parallel sgemm

Verdict: **SHIP.** Native (`-DNN_NATIVE=ON`, `NN_NUM_THREADS=8 taskset -c 0-7`) mlp_wide is 1.40x and cnn_mnist 1.36x faster than the pre-change code (bars: mlp_wide more than 5%, cnn_mnist not slower than 3%). Portable is faster on every guarded row (mlp_wide 1.95x, cnn_mnist 1.36x). `ctest` passes in both build dirs (187/187).

Reference: commit 609109d (main, Phase 1). New: branch `perf/phase2-threads`. Both Release, built in `build/base-{native,portable}` and `build/{native,portable}`.

## What shipped

- `nn/kernels/parallel.hpp`: lazily created thread pool, `parallel_for`, `num_threads`, `set_num_threads`; default `NN_NUM_THREADS` else `hardware_concurrency()`.
- `sgemm` parallelised in `blocked()`. Chunks are (kMC row block, run of kNR column panels), about two per thread; the B panel is packed once by the caller, each chunk packs its own A block and writes disjoint C. No cross-thread accumulation, so C is bitwise identical for any thread count (tested for 1, 2, 4, 8).
- `kParallelMin = 1<<20` multiply-adds (about a 101^3 product).
- bench applies `NN_NUM_THREADS` (before this, the CSV column only labelled the run).

## Diagnosis: why 8 threads was slower than 4

Machine: i5-12450HX, cpus 0-7 are 4 P-cores with their hyperthread siblings (0-1, 2-3, ...). sgemm 1024^3, GFLOP/s:

| setup | GFLOP/s |
|---|---|
| 1 thread | 107 |
| 4 threads on cpus 0,2,4,6 (one per physical core) | 350 |
| 4 threads on cpus 0-3 (2 cores, 2 HT each) | 187 |
| 4 threads, free within 0-7 | 352 |
| 8 threads on 0-7, row-only split (old) | 290 |
| 8 threads, m=768 (exactly 8 row blocks, no imbalance) | 362 vs 386 at 4 threads |

Two causes. (1) Hyperthread siblings share one core's FMA units: with perfect balance (m=768) 8 threads gave 362 against 386 for 4, so the siblings add nothing for this kernel. (2) Imbalance: m=1152 has 12 row blocks, so with 8 threads some take 2 and some 1 (280 vs 386 for 4 threads). mlp_wide has m=256 (3 blocks) and m=784 (9 blocks), so most threads idled.

Fix: also split the columns when there are fewer than two row blocks per thread. Tuned values (kChunksPerThread=2, kMinPanels=4; 1 to 4 chunks and 2 to 8 panels tried), 8 threads pinned 0-7, GFLOP/s old -> new: 256x1024x784 242 -> 311, 256x784x1024 220 -> 320, 1152x1024x1024 290 -> 385, 1024^3 278 -> 343. Scheduling cannot pass the 4-physical-core limit (about 350 to 390).

## Headline (medians of three, alternated old/new, `NN_NUM_THREADS=8 taskset -c 0-7`, 15 runs per row)

Ratio = old/new total_ms. Max/min spread of the three runs is in brackets. Old-vs-old spread on mlp_wide and cnn_mnist is at most 1.1%, below the 3% gate. Load average before runs 0.6 to 1.6 (`load.txt`).

| row | native old ms | native new ms | speedup | portable old ms | portable new ms | speedup |
|---|---|---|---|---|---|---|
| xor | 0.008 | 0.009 | noise (spread 14 to 22%) | 0.008 | 0.008 | noise |
| mlp_wide | 47.496 (0.3%) | 33.829 (2.2%) | 1.40x | 87.029 | 44.733 | 1.95x |
| mlp_adam | 44.096 | 30.480 | 1.45x | 84.006 | 41.085 | 2.04x |
| cnn_mnist | 8.047 (1.1%) | 5.931 (0.8%) | 1.36x | 9.189 | 6.734 | 1.36x |
| cnn_large | 495.193 | 470.892 | 1.05x | 635.876 | 518.796 | 1.23x |
| infer_nograd_b1 | 0.082 | 0.079 | 1.04x | 0.108 | 0.106 | 1.02x |
| infer_nograd_b256 | 22.111 | 22.127 | 1.00x | 24.948 | 24.307 | 1.03x |

Matmul sweep, GFLOP/s old -> new:

| n | native | portable |
|---|---|---|
| 128 | 98 -> 127 | 37 -> 59 |
| 256 | 108 -> 252 | 39 -> 108 |
| 512 | 106 -> 347 | 39 -> 137 |
| 1024 | 106 -> 329 | 39 -> 128 |
| 2048 | 102 -> 330 | 38 -> 129 |

## Thread scaling (new code; pinned 0-7 except t12 unpinned = hardware_concurrency)

Native, total ms for workloads, GFLOP/s for matmul:

| row | t1 | t2 | t4 | t8 | t12 |
|---|---|---|---|---|---|
| mlp_wide | 47.7 | 37.9 | 33.7 | 33.6 | 33.8 |
| mlp_adam | 44.3 | 35.2 | 30.4 | 30.3 | 30.5 |
| cnn_mnist | 8.0 | 5.6 | 5.8 | 5.9 | 7.8 |
| cnn_large | 495.8 | 475.2 | 463.6 | 465.5 | 463.3 |
| matmul 1024 | 108 | 184 | 339 | 335 | 334 |
| matmul 2048 | 104 | 195 | 333 | 330 | 257 |

Portable:

| row | t1 | t2 | t4 | t8 | t12 |
|---|---|---|---|---|---|
| mlp_wide | 87.2 | 57.7 | 45.8 | 44.5 | 38.8 |
| mlp_adam | 84.0 | 54.3 | 42.4 | 41.0 | 34.4 |
| cnn_mnist | 9.1 | 6.5 | 6.7 | 6.7 | 8.5 |
| cnn_large | 633.8 | 551.0 | 507.9 | 518.9 | 500.4 |
| matmul 1024 | 39 | 72 | 127 | 129 | 196 |
| matmul 2048 | 38 | 74 | 130 | 130 | 193 |

Four threads on cpus 0,2,4,6 (one per physical core, native, median of 3): mlp_wide 33.6, mlp_adam 30.4, cnn_mnist 5.8, cnn_large 463.7 ms, equal to 8 threads on 0-7.

Thread policy: the default stays `hardware_concurrency()` (12 here, including the 4 E-cores). Native gets nothing from threads beyond the 4 physical P-cores; cnn_mnist is 30% slower at 12 threads than at 4 to 8 (7.8 vs 5.8 ms, though still no slower than 1 thread) and matmul 2048 is 22% slower. The portable kernel (not FMA-bound) does gain from 12 threads (mlp_wide 38.8 vs 44.5). Whether to cap the default at physical cores is left as a decision.

## Exit rule

- Native mlp_wide 1.40x (bar 1.05x): pass. cnn_mnist 1.36x (bar: not slower than 3%): pass.
- Portable not slower on any guarded row: pass.
- ctest in both build dirs: pass (187/187).

## Real MNIST (native, `NN_NUM_THREADS=8 taskset -c 0-7 train_mnist data/mnist`)

Test accuracy 98.09% (tolerance 98.0%), wall 36.2 s (Phase 1: 98.25%, 38.8 s). Raw output in `docs/baseline/phase2/mnist-native.txt`. The CNN's matmuls are small, so the run is mostly serial (109% CPU).

## Caveats

- cnn_mnist and cnn_large are dominated by memory traffic and allocation, not GEMM; their gain is limited (cnn_large 1.05x native). Elementwise ops, im2col and allocation reuse are Phase 3.
- B packing is serial on the caller thread.
- The timings were taken in a short window on a lightly loaded desktop (load average about 1); xor is noise.
- An existing flaky test (test_nn.cpp, Linear-ReLU-Linear gradcheck) fails about once in 20 to 60 runs: unseeded weights can put a ReLU input near its kink. Unrelated to this phase and not fixed here.
