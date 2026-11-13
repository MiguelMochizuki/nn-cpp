# Performance reports

Measurements behind the speedups quoted in the main README. Machine: Intel i5-12450HX; headline protocol `NN_NUM_THREADS=8 taskset -c 0-7`, medians of three runs, 3% noise gate. Raw CSVs, profiles and MNIST runs are in [baseline/](baseline/): the top level holds the Phase 0 baseline, subdirectories hold later runs.

| report | what it covers |
|---|---|
| [phase0-results.md](phase0-results.md) | baseline numbers, first profile and phase ranking |
| [phase1-results.md](phase1-results.md) | blocked GEMM kernel and transpose-free matmul paths |
| [phase2-results.md](phase2-results.md) | thread pool and parallel sgemm |
| [phase3-results.md](phase3-results.md) | fewer copies, elementwise fast paths, parallel im2col and max-pooling |
| [followups-results.md](followups-results.md) | flaky-test fix, thread-default measurement, parallel B packing, zero-fill and backward allocations |
| [v1.1.0-results.md](v1.1.0-results.md) | guard against v1.0.0 and the `cnn_fer` workload |
| [v2.0.0 against PyTorch](#v200-against-pytorch) | fused ops, per-sample convolution, uninitialised storage; comparison with PyTorch 2.15 CPU |

## v1.3.0: `transformer_tiny` and the v1.2.0 guard

`transformer_tiny` is a decoder-only `Transformer` (vocabulary 64, width 64, 4 heads, 2 layers, 64 positions) on a batch of 16 x 64 token ids with `cross_entropy_loss` over the 1024 flattened positions and AdamW at 1e-3. Protocol: `NN_NUM_THREADS=8 taskset -c 0-7`, CPU governor `performance`, 3 passes of `bench --runs 50 --warmup 5`, medians of the 3 passes:

| build | forward ms | backward ms | optim ms | total ms | run spread |
|---|---|---|---|---|---|
| native | 24.3 | 23.8 | 0.35 | 48.8 | 3% |
| portable | 29.0 | 31.7 | 0.35 | 61.2 | 2% |

Roughly half of a step is forward, half backward and under 1% the optimizer. A `perf record` profile of the whole run (`apps/bench/profile.sh build/<native|portable> transformer_tiny <dir>`, `perf_event_paranoid=1`; raw output in [baseline/v1.3.0/](baseline/v1.3.0/)) shows no single dominant op. Shares as printed by `perf report`, native build, grouped by hand:

| group | share | symbols |
|---|---|---|
| GEMM | about 14% | `micro_avx2` 10.5%, `blocked` 3.6% (portable: `micro_sse2` 21.5%) |
| GELU | about 12% | `tanhf` 7.6%, the `gelu` forward and backward lambdas 4.2% |
| `transpose` | about 7% | `Tensor::transpose` 7.2% |
| allocation and copies | about 14% | `memset` 4.2%, `memmove` 2.8%, `malloc`, `free`, `operator new` and the `Tensor` constructor 7.2% |
| other elementwise and reductions | about 18% | `Tensor::operator+=` 6.1%, `masked_fill` 5.4%, `Tensor::sum` 3.2%, `mul` 3.1% |

Softmax and LayerNorm have no row of their own: they are composed of `exp`, `sum`, `mul` and `div`, so their cost is spread over the groups above and this profile does not isolate it. What it does show is that the matrix multiplies are a small part of a step, and that `tanhf`, `transpose`, `masked_fill` (a serial per-element loop) and allocation each cost about as much as the GEMM kernel's inner loop. Cheaper candidates than a fused attention are therefore a vectorised `gelu`, a cheaper `transpose` for the head split and merge, a parallel or row-wise `masked_fill`, and fewer temporaries. None of this is implemented; the profile is one run on one machine and the shares are approximate.

**The guard run against v1.2.0 passes the 3% gate.** v1.2.0 and the 1.3.0 `main` were built side by side (native and portable) and run interleaved, 3 passes of `--runs 50 --warmup 5` per build. Medians of the 3 passes, new against old, total ms per step:

| workload | native | portable |
|---|---|---|
| `cnn_fer` | -0.4% | -0.1% |
| `cnn_large` | +0.1% | -0.1% |
| `cnn_mnist` | +0.1% | -0.5% |
| `mlp_adam` | -0.4% | -0.2% |
| `mlp_wide` | +0.1% | -0.3% |
| `infer_nograd_b256` | -2.1% | +0.3% |
| `infer_nograd_b1` | +0.4% | +5.0% (see below) |
| `xor` | -3.0% | +3.3% (see below) |

Run-to-run spreads were 1% to 6% on the workloads above, except `xor` (8% to 24%, a 5 microsecond step). Two portable rows read above 3%. `infer_nograd_b1` (93 microseconds) read +5.0% in the 3x50 passes, but a separate re-run of 5 interleaved old/new pairs of 300 runs gave medians of 0.0930 ms (v1.2.0) and 0.0935 ms (new), +0.5%, so the +5.0% did not reproduce. `xor` (about 5.5 microseconds per step) reads +3.3% with a 7% to 8% spread; one of its three new passes equals the old median, which makes it timer noise rather than a regression. An earlier attempt under the `powersave` governor was not usable (spreads of 25% to 250%, deltas flipping sign between builds) and is not quoted. This release only adds code (`git diff v1.2.0 -- src include` deletes no lines), and the measurement agrees.

Reproduce with `apps/bench/run_baseline.sh`; compare two result files with `python3 apps/bench/compare_csv.py before.csv after.csv`.

## v2.0.0 against PyTorch

Protocol: `NN_NUM_THREADS=8 taskset -c 0-7` (four P-cores with hyper-threading), CPU governor and platform profile `performance`, laptop on mains power (checked before and after each run), 3 interleaved passes of `--runs 50 --warmup 5` for the native build, the portable build and `apps/bench/bench_torch.py` (PyTorch 2.15, CPU wheel, eager mode, the same six workloads), median of the 3 passes. Raw CSVs: [baseline/v2.0.0/](baseline/v2.0.0/); the earlier PyTorch comparison, before these changes, is in [baseline/torch-before-2.0.0/](baseline/torch-before-2.0.0/).

| workload | before ms | PyTorch ms | native ms (x PyTorch) | portable ms (x PyTorch) |
|---|---|---|---|---|
| `mlp_wide` | 10.35 | 8.24 | 8.28 (1.00) | 21.15 (2.57) |
| `mlp_adam` | 11.35 | 9.80 | 8.81 (0.90) | 22.22 (2.27) |
| `cnn_mnist` | 1.19 | 1.29 | 0.59 (0.46) | 1.10 (0.85) |
| `cnn_fer` | 38.44 | 20.51 | 15.01 (0.73) | 24.61 (1.20) |
| `cnn_large` | 101.78 | 72.66 | 48.12 (0.66) | 85.95 (1.18) |
| `transformer_tiny` | 14.33 | 6.24 | 4.56 (0.73) | 8.56 (1.37) |

"Before" is the native build after the first round of fused ops and uninitialised storage but before the changes below; at v1.3.0 the same native build took 48.8 ms on `transformer_tiny`. Individual passes of the native build spread up to 13% on `mlp_wide` and 16% on `cnn_large` (one slow pass each, visible in the CSVs); the medians are stable to about 3%. The first comparison, before any of this work, had nn-cpp 1.7x to 7.8x slower than PyTorch on these workloads (native build).

What moved the numbers, each step measured as its own build, interleaved, on mains power (relative change in step time against the previous step; steps marked "pack" were measured together):

| change | effect |
|---|---|
| `matmul` over a batch runs one matrix per task; AVX2 microkernel with named accumulators; spinning pool; `im2col` without zero-fill | `mlp_wide` -13%, `cnn_fer` -36%, `cnn_large` -31%, `transformer_tiny` -10% |
| bias gradient reduced without copying the gradient | `mlp_wide` -1.9%, `mlp_adam` -2.7%, `transformer_tiny` -1.4% |
| A packed once per block; weight gradients cut along `k` (see below) | `transformer_tiny` -7.8% |
| first accumulation writes the gradient instead of zero-fill plus add (`Value::grad_accum`) | `transformer_tiny` -16%, `mlp_wide` -9%, `mlp_adam` -6% |
| vectorised GELU (`kernels/vmath`) | `transformer_tiny` -3.6% |
| pack: reshape and closure copies, BatchNorm, max-pooling (`uint32` indices), `transpose`, cross-entropy | `transformer_tiny` -28%, `cnn_fer` -12%, `cnn_mnist` -10%, `cnn_large` -8% |
| fused `linear` | `transformer_tiny` -7.3%, MLPs -3% |
| fused `attention` | `transformer_tiny` -18% |
| GEMM cut along `k`, faster packing | `transformer_tiny` -12%, `cnn_mnist` -8% |
| per-sample convolution with a private column buffer | `cnn_fer` -28%, `cnn_large` -27%, `cnn_mnist` -25% |

These stage-by-stage figures come from the build-per-stage runs made while developing the change; only the final table above is archived as CSV.

Findings worth keeping:

- **The GEMM was the largest single lever on the MLPs and almost nothing on the CNNs.** The AVX2 microkernel spilled 10 accumulator registers to the stack on every step of `k`; naming them removed it. The weight gradient of a `Linear` (64 x 64 x 1024) used one thread because the work is in the depth, not in the output; cutting `k` into pieces that are summed in order fixed that without making the result depend on the thread count.
- **The CNNs were bound by memory traffic, not arithmetic.** The `col` tensor (103 MB for `cnn_large`) was written, kept and read back. A per-thread buffer of 1.18 MB that stays in the L2 cache removed most of it.
- **The pool needed to spin.** An empty fork/join cost 9.8 µs with 8 threads through a mutex and condition variable. Spinning 100 µs before sleeping (`NN_SPIN_US`) is worth -17% on `transformer_tiny` and -16% on `cnn_mnist` against no spin; 500 µs is about the same as 100.
- **Allocation was not the problem.** `malloc`/`free`/`new` are 0.03% to 0.17% of samples on four profiled workloads, so there is no arena. The remaining cost was zero-filling and copying.
- **FMA contraction broke determinism.** GCC contracted `a * b + c` differently in the inlined single-thread copy of a lambda and in the one the pool calls, which made `batchnorm`'s backward differ by about 1e-6 between 1 and 2 threads on `-march=native` only. The library now builds with `-ffp-contract=off`; the GEMM uses explicit intrinsics and is unaffected.
- **Threads:** no default beats 8 everywhere. With 4 threads `cnn_large` is 19% faster (38.3 vs 47.1 ms) but `cnn_mnist` is 45% slower and the MLPs 6% to 9% slower, so the pool still uses every hardware thread. A likely cause for `cnn_large` is two 1.18 MB column buffers sharing a 1.25 MB L2; not tested.

Where the remaining time is (native, after the change): on `cnn_fer`, BatchNorm is about a third of the step (backward 3.1 ms, forward 1.8 ms of 14.7); max-pooling is 9% to 13% of the CNN steps, a fifth of its backward being the `memset` of the plane before the scatter; on `transformer_tiny`, `Linear` is 53% of the step and the scalar `expf` about 2% to 3%; `mlp_wide` is 84% GEMM, so only a better multi-threaded GEMM moves it.

Not done: batches smaller than the thread count (convolution and attention) run their samples serially with a parallel GEMM inside, which only the tests exercise; the results are not bitwise equal to those of v1.3.0, only invariant to the thread count (the depth-cut GEMM changes the summation order for `m * n <= 65536` and `k >= 512`, GELU changes in the last ulp, and the convolution bias gradient is summed per sample); and the comparison is on one machine, so a CPU with AVX-512 or fewer cores would shift the ratios.
