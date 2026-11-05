# Follow-ups: flaky test and thread default

## Flaky-test fix

`end-to-end gradcheck through a small Linear-ReLU-Linear network` (tests/test_nn.cpp) failed about 1 run in 20-60. The layers used the library's unseeded `Tensor::random`, so now and then a pre-activation landed within the finite-difference step (1e-3) of the ReLU kink and the numeric gradient straddled it.

The other tests set parameters directly (`layer.parameters()[i].data() = Tensor(...)`), and there is no seeding API, so the test now does the same: fixed weights and biases for both Linear layers, plus a `CHECK` that every first-layer pre-activation is more than 0.3 from zero (the actual minimum is 0.5). No library change.

Other gradchecks through relu/maxpool: `relu gradcheck away from the kink` and `maxpool2d gradcheck` use hand-written inputs (|x| >= 1, window maxima separated by >= 1), and no other test feeds random init through relu or maxpool into a finite-difference check. Nothing else needed fixing.

Proof: the test case run 500 times in a loop on build/native, 0 failures. Full ctest passes in build/native and build/portable (198/198).

## Thread-default measurement

Method: `apps/bench --suite workloads`, 15 runs per call (the tool's default), one call per cell, 3 calls per cell, interleaved round-robin over all cells so slow drift hits every cell alike. The tables give the median of the 3 `total_ms` values, and in parentheses the max/min-1 spread across the 3 calls. Unpinned is what a user gets by default; pinned uses `taskset -c 0-7` (the 4 P-cores with their hyperthreads, so t=12 there is 12 threads on 8 CPUs). Raw per-call CSVs: `docs/performance/baseline/followups/<config>-<label>-t<N>-r<K>.csv`.

Machine: i5-12450HX (4 P-cores with HT = cpus 0-7, 4 E-cores = 8-11), idle at start (load average 0.7), cpufreq governor `powersave` (could not be changed from here, so absolute numbers are noisier than a `performance` run; comparisons below are all within the same sweep). The old-vs-old spread bar of 3% holds for cnn_mnist, cnn_large, mlp_adam pinned and the portable rows; it does not hold for unpinned mlp_wide/mlp_adam at some thread counts (up to 30-40% on a single call, which is the scheduler placing threads on E-cores or sharing a core), and xor (10 us) is dominated by timer noise. The conclusions below do not rest on those cells.

```
native unpinned (median ms of 3; run spread max/min-1 in %)
bench                          t=1           t=2           t=3           t=4           t=6           t=8          t=12
xor                    0.010( 50%)   0.007(295%)   0.010( 81%)   0.010( 49%)   0.010(  2%)   0.010( 78%)   0.010(294%)
mlp_wide              31.197( 17%)  21.494(  4%)  17.249(  2%)  14.954(  2%)  20.472( 25%)  15.592(  7%)  14.639( 34%)
cnn_mnist              5.243(  2%)   2.869(  3%)   2.675( 11%)   2.570(  2%)   2.671(  4%)   2.653(  3%)   2.654(  0%)
cnn_large            306.032(  7%) 276.034(  2%) 257.135(  1%) 249.167(  1%) 255.109(  1%) 256.498(  2%) 257.240(  1%)
mlp_adam              32.604( 16%)  23.662(  0%)  19.365(  1%)  17.796(  2%)  20.741(  2%)  19.200( 27%)  17.900(  3%)
infer_nograd_b1        0.066(  5%)   0.067(  4%)   0.065(  0%)   0.067(  2%)   0.067(  2%)   0.067(  2%)   0.068(  1%)
infer_nograd_b256     12.842(  6%)  12.566(  1%)  11.935(  2%)  11.372(  3%)  11.873(  1%)  11.401(  4%)  11.152(  1%)

native pinned (median ms of 3; run spread max/min-1 in %)
bench                          t=1           t=2           t=3           t=4           t=6           t=8          t=12
xor                    0.006(  2%)   0.006(334%)   0.007(  7%)   0.006( 20%)   0.006(348%)   0.006( 16%)   0.006(  7%)
mlp_wide              30.550(  0%)  21.417(  1%)  17.698(  0%)  15.033(  2%)  15.011(  3%)  14.864(  2%)  14.714(  1%)
cnn_mnist              5.238(  0%)   2.868(  2%)   2.686(  1%)   2.553(  0%)   2.643(  1%)   2.614(  1%)   2.789(  2%)
cnn_large            305.354(  1%) 273.196(  2%) 257.308(  1%) 249.984(  1%) 254.383(  1%) 252.240(  1%) 255.711(  0%)
mlp_adam              32.479(  0%)  23.485(  0%)  19.508(  1%)  18.141(  2%)  18.324(  1%)  17.711(  1%)  17.850(  1%)
infer_nograd_b1        0.066(  1%)   0.069(  1%)   0.065(  2%)   0.067(  2%)   0.068(  2%)   0.068(  5%)   0.069(  7%)
infer_nograd_b256     12.967(  1%)  12.526(  2%)  12.099(  3%)  11.520(  0%)  11.601(  3%)  11.095(  1%)  11.409(  4%)

portable unpinned (median ms of 3; run spread max/min-1 in %)
bench                          t=1           t=2           t=3           t=4           t=6           t=8          t=12
xor                    0.010(  4%)   0.006( 70%)   0.010(  3%)   0.010(  3%)   0.006( 70%)   0.010( 70%)   0.010(  1%)
mlp_wide              72.619(  1%)  42.221(  1%)  33.644(  0%)  29.472(  1%)  34.047( 30%)  27.584( 10%)  21.297(  1%)
cnn_mnist              6.751(  1%)   4.010(  5%)   3.927(  1%)   3.788(  0%)   3.881(  1%)   3.797(  2%)   3.821( 10%)
cnn_large            457.197(  0%) 350.415(  1%) 316.672(  1%) 296.482(  0%) 299.066(  0%) 295.133(  1%) 295.815(  1%)
mlp_adam              73.230(  1%)  43.633(  1%)  34.896(  0%)  31.169(  1%)  29.323(  2%)  25.270( 40%)  23.340(  3%)
infer_nograd_b1        0.093(  0%)   0.094(  1%)   0.094(  1%)   0.094(  0%)   0.094(  0%)   0.094(  0%)   0.094(  1%)
infer_nograd_b256     16.422(  1%)  15.574(  2%)  14.774(  0%)  14.513(  1%)  15.206(  2%)  14.722(  2%)  14.694(  2%)

portable pinned (median ms of 3; run spread max/min-1 in %)
bench                          t=1           t=2           t=3           t=4           t=6           t=8          t=12
xor                    0.006(  5%)   0.006(  0%)   0.006(  7%)   0.006(  6%)   0.006(  2%)   0.006(331%)   0.006(  5%)
mlp_wide              72.309(  1%)  42.163(  1%)  33.547(  1%)  29.373(  0%)  29.443(  3%)  27.961(  1%)  26.980(  1%)
cnn_mnist              6.761(  1%)   4.024(  1%)   3.923(  2%)   3.898(  4%)   3.846(  1%)   3.848(  1%)   3.975(  1%)
cnn_large            458.420(  0%) 347.516(  0%) 316.173(  0%) 296.357(  1%) 309.579(  1%) 306.043(  0%) 303.182(  8%)
mlp_adam              73.304(  1%)  43.344(  0%)  34.908(  1%)  31.324(  1%)  31.370(  1%)  29.842(  1%)  29.128(  6%)
infer_nograd_b1        0.093(  1%)   0.094(  1%)   0.094(  1%)   0.094(  1%)   0.094(  0%)   0.094(  0%)   0.094(  1%)
infer_nograd_b256     16.504(  1%)  15.521(  0%)  14.969(  1%)  14.591(  2%)  14.607(  1%)  14.243(  1%)  14.721(  1%)
```

## Decision

Keep the default (`NN_NUM_THREADS`, else `hardware_concurrency()`). No code change.

Native, unpinned, default (12) against the best count:

| benchmark | t=12 | best | gap |
|---|---|---|---|
| cnn_mnist | 2.654 ms | 2.570 ms (t=4) | 3.3%, inside the 2-11% spread |
| mlp_wide | 14.64 ms | 14.64 ms (t=12) | none (t=4 is 2% slower) |
| cnn_large | 257.2 ms | 249.2 ms (t=4) | 3.2% |
| mlp_adam | 17.90 ms | 17.80 ms (t=4) | 0.6% |
| infer_nograd_b256 | 11.15 ms | 11.15 ms (t=12) | none |

The 7.8 ms vs 5.8 ms cnn_mnist penalty and the slower 2048 matmul seen in phase 2 do not reproduce after the phase 3 changes (fewer copies, parallel elementwise/im2col): cnn_mnist is flat from 3 to 12 threads. The criterion for a change was that a cap improves cnn_mnist and mlp_wide by more than noise without slowing a guarded row; on native it improves neither, so the first condition fails.

Portable, where a cap would hurt: unpinned mlp_wide is 21.30 ms at 12 threads against 29.47 ms at 4 (cap would be 38% slower), mlp_adam 23.34 against 31.17 ms (34% slower). A cap at physical cores (4 here, read from thread_siblings_list) or an AVX2-only cap would therefore either do nothing measurable on native or cost the portable build; an AVX2-conditional cap would buy at most ~3% on one native row, which is not worth a topology parser plus a sysfs dependency. Pinned numbers agree (native pinned t=12 vs t=4: cnn_mnist 9% slower, but that is 12 threads on 8 CPUs, a case the default does not produce unless the user pins).

`NN_NUM_THREADS` and `set_num_threads` still override the default and the documentation in `include/nn/kernels/parallel.hpp` and README.md (default is `hardware_concurrency()`) stays accurate. Users who see SMT contention on a native build can set `NN_NUM_THREADS` to the physical core count.

## Packing, zero-fill and backward allocations

Method: `NN_NUM_THREADS=8 taskset -c 0-7 apps/bench --suite workloads`, 3 calls per cell interleaved old/new, median of the 3 `total_ms` (max/min-1 spread in parentheses). Old is head df02721 built in a separate worktree (native and portable); new is the packing commit. Machine idle apart from a browser (CPU 5% busy during the runs). Raw CSVs: `docs/performance/baseline/followups/packb-native/`, `packb-portable/`, and `packb-infer-*` (5 interleaved calls of `--workload infer_nograd_b256 --runs 40`, used to settle a 3% reading).

Fresh `perf record -g` on head (`apps/bench/profile.sh`, native, 8 threads), share of samples:

| candidate | cnn_mnist | mlp_wide | cnn_large | verdict |
|---|---|---|---|---|
| B zero-fill (`__memset_avx2_unaligned_erms`) | 8.4% | 1.9% | 6.3% | stopped (public API) |
| A serial B pack (`blocked` self time) | 7.1% | 1.9% | 3.0% | kept |
| C softmax/CE/sigmoid/tanh backward | 0 | 0 | 0 | skipped |

```
native (ms)        old              new
mlp_wide      14.951( 1.2%)   14.180( 1.0%)   -5.2%
cnn_mnist      2.655( 3.2%)    2.640( 0.9%)   noise
cnn_large    252.423( 1.0%)  250.217( 0.5%)   noise
mlp_adam      17.638(13.6%)   16.391( 1.1%)   -7.1%
infer_b1       0.068           0.067           noise
infer_b256    11.358( 3.0%)   11.689( 2.8%)   see below

portable (ms)      old              new
mlp_wide      28.002( 1.4%)   26.557( 0.7%)   -5.2%
cnn_mnist      3.920( 0.6%)    3.880( 1.4%)   noise
cnn_large    303.907( 0.5%)  304.099( 0.4%)   noise
mlp_adam      29.891( 0.7%)   29.013( 0.8%)   -2.9%
infer_b1       0.094           0.093           noise
infer_b256    14.223( 1.7%)   14.588( 1.7%)   see below
```

infer_nograd_b256 read +2.9% (native) and +2.6% (portable) in that sweep; the 5-call recheck gives native 11.373 -> 11.408 ms (+0.3%) and portable 14.287 -> 14.308 ms (+0.1%), so no slowdown.

**A. Parallel B packing: kept.** `pack_b` is called per range of kNR column panels through `parallel_range` (at least 8 panels per chunk; below that, or when the product is serial, it stays on the caller). Writes are disjoint runs of `b_pack` and pure copies, so results are bitwise identical for any thread count (existing determinism test plus a new test with n = kNC+37, k = kKC+5, both `trans_b`). Threshold sweep (native, 3-5 runs): 4 and 8 panels tie, 1 panel was slower on cnn_large (260.5 vs 254.2 ms); 8 chosen. Only the two large-matmul rows move; cnn_mnist's packs are too small to split.

**B. Zero-fill of fresh tensors: stopped, no code.** `Tensor` stores a `std::vector<float>` and `data()` returns `std::vector<float>&` in the public header; a vector of size n is always value-initialised, and the only way to skip that is a different storage type (e.g. a default-init allocator), which changes the public `data()` signature. Per the brief this is reported, not implemented. Share to recover if the API is ever changed: about 8% of cnn_mnist and 6% of cnn_large samples.

**C. Backward closures of softmax/CE/sigmoid/tanh: skipped.** None of the bench workloads call these ops (the workloads use relu, maxpool, linear, conv, sum), so their share in all three profiles is 0, under the 1% bar.
