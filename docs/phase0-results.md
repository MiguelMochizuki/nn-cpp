# Phase 0 results: baseline, profile and phase ranking

Machine: shared laptop, 4 P-cores with hyperthreading (cpus 0-7, 8 hardware threads) plus 4 E-cores (cpus 8-11), governor `powersave`. Data is in `docs/baseline/`. Everything below is one pinned (`taskset -c 0-7`) `NN_NATIVE=ON` run unless stated.

## Baseline (native, pinned, t8)

Source: `docs/baseline/native-pinned-t8.csv`, 15 runs per row, times in ms as recorded by the harness. Spread is the harness's relative spread statistic.

| workload | forward | backward | optimizer | total | fwd / bwd / opt share | spread |
|---|---|---|---|---|---|---|
| xor | 0.0066 | 0.0094 | 0.00026 | 0.0162 | 41% / 58% / 2% | 0.127 |
| mlp_wide | 79.8 | 122.6 | 1.73 | 209.4 | 38% / 59% / 1% | 0.118 |
| cnn_mnist | 4.66 | 17.8 | 0.067 | 22.6 | 21% / 79% / 0.3% | 0.013 |
| cnn_large | 343.5 | 740.4 | 0.126 | 1085.8 | 32% / 68% / 0.01% | 0.035 |
| mlp_adam | 68.8 | 111.2 | 5.78 | 186.8 | 37% / 60% / 3% | 0.030 |
| infer_nograd_b1 | 1.38 | 0 | 0 | 1.38 | forward only | 0.014 |
| infer_nograd_b256 | 32.1 | 0 | 0 | 32.1 | forward only | 0.025 |

Measurement caveats: `backward_ms` includes `zero_grad`; `forward_ms` includes loss construction; the `infer_nograd` rows have 0 in backward and optimizer by design. Backward is 58% to 79% of a training step, so it is the larger target in every training workload.

Caveat: this t8 file was recorded on a busy machine. The mlp_wide row (209 ms) is inflated against about 170 ms in the quiet Task 7 check runs, and mlp_adam (186.8 ms) is faster than mlp_wide, which does not hold on a quiet machine. A re-record on a quiet machine is still pending; treat this table as approximate.

## Measured noise (Task 7)

Two back-to-back native pinned t8 runs, after closing Firefox (load average about 1.1):

- mlp_wide: 171.7 vs 166.9 ms (1.03x, 3%), spread 0.027 / 0.023.
- cnn_mnist: 17.52 vs 17.35 ms (1.01x, 1%), spread 0.029 / 0.010.
- Other rows ranged within a 0.78x to 0.95x ratio band between the two runs (matmul/16 0.78x); xor 2.60x (a 16 microsecond workload, not usable).
- With Firefox running, the same check gave 13% on mlp_wide, so the noise bar for a phase is "more than about 3% on a quiet machine, more than about 13% on a busy one". The committed t8 CSV was a separate run, so its absolute numbers differ from the Task 7 check runs by up to the amounts above (mlp_wide total 209 ms here vs about 170 ms there). Compare phases only against a fresh baseline taken the same session.

## matmul GFLOP/s (native, pinned, t8)

| n | 8 | 16 | 32 | 64 | 128 | 256 | 512 | 1024 | 2048 |
|---|---|---|---|---|---|---|---|---|---|
| GFLOP/s | 5.3 | 13.4 | 19.7 | 27.8 | 23.5 | 23.8 | 21.9 | 25.2 | 13.2 |

Context (estimate, not measured): one P-core at about 4.4 GHz with 2 FMA ports x 8 floats x 2 flops = 32 flop/cycle gives about 140 GFLOP/s AVX2 single-core peak. The current matmul sits at roughly 15% to 20% of that for n from 64 to 1024 and drops to about 9% at n=2048 (cache misses). The usable headroom for a single-threaded kernel is therefore several times, before any threading.

Thread points: the library has no thread pool yet, so the t1, t2, t4, t8 and unpinned t12 files all measure single-threaded speed; the thread count only affects which CPUs the process is pinned to and the recorded label. They become meaningful after a threading phase. The committed t1, t2 and t4 files were taken with `taskset -c 0-(n-1)`, which makes no difference for a single-threaded library; `run_baseline.sh` now pins all thread points to cpus 0-7 (cpus 0/1, 2/3, 4/5, 6/7 are hyperthread siblings).

Real-MNIST reference (`train_mnist`, from `docs/baseline/mnist-native.txt` and `mnist-portable.txt`): native 98.11% test accuracy, 1:42.46 wall time; portable 98.47%, 1:47.72. `train_mnist` initialises weights and shuffles without a seed, so later phases compare accuracy by tolerance, not equality.

## Profile (perf record -g, RelWithDebInfo, NN_NATIVE=ON, pinned to cpus 0-7)

Full top-25 lists: `docs/baseline/perf-cnn_mnist.txt` and `docs/baseline/perf-mlp_wide.txt`. Percentages are self time (`--no-children`).

cnn_mnist:

| share | symbol |
|---|---|
| 40.88% | `nn::matmul` |
| 20.49% | `nn::Tensor::transpose` |
| 5.33% | `__memset_avx2_unaligned_erms` |
| 4.12% | `malloc` |
| 4.00% | `cfree` |
| 3.53% | `nn::relu` backward/forward lambda |
| 2.67% | `__memmove_avx_unaligned_erms` |
| 2.61% | `operator new` |
| 1.88% | `nn::conv2d` |
| 1.69% | `nn::conv2d` lambda (backward) |

mlp_wide:

| share | symbol |
|---|---|
| 80.36% | `nn::matmul` |
| 5.91% | `nn::Tensor::transpose` |
| 2.35% | `cfree` |
| 1.54% | `__memset_avx2_unaligned_erms` |
| 1.12% | `malloc` |
| 1.03% | `__memmove_avx_unaligned_erms` |
| 0.71% | `operator new` |
| 0.61% | `nn::relu` lambda |
| 0.49% | `nn::Tensor::operator+=` |

Notes: `transpose` is called from `Linear::forward`, the transpose op and the matmul backward path in `src/autograd/ops.cpp`; this was found by reading the call sites, the profile has no call-graph rows. `im2col`/`col2im` appear to be inlined into the `conv2d` symbols. The memset share is probably zero-initialised tensor storage, which is a guess from the symbol name.

## Phase ranking (by measured share of time)

Shares are self time from the profiles above. "Alloc" below is the sum of the malloc/free/operator new/delete rows (including their `@plt` stubs) in each file, excluding memset and memmove; the sums are derived, the individual rows are in the perf files.

| rank | candidate phase | cnn_mnist | mlp_wide | note |
|---|---|---|---|---|
| 1 | `matmul` rewrite (fast path, blocking, packing, AVX2 microkernel) | 40.88% (+20.49% transpose) | 80.36% (+5.91% transpose) | Largest in both. A contiguous fast path may remove most of the transpose time too, which is a hypothesis to test, not a result. |
| 2 | Thread pool and parallel-for, `gemm` first | no separate share | no separate share | It multiplies whatever the matmul phase leaves, so it is worth doing after rank 1; its ceiling is the matmul share above. |
| 3 | Allocation (aligned storage, arena) | about 14.66% alloc, plus 6.18% memset (5.33% + 0.85% `memset@plt`) | about 6.02% alloc, plus 1.66% memset (1.54% + 0.12% `memset@plt`) | Supported on cnn_mnist (second largest group after matmul and transpose), weak on mlp_wide. |
| 4 | `conv2d` (im2col, col2im, bias fusion) | about 3.57% (1.88% + 1.69%) | not present | Small on its own. Most of the conv cost shows up as matmul and transpose, which rank 1 addresses. |
| 5 | Elementwise, activation, pooling, optimizer loops | about 4.94% (relu 3.53% + `operator+=` 0.85% + maxpool 0.56%) | about 1.10% (relu 0.61% + `operator+=` 0.49%) | Optimizer and the SGD/Adam update loops are not in the top 25 of either profile; optim_ms is 0.3% to 3% of a step in the baseline. |

Not supported by the profile: a dedicated optimizer-loop phase (not in either top 25, at most 3% of a step by the baseline split); an elementwise/activation phase as a standalone item (about 5% at best on cnn_mnist, about 1% on mlp_wide); a conv2d-specific phase before matmul is fixed (about 3.6% direct share). The allocation phase is supported for the CNN only. Threading cannot be ranked by a profile of a single-threaded library, so it is placed after rank 1 by dependence, not by a measured share.

The first entry is the largest measured share in both files (40.88% and 80.36% against at most 20.49% and 5.91% for the next symbol).
