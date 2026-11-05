# Phase 3 results: fewer copies and parallel elementwise/conv kernels

Verdict: **SHIP.** Native (`-DNN_NATIVE=ON`, `NN_NUM_THREADS=8 taskset -c 0-7`) cnn_mnist is 2.19x faster than the pre-change code (bar: more than 5%). Portable is faster on every guarded row (cnn_mnist 1.71x). `ctest` passes in both build dirs (196/196, reported).

Reference: commit 08af261 (Phase 2). New: branch `perf/phase3-memory`. Both Release, native and portable builds, old built in a separate worktree.

## What shipped

Commits `08af261..HEAD` (`git log --oneline`), oldest first. Per-change deltas are medians of three against the immediately preceding commit (native, then portable where measured); the per-step CSVs are in `docs/performance/baseline/phase3/steps/`, the figures below are as reported in the implementer notes (uncommitted).

| commit | change | measured delta |
|---|---|---|
| f663b62 | Op results are moved into graph nodes instead of copied; `NodeImpl` moves its data; the conv2d column buffer is moved into the backward closure; `Tensor::reshape` gains an rvalue overload | native cnn_mnist 5.91 -> 5.44 ms, cnn_large 465.5 -> 383.7, mlp_wide 33.81 -> 32.62, infer_b256 21.77 -> 16.52; portable cnn_large 520.5 -> 440.3 |
| a9f1cc2 | `Tensor::size` defined inline | native cnn_mnist 5.48 -> 5.29 (about 3.6%, borderline) |
| e3f0667 | add/sub/mul/matmul backward skip the gradient copy when shapes already match | native mlp_wide 32.09 -> 28.64; portable 43.21 -> 39.60 |
| dc5bed5 | A non-leaf node's grad is allocated only when backward sizes it | native mlp_wide 28.86 -> 27.19, cnn_mnist 5.37 -> 5.20 |
| 4763214 | relu backward accumulates straight into the parent grad | native cnn_mnist 5.27 -> 4.19, cnn_large 374.5 -> 321.8, mlp_wide 27.09 -> 23.78 |
| 5dbe88a | Fast paths for scalar, row and column broadcast in elementwise ops | native mlp_wide 23.75 -> 17.19, mlp_adam 25.61 -> 19.01; portable mlp_wide 36.48 -> 30.50 |
| 67be25d | relu forward/backward and in-place `+=`/`-=` run on the thread pool via `parallel_range` (8192 elements per chunk) | native mlp_wide 17.14 -> 15.00, cnn_mnist 4.00 -> 3.76; portable cnn_large 391.0 -> 369.5 |
| 095979b | im2col, col2im and maxpool2d run in parallel with hoisted window bounds | native cnn_mnist 3.76 -> 2.67, cnn_large 319.3 -> 270.3, infer_b256 16.29 -> 12.68; portable cnn_mnist 4.81 -> 3.93 |

Two further commits are not performance changes: 056f715 adds the raw CSVs, and 1cb334a (review fix) adds shape checks in relu and maxpool2d backward before they write into the parent grad (see Behaviour and API notes).

`parallel_range(n, min_per_chunk, fn)` in `nn/kernels/parallel.hpp` splits a range over the Phase 2 pool. Chunk size was chosen by a sweep (native, speedup vs the serial-equivalent previous commit, cnn_mnist / mlp_wide): 2048: 1.124 / 1.156; 8192: 1.126 / 1.143; 32768: 1.060 / 1.145; 65536: 1.100 / 1.126. 8192 is on the plateau. Nothing was reverted. The im2col/col2im and maxpool2d parts of 095979b were measured together.

## Headline (medians of three, alternated old/new, `NN_NUM_THREADS=8 taskset -c 0-7`, 15 runs per row)

Ratio = old/new total_ms. Max/min spread of the three runs is in brackets. Raw files: `docs/performance/baseline/phase3/{old,new}-{native,portable}-run{1,2,3}.csv`.

| row | native old ms | native new ms | speedup | portable old ms | portable new ms | speedup |
|---|---|---|---|---|---|---|
| xor | 0.009 | 0.006 | noise (old spread 18 to 35%) | 0.009 | 0.006 | noise |
| mlp_wide | 33.945 (1.0%) | 14.997 (0.4%) | 2.26x | 45.057 (1.1%) | 28.116 (0.4%) | 1.60x |
| mlp_adam | 30.458 (1.6%) | 17.820 (1.2%) | 1.71x | 41.291 (0.6%) | 30.210 (0.2%) | 1.37x |
| cnn_mnist | 5.963 (1.7%) | 2.721 (1.3%) | 2.19x | 6.846 (2.0%) | 4.002 (11.3%) | 1.71x |
| cnn_large | 473.758 (1.1%) | 269.725 (0.6%) | 1.76x | 525.294 (0.2%) | 321.746 (0.3%) | 1.63x |
| infer_nograd_b1 | 0.080 (1.3%) | 0.069 (1.3%) | 1.15x | 0.106 (0.9%) | 0.096 (5.7%) | 1.11x |
| infer_nograd_b256 | 22.040 (1.5%) | 13.113 (2.3%) | 1.68x | 24.741 (1.9%) | 15.442 (3.7%) | 1.60x |

Portable cnn_mnist new has an 11.3% spread across its three runs; the ratio is still well clear of the gate.

## Exit rule

- Native cnn_mnist 2.19x (bar: more than 5% faster): pass.
- Portable not slower on any guarded row: pass.
- ctest in both build dirs: pass (196/196, reported). A ThreadSanitizer run of `nn_tests` with `NN_NUM_THREADS=8` was clean (reported).

## Real MNIST (native, `NN_NUM_THREADS=8 taskset -c 0-7 train_mnist data/mnist`)

Test accuracy 98.58% (9858/10000), wall 17.8 s (Phase 2: 98.09%, 36.2 s). Raw output in `docs/performance/baseline/phase3/mnist-native.txt`. Weight init is unseeded, so the accuracy difference from Phase 2 is run-to-run variation, not an effect of this phase.

## Behaviour and API notes (need the author's sign-off)

- `Value::grad()` on a non-leaf node returns an empty placeholder (shape `{0}`) until `backward()` reaches the node, which then sizes it. Leaves are unchanged. Documented in `value.hpp`. Code that read the grad of an intermediate node before calling backward now sees an empty tensor.
- `Tensor::reshape` now has `const&` and `&&` overloads; existing callers compile unchanged.
- The `NodeImpl` constructor gained a defaulted third argument.
- relu and maxpool2d backward now write through raw pointers into the parent grad. Before, the `+=` threw `TensorShapeError` on a mismatch. The review fix (1cb334a) restores an explicit shape check before the write, so a mismatch still throws.

## Remaining profile and follow-ups

- Zero-filling fresh tensors (memset) is about 6.5% of cnn_mnist (reported). `Tensor` stores a `std::vector<float>`, which value-initialises; skipping the fill needs different storage.
- conv2d backward copies in `grad_out.reshape`; avoiding it needs a view type.
- softmax, cross-entropy, sigmoid, tanh and sum backward still allocate local tensors.
- The bias-add loop in conv2d is serial, and the conv2d result is not moved.
- Under `NoGradGuard`, results still allocate a zero grad.
- B packing in sgemm is still serial on the caller thread.
- Thread-default policy: the default stays `hardware_concurrency()` (12 here, 4 P-cores plus E-cores); Phase 2 showed small kernels get slower above the physical P-core count, and `parallel_range` makes more of the code sensitive to it. Not re-swept for Phase 3; a cap is left as a decision.
- The existing flaky Linear-ReLU-Linear gradcheck (unseeded weights near a ReLU kink) did not fail in any Phase 3 run but is not fixed.

## Caveats

- Timings were taken on a lightly loaded desktop (load average about 1.2 to 1.3, `docs/performance/baseline/phase3/load.txt`); xor is too short (microseconds) to read.
- Per-commit deltas in the table above are from the implementer notes; the final old/new numbers are recomputed from the committed CSVs.
