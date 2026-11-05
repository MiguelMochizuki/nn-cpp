# Phase 1 results: blocked GEMM kernel and transpose-free matmul paths

Verdict: **DROP as it stands, by the exit rule.** The native build improves enormously (cnn_mnist 2.17x, mlp_wide 3.60x), but the portable build (`NN_NATIVE=OFF`) is slower than the pre-change code on `mlp_wide` (0.70x and 0.74x in two pairs) and on `cnn_large`, `mlp_adam` and `infer_nograd_b256`. The rule requires the portable build to be no slower on `cnn_mnist` and `mlp_wide`; `mlp_wide` fails, so the phase does not ship. The README was not changed. The decision on whether to fix the portable kernel and re-measure, or to accept a portable regression, belongs to the author.

Reference: commit b7856c8 (main). New: branch `perf/phase1-matmul` at e8b20a5. Both built Release; `ctest` passes in both new build dirs (171/171 native, 171/171 portable).

## Method

Machine: i5-12450HX, `powersave` governor, no sudo. Headline protocol `NN_NUM_THREADS=8 taskset -c 0-7 apps/bench --out <csv> --label pinned`, default runs and warm-up (15 runs per row). Native: old/new alternated three times (old1, new1, old2, new2, old3, new3). Portable: one old/new pair, plus a second pair afterwards as a check on the surprise. Load average was 0.9 to 1.2 throughout (one bench thread is itself about 1.0); firefox idle at about 5% CPU, gnome-shell about 7%. CSVs are in `docs/baseline/phase1/` (`{old,new}-{native,portable}-run<k>.csv`).

Old-vs-old stability (stop condition was a 10% pairwise difference): `cnn_mnist` total_ms 17.428, 17.418, 17.425; `mlp_wide` 170.686, 170.412, 169.883. Spread under 0.5%, the condition was not triggered. New native: `cnn_mnist` 7.983, 8.050, 8.029; `mlp_wide` 47.277, 47.443, 47.338.

## Native (medians of three) vs old

Ratio = old / new, > 1 is faster. Times in ms, median of three runs per side.

| row | old fwd | old bwd | old total | new fwd | new bwd | new total | speedup |
|---|---|---|---|---|---|---|---|
| xor | 0.0053 | 0.0076 | 0.0132 | 0.0040 | 0.0041 | 0.0084 | 1.58x |
| mlp_wide | 66.31 | 103.57 | 170.41 | 20.41 | 25.48 | 47.34 | 3.60x |
| cnn_mnist | 3.47 | 13.87 | 17.43 | 4.35 | 3.62 | 8.03 | 2.17x |
| cnn_large | 327.6 | 678.7 | 1006.0 | 273.5 | 221.8 | 495.2 | 2.03x |
| mlp_adam | 56.92 | 89.60 | 151.71 | 17.77 | 21.05 | 44.03 | 3.45x |
| infer_nograd_b1 | 1.130 | 0 | 1.130 | 0.081 | 0 | 0.081 | 14.0x |
| infer_nograd_b256 | 25.61 | 0 | 25.61 | 22.14 | 0 | 22.14 | 1.16x |

Forward vs backward split speedups (old/new): mlp_wide forward 3.25x, backward 4.06x; mlp_adam forward 3.20x, backward 4.26x; cnn_large forward 1.20x, backward 3.06x; cnn_mnist forward **0.80x (slower)**, backward 3.84x. Backward gained more everywhere because it lost the transpose copies (the Phase 0 profile put 20% of cnn_mnist in `Tensor::transpose`) on top of the faster kernel. The shift is visible in the share: cnn_mnist is now 54% forward / 45% backward, versus 20% / 80% before.

Matmul sweep (square n x n, GFLOP/s; estimated single-core peak about 140, an estimate and not a measurement):

| n | old | new | speedup | new % of 140 |
|---|---|---|---|---|
| 8 | 6.7 | 6.1 | 0.91x (within noise) | 4% |
| 16 | 17.9 | 35.0 | 1.95x | 25% |
| 32 | 25.4 | 63.0 | 2.48x | 45% |
| 64 | 36.4 | 81.4 | 2.23x | 58% |
| 128 | 35.0 | 99.2 | 2.83x | 71% |
| 256 | 38.2 | 107.5 | 2.81x | 77% |
| 512 | 36.3 | 106.5 | 2.93x | 76% |
| 1024 | 29.9 | 106.1 | 3.55x | 76% |
| 2048 | 13.9 | 102.1 | 7.33x | 73% |

Per-row spreads are in the CSVs; every ratio above except matmul/8 is outside the larger of the two spreads (compare_csv.py flags only matmul/8 as within noise). The cnn_mnist "spread" value of the new runs (0.05) is high relative to the 2.17x effect but far from it.

## Portable (single pair, then a second pair) vs old

| row | old total | new total | ratio, pair 1 | ratio, pair 2 |
|---|---|---|---|---|
| xor | 0.0116 | 0.0094 | 1.24x | 1.27x |
| mlp_wide | 193.1 | 274.2 | **0.70x** | **0.74x** |
| cnn_mnist | 17.96 | 15.79 | 1.14x | 1.16x |
| cnn_large | 1104.5 | 1287.8 | **0.86x** | **0.84x** |
| mlp_adam | 189.2 | 271.8 | **0.70x** | **0.70x** |
| infer_nograd_b1 | 1.121 | 0.252 | 4.46x | 4.46x |
| infer_nograd_b256 | 27.60 | 42.74 | **0.65x** | **0.64x** |

Portable matmul GFLOP/s, old to new: n=8 6.2 to 2.5, n=16 11.4 to 7.8, n=32 19.3 to 8.6, n=64 23.8 to 9.4, n=128 23.7 to 9.6, n=256 22.9 to 9.9, n=512 21.2 to 9.9, n=1024 20.6 to 9.9, n=2048 11.1 to 9.8 (ratios 0.38x to 0.89x; the large-size ratio is only near 0.9x because the old code collapsed at 2048). Both pairs agree within 5%, so this is a real regression, not noise. The portable scalar sgemm (about 10 GFLOP/s) is roughly half as fast as the old triple loop (20 to 24 GFLOP/s) at mid sizes. The old loop's inner access pattern was presumably vectorised by the compiler (SSE2 baseline), which the packed scalar kernel does not match; this is a hypothesis from the numbers, not something profiled here. cnn_mnist and the small shapes still gain in portable because the transpose copies are gone from backward (cnn_mnist backward 14.1 to 8.8 ms, forward 3.76 to 6.93 ms).

## Exit-rule check

| condition | result |
|---|---|
| native cnn_mnist total improves by more than 5% | pass, 2.17x |
| native mlp_wide total improves by more than 5% | pass, 3.60x |
| portable not slower on cnn_mnist | pass, 1.14x and 1.16x |
| portable not slower on mlp_wide | **fail, 0.70x and 0.74x** |
| ctest passes in build/native and build/portable | pass, 171/171 each |

## Real-data check (native)

`taskset -c 0-7 build/native/apps/train_mnist data/mnist`: test accuracy 98.37% (9837/10000), at or above the 98.0% bar (Phase 0: 98.11% native, 98.47% portable; the program is unseeded). Wall time 0:33.13 versus 1:42.46 in Phase 0 (3.1x). Raw output: `docs/baseline/phase1/mnist-native.txt`. Accuracy for the portable build was not re-measured.

## Surprises

- cnn_mnist forward is 20% slower in the native build (3.47 to 4.35 ms), while the whole step is 2.17x faster. Likely the small-shape conv forward path (im2col plus a small matmul) pays packing overhead the old loop did not; not profiled. cnn_large forward gains only 1.20x for the same reason.
- infer_nograd_b1 at 14x (1.13 to 0.081 ms) is mostly the tiny-m matmul now skipping a transpose copy of the weight; take it as a real but workload-specific gain.
- matmul/8 and the xor row are microsecond-scale; xor's 1.58x is not trustworthy beyond "not slower".
- infer_nograd_b256 gains only 1.16x natively: forward-only inference of this model is limited by something other than the matmul (not investigated).

## Next

A fresh `perf` profile belongs at the start of Phase 2; no new profile ranking was taken here. For this phase the open question is the portable scalar kernel (blocking/packing overhead at about 10 GFLOP/s against the old 20 to 24), which decides whether it can pass the guard.
