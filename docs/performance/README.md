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

## v1.3.0: `transformer_tiny` and the v1.2.0 guard

`transformer_tiny` is a decoder-only `Transformer` (vocabulary 64, width 64, 4 heads, 2 layers, 64 positions) on a batch of 16 x 64 token ids with `cross_entropy_loss` over the 1024 flattened positions and AdamW at 1e-3. Protocol: `NN_NUM_THREADS=8 taskset -c 0-7`, CPU governor `performance`, 3 passes of `bench --runs 50 --warmup 5`, medians of the 3 passes:

| build | forward ms | backward ms | optim ms | total ms | run spread |
|---|---|---|---|---|---|
| native | 24.3 | 23.8 | 0.35 | 48.8 | 3% |
| portable | 29.0 | 31.7 | 0.35 | 61.2 | 2% |

Roughly half of a step is forward, half backward and under 1% the optimizer. There is no finer split: `apps/bench/profile.sh` could not run because `perf record` is not permitted on the measuring machine (`perf_event_paranoid=4`), so which ops dominate is unknown. Fused attention, softmax or LayerNorm ops are a possible follow-up only after a profile shows where the time goes.

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
