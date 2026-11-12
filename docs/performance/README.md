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

`transformer_tiny` is a decoder-only `Transformer` (vocabulary 64, width 64, 4 heads, 2 layers, 64 positions) on a batch of 16 x 64 token ids with `cross_entropy_loss` over the 1024 flattened positions and AdamW at 1e-3. Same protocol as above (`NN_NUM_THREADS=8 taskset -c 0-7`, medians of 3 runs):

| build | forward ms | backward ms | optim ms | total ms (median of 3) | run spread |
|---|---|---|---|---|---|
| native | 26.8 | 33.2 | 0.35 | 60.7 | 9% |
| portable | 32.4 | 40.6 | 0.35 | 73.4 | 3% |

Roughly 44% of a step is forward, 55% backward and 0.6% the optimizer. There is no finer split: `apps/bench/profile.sh` could not run because `perf record` is not permitted on the measuring machine (`perf_event_paranoid=4`), so which ops dominate is unknown. Fused attention, softmax or LayerNorm ops are a possible follow-up only after a profile shows where the time goes.

**The guard run against v1.2.0 is inconclusive; the 3% gate was not shown to pass.** The machine's CPU governor was `powersave`, not `performance`, and identical code did not reproduce itself: run-to-run spreads were 25% to 250% on most workloads, and old-versus-new deltas flipped sign between native and portable builds. Over three passes the worst raw deltas of the new build against v1.2.0 were native +95.5% (`cnn_mnist`, median, pass 2) and +37.9% (`infer_nograd_b256`, median, pass 3), and portable +11.7% (`cnn_large`, median, pass 2) and +80.9% (`infer_nograd_b256`, median, pass 3); all came from rows with 30% to 250% spread, and nothing here is evidence of a regression or of its absence. The only low-spread row, `cnn_large`, read +3.4% median native and 0.0% portable in pass 3 and -7.0% native, +11.7% portable in pass 2. This release changes no existing library code path outside the new ops and layers (it only adds code; `apps/bench` gained the new workload), but that is an argument, not a measurement. **Rerun the guard under the `performance` governor (needs root) before releasing 1.3.0.**

Reproduce with `apps/bench/run_baseline.sh`; compare two result files with `python3 apps/bench/compare_csv.py before.csv after.csv`.
