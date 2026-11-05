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

Reproduce with `apps/bench/run_baseline.sh`; compare two result files with `python3 apps/bench/compare_csv.py before.csv after.csv`.
