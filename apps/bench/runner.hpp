/**
 * runner.hpp
 * Author: Miguel Mochizuki Silva
 * Description: Timed training steps, inference and matmul sweeps for the benchmark harness
 */

#pragma once

#include "workloads.hpp"

#include <cstddef>
#include <string>

namespace bench {

/**
 * Wall-clock time of each phase of one training step
 *
 * Members:
 * double forward_ms, backward_ms, optim_ms: phase durations in milliseconds
 * double loss: scalar loss of the step
 */
struct StepTimes {
	double forward_ms;
	double backward_ms;
	double optim_ms;
	double loss;
};

/**
 * Aggregated timing of one workload; each time is the median over runs
 *
 * Members:
 * string name: workload name
 * double forward_ms, backward_ms, optim_ms, total_ms: median phase and step times
 * double spread: relative spread (IQR / median) of total_ms
 * int runs: number of timed runs
 */
struct Result {
	std::string name;
	double forward_ms, backward_ms, optim_ms, total_ms;
	double spread;
	int runs;
};

/**
 * Aggregated timing of one square matmul size
 *
 * Members:
 * size_t n: matrix side
 * double ms: median time in milliseconds
 * double gflops: 2*n^3 / (ms*1e6)
 * double spread: relative spread of the times
 * int runs: number of timed runs
 */
struct MatmulResult {
	size_t n;
	double ms;
	double gflops;
	double spread;
	int runs;
};

/**
 * Runs one forward, zero_grad, backward and optimizer step, timing each phase
 *
 * Parameters:
 * Workload& w: workload to step (parameters are updated)
 *
 * Returns StepTimes: phase times and the loss
 */
StepTimes time_train_step(Workload& w);

/**
 * Times a workload: warmup untimed steps, then runs timed steps
 *
 * Parameters:
 * string name: one of workload_names()
 * unsigned seed: workload seed
 * int warmup, runs: untimed and timed step counts
 *
 * Returns Result: medians over runs
 */
Result run_workload(const std::string& name, unsigned seed, int warmup, int runs);

/**
 * Times cnn_mnist forward under NoGradGuard on an input of the given batch size
 *
 * Parameters:
 * unsigned seed: seed for parameters and input
 * size_t batch: batch size
 * int warmup, runs: untimed and timed forward counts
 *
 * Returns Result: name infer_nograd_b<batch>; only forward_ms and total_ms are nonzero
 */
Result run_inference(unsigned seed, size_t batch, int warmup, int runs);

/**
 * Times a square n x n matmul on seeded operands
 *
 * Parameters:
 * size_t n: matrix side
 * unsigned seed: operand seed
 * int warmup, runs: untimed and timed matmul counts
 *
 * Returns MatmulResult: median time and GFLOP/s
 */
MatmulResult run_matmul(size_t n, unsigned seed, int warmup, int runs);

/**
 * Sets the library thread count used by every timed workload and matmul
 *
 * Parameters:
 * size_t threads: number of threads (at least 1)
 *
 * Returns void
 */
void apply_threads(size_t threads);

}  // namespace bench
