/**
 * runner.cpp
 * Author: Miguel Mochizuki Silva
 * Description: Implements the timed benchmark runner
 */

#include "runner.hpp"

#include "stats.hpp"
#include "nn/autograd/ops.hpp"
#include "nn/kernels/parallel.hpp"

#include <chrono>
#include <random>

namespace bench {

using namespace nn;
using Clock = std::chrono::steady_clock;

/**
 * Milliseconds elapsed between two time points
 *
 * Parameters:
 * time_point a, b: start and end
 *
 * Returns double: b - a in ms
 */
static double ms_between(Clock::time_point a, Clock::time_point b) {
	return std::chrono::duration<double, std::milli>(b - a).count();
}

StepTimes time_train_step(Workload& w) {
	Value x = Value::leaf(w.x, false);
	auto t0 = Clock::now();
	Value out = w.model->forward(x);
	Value loss = w.classification ? cross_entropy_loss(out, w.y) : mse_loss(out, Value::leaf(w.y, false));
	auto t1 = Clock::now();
	w.optimizer->zero_grad();
	loss.backward();
	auto t2 = Clock::now();
	w.optimizer->step();
	auto t3 = Clock::now();
	return {ms_between(t0, t1), ms_between(t1, t2), ms_between(t2, t3), loss.data().data()[0]};
}

Result run_workload(const std::string& name, unsigned seed, int warmup, int runs) {
	Workload w = make_workload(name, seed);
	for (int i = 0; i < warmup; ++i) time_train_step(w);
	std::vector<double> f, b, o, t;
	for (int i = 0; i < runs; ++i) {
		StepTimes s = time_train_step(w);
		f.push_back(s.forward_ms);
		b.push_back(s.backward_ms);
		o.push_back(s.optim_ms);
		t.push_back(s.forward_ms + s.backward_ms + s.optim_ms);
	}
	return {name, median(f), median(b), median(o), median(t), spread(t), runs};
}

Result run_inference(unsigned seed, size_t batch, int warmup, int runs) {
	Workload w = make_workload("cnn_mnist", seed);
	std::mt19937 gen(seed);
	std::uniform_real_distribution<float> dist(0.0f, 1.0f);
	Tensor x({batch, 1, 28, 28});
	for (float& v : x.data()) v = dist(gen);
	Value xv = Value::leaf(x, false);
	NoGradGuard guard;
	for (int i = 0; i < warmup; ++i) w.model->forward(xv);
	std::vector<double> t;
	for (int i = 0; i < runs; ++i) {
		auto t0 = Clock::now();
		w.model->forward(xv);
		t.push_back(ms_between(t0, Clock::now()));
	}
	double m = median(t);
	return {"infer_nograd_b" + std::to_string(batch), m, 0.0, 0.0, m, spread(t), runs};
}

MatmulResult run_matmul(size_t n, unsigned seed, int warmup, int runs) {
	std::mt19937 gen(seed);
	std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
	Tensor a({n, n}), b({n, n});
	for (float& v : a.data()) v = dist(gen);
	for (float& v : b.data()) v = dist(gen);
	for (int i = 0; i < warmup; ++i) matmul(a, b);
	std::vector<double> t;
	for (int i = 0; i < runs; ++i) {
		auto t0 = Clock::now();
		Tensor c = matmul(a, b);
		t.push_back(ms_between(t0, Clock::now()));
	}
	double m = median(t);
	double flops = 2.0 * n * n * n;
	return {n, m, flops / (m * 1e6), spread(t), runs};
}

void apply_threads(size_t threads) {
	kernels::set_num_threads(threads);
}

}  // namespace bench
