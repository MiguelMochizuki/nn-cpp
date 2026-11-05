/**
 * test_bench.cpp
 * Author: Miguel Mochizuki Silva
 * Description: Unit tests for the benchmark statistics and CSV helpers
 */

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"
#include "csv.hpp"
#include "runner.hpp"
#include "stats.hpp"
#include "workloads.hpp"
#include "nn/autograd/ops.hpp"
#include "nn/kernels/parallel.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>

using namespace bench;

TEST_CASE("median") {
	CHECK(median({3, 1, 2}) == doctest::Approx(2.0));
	CHECK(median({1, 2, 3, 4}) == doctest::Approx(2.5));
	CHECK_THROWS_AS(median({}), std::invalid_argument);
}

TEST_CASE("spread") {
	CHECK(spread({5}) == 0.0);
	CHECK(spread({1, 2, 3, 4, 5}) == doctest::Approx(2.0 / 3.0));
}

TEST_CASE("CsvWriter") {
	CHECK_THROWS_AS(CsvWriter("/nonexistent/x.csv"), std::runtime_error);

	const std::string path = "bench_tests_out.csv";
	{
		CsvWriter w(path);
		w.header({"a", "b"});
		w.row({"1", "2"});
		CHECK_THROWS_AS(w.row({"1"}), std::invalid_argument);
		w.row({"3", "4"});
	}
	std::ifstream in(path);
	std::stringstream ss;
	ss << in.rdbuf();
	CHECK(ss.str() == "a,b\n1,2\n3,4\n");
	std::remove(path.c_str());
}

/**
 * Runs steps of training on a workload and returns the first and last loss
 *
 * Parameters:
 * Workload& w: workload to train
 * int steps: number of optimizer steps
 *
 * Returns std::pair<float, float>: loss before the first step and after the last
 */
static std::pair<float, float> train_losses(Workload& w, int steps) {
	float first = 0.0f, last = 0.0f;
	for (int i = 0; i <= steps; i++) {
		nn::Value out = w.model->forward(nn::Value::leaf(w.x));
		nn::Value loss = w.classification ? nn::cross_entropy_loss(out, w.y)
		                                  : nn::mse_loss(out, nn::Value::leaf(w.y));
		float l = loss.data().get({0});
		if (i == 0) first = l;
		last = l;
		if (i == steps) break;
		w.optimizer->zero_grad();
		loss.backward();
		w.optimizer->step();
	}
	return {first, last};
}

TEST_CASE("workloads are reproducible per seed") {
	Workload a = make_workload("cnn_mnist", 7);
	Workload b = make_workload("cnn_mnist", 7);
	Workload c = make_workload("cnn_mnist", 8);
	auto pa = a.model->parameters(), pb = b.model->parameters(), pc = c.model->parameters();
	REQUIRE(pa.size() == pb.size());
	bool differs = false;
	for (size_t i = 0; i < pa.size(); i++) {
		CHECK(pa[i].data().data() == pb[i].data().data());
		differs = differs || pa[i].data().data() != pc[i].data().data();
	}
	CHECK(differs);
	CHECK(a.x.data() == b.x.data());
	CHECK(a.y.data() == b.y.data());
}

TEST_CASE("unknown workload throws") {
	CHECK_THROWS_AS(make_workload("nope", 1), std::invalid_argument);
}

TEST_CASE("workload shapes") {
	CHECK(make_workload("cnn_mnist", 1).x.shape() == std::vector<size_t>{32, 1, 28, 28});
	CHECK(make_workload("cnn_large", 1).x.shape() == std::vector<size_t>{64, 3, 64, 64});
	CHECK(make_workload("mlp_wide", 1).x.shape() == std::vector<size_t>{256, 784});
	CHECK(workload_names().size() == 5);
}

TEST_CASE("workloads are learnable") {
	Workload xor_w = make_workload("xor", 1);
	auto xl = train_losses(xor_w, 30);
	CHECK(xl.second < xl.first);
	Workload wide = make_workload("mlp_wide", 1);
	auto wl = train_losses(wide, 15);
	CHECK(wl.second < wl.first);
}

TEST_CASE("time_train_step returns finite positive times") {
	Workload w = make_workload("xor", 1);
	StepTimes t = time_train_step(w);
	CHECK(t.forward_ms > 0);
	CHECK(t.backward_ms > 0);
	CHECK(t.optim_ms > 0);
	CHECK(std::isfinite(t.loss));
}

TEST_CASE("run_workload aggregates medians") {
	Result r = run_workload("xor", 1, 1, 3);
	CHECK(r.runs == 3);
	CHECK(r.total_ms > 0);
	CHECK(r.total_ms >= std::max({r.forward_ms, r.backward_ms, r.optim_ms}));
	Result one = run_workload("xor", 1, 0, 1);
	CHECK(one.spread == 0.0);
}

TEST_CASE("run_inference and run_matmul") {
	CHECK(run_inference(1, 1, 1, 2).name == "infer_nograd_b1");
	CHECK(run_inference(1, 256, 0, 1).name == "infer_nograd_b256");
	MatmulResult m = run_matmul(8, 1, 1, 3);
	CHECK(m.gflops > 0);
	CHECK(m.n == 8);
}

TEST_CASE("apply_threads sets the library thread count") {
	const size_t before = nn::kernels::num_threads();
	apply_threads(3);
	CHECK(nn::kernels::num_threads() == 3);
	apply_threads(1);
	CHECK(nn::kernels::num_threads() == 1);
	nn::kernels::set_num_threads(before);
}
