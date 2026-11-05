/**
 * test_parallel.cpp
 * Author: Miguel Mochizuki Silva
 * Description: Unit tests for the thread pool and parallel_for
 */

#include "doctest/doctest.h"
#include "nn/kernels/parallel.hpp"

#include <atomic>
#include <vector>

using namespace nn::kernels;

namespace {

/**
 * Restores the thread count when it goes out of scope
 */
struct ThreadGuard {
	size_t saved = num_threads();
	~ThreadGuard() { set_num_threads(saved); }
};

/**
 * Runs parallel_for over n chunks and checks each index was hit exactly once
 *
 * Parameters:
 * size_t n: Number of chunks
 *
 * Returns bool: True when every chunk ran exactly once
 */
bool covers_once(size_t n) {
	std::vector<std::atomic<int>> hits(n);
	for (auto& h : hits) h = 0;
	parallel_for(n, [&](size_t i) { ++hits[i]; });
	for (auto& h : hits)
		if (h != 1) return false;
	return true;
}

}  // namespace

TEST_CASE("parallel_for hits every chunk exactly once") {
	ThreadGuard g;
	for (size_t t : {1, 2, 4, 8}) {
		set_num_threads(t);
		CHECK(num_threads() == t);
		CHECK(covers_once(1000));
	}
}

TEST_CASE("parallel_for with zero and one chunk") {
	ThreadGuard g;
	set_num_threads(4);
	CHECK(covers_once(0));
	CHECK(covers_once(1));
}

TEST_CASE("nested parallel_for completes") {
	ThreadGuard g;
	set_num_threads(4);
	std::vector<std::atomic<int>> hits(20 * 50);
	for (auto& h : hits) h = 0;
	parallel_for(20, [&](size_t i) {
		parallel_for(50, [&](size_t j) { ++hits[i * 50 + j]; });
	});
	for (auto& h : hits) CHECK(h == 1);
}

TEST_CASE("set_num_threads(0) means one thread") {
	ThreadGuard g;
	set_num_threads(0);
	CHECK(num_threads() == 1);
	CHECK(covers_once(100));
}

TEST_CASE("changing the thread count between calls") {
	ThreadGuard g;
	for (size_t t : {8, 2, 8}) {
		set_num_threads(t);
		CHECK(covers_once(500));
	}
}

TEST_CASE("per-chunk sums match the serial sum") {
	ThreadGuard g;
	set_num_threads(8);
	const size_t n = 10000;
	std::vector<long> slots(n, 0);
	parallel_for(n, [&](size_t i) { slots[i] = static_cast<long>(i) * 3 + 1; });
	long got = 0, want = 0;
	for (size_t i = 0; i < n; ++i) {
		got += slots[i];
		want += static_cast<long>(i) * 3 + 1;
	}
	CHECK(got == want);
}
