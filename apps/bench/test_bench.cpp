/**
 * test_bench.cpp
 * Author: Miguel Mochizuki Silva
 * Description: Unit tests for the benchmark statistics and CSV helpers
 */

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"
#include "csv.hpp"
#include "stats.hpp"

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
