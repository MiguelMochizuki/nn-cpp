/**
 * bench.cpp
 * Author: Miguel Mochizuki Silva
 * Description: Command-line benchmark tool; writes workload and matmul timings to a CSV file
 */

#include "csv.hpp"
#include "runner.hpp"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

/**
 * Parses a strict non-negative decimal integer (digits only, no sign, space or junk)
 *
 * Parameters:
 * string s: text to parse
 * unsigned long& out: parsed value
 *
 * Returns bool: true on success
 */
static bool parse_uint(const std::string& s, unsigned long& out) {
	if (s.empty() || s.size() > 9 || s.find_first_not_of("0123456789") != std::string::npos) return false;
	out = std::stoul(s);
	return true;
}

/**
 * Formats a double with 6 significant digits
 *
 * Parameters:
 * double v: value
 *
 * Returns string: formatted value
 */
static std::string fmt(double v) {
	std::ostringstream o;
	o.precision(6);
	o << v;
	return o.str();
}

/**
 * Prints an error to stderr and returns the usage-error exit code
 *
 * Parameters:
 * string msg: message
 *
 * Returns int: 2
 */
static int fail(const std::string& msg) {
	std::cerr << "bench: " << msg << "\n";
	return 2;
}

/**
 * Entry point: parses flags, validates the environment, opens the CSV, then times the suites
 *
 * Parameters:
 * int argc, char** argv: command-line arguments
 *
 * Returns int: 0 on success, 2 on invalid usage or environment
 */
int main(int argc, char** argv) {
	std::string out, suite = "all", label = "pinned", only;
	unsigned long runs = 15, warmup = 3, seed = 1;

	for (int i = 1; i < argc; ++i) {
		const std::string a = argv[i];
		if (a == "--mnist") return fail("--mnist: not implemented in Phase 0");
		const bool known = a == "--out" || a == "--suite" || a == "--runs" || a == "--warmup" || a == "--seed" ||
		                   a == "--label" || a == "--workload";
		if (!known) return fail("unknown flag: " + a);
		if (i + 1 >= argc) return fail("missing value for " + a);
		const std::string v = argv[++i];
		if (a == "--out") out = v;
		else if (a == "--suite") suite = v;
		else if (a == "--label") label = v;
		else if (a == "--workload") only = v;
		else {
			unsigned long& dst = a == "--runs" ? runs : a == "--warmup" ? warmup : seed;
			if (!parse_uint(v, dst)) return fail("invalid value for " + a + ": '" + v + "'");
		}
	}
	if (out.empty()) return fail("--out <file.csv> is required");
	if (suite != "workloads" && suite != "matmul" && suite != "all") return fail("--suite must be workloads|matmul|all");
	if (label != "pinned" && label != "unpinned") return fail("--label must be pinned|unpinned");
	if (!only.empty()) {
		auto names = bench::workload_names();
		names.push_back("infer_nograd_b1");
		names.push_back("infer_nograd_b256");
		if (std::find(names.begin(), names.end(), only) == names.end()) return fail("unknown workload: " + only);
	}
	if (runs < 1) return fail("--runs must be at least 1");

	unsigned long threads = std::thread::hardware_concurrency();
	if (const char* env = std::getenv("NN_NUM_THREADS")) {
		if (!parse_uint(env, threads) || threads == 0)
			return fail(std::string("invalid NN_NUM_THREADS: '") + env + "' (expected a positive integer)");
	}

	bench::apply_threads(threads);

	try {
		// Open the output before any timing so an unwritable path fails fast.
		bench::CsvWriter csv(out);
		csv.header({"config", "label", "threads", "benchmark", "size", "forward_ms", "backward_ms", "optim_ms",
		            "total_ms", "gflops", "spread", "runs"});
		const std::string config = NN_NATIVE_BUILD ? "native" : "portable";
		const std::string th = std::to_string(threads);
		const int w = static_cast<int>(warmup), r = static_cast<int>(runs);
		const unsigned s = static_cast<unsigned>(seed);

		auto emit = [&](const bench::Result& x) {
			csv.row({config, label, th, x.name, "", fmt(x.forward_ms), fmt(x.backward_ms), fmt(x.optim_ms),
			         fmt(x.total_ms), "", fmt(x.spread), std::to_string(x.runs)});
		};

		if (suite != "matmul") {
			for (const auto& n : bench::workload_names())
				if (only.empty() || only == n) emit(bench::run_workload(n, s, w, r));
			for (size_t b : {1, 256})
				if (only.empty() || only == "infer_nograd_b" + std::to_string(b)) emit(bench::run_inference(s, b, w, r));
		}
		if (suite != "workloads") {
			for (size_t n : {8, 16, 32, 64, 128, 256, 512, 1024, 2048}) {
				const bench::MatmulResult m = bench::run_matmul(n, s, w, r);
				csv.row({config, label, th, "matmul", std::to_string(m.n), "", "", "", fmt(m.ms), fmt(m.gflops),
				         fmt(m.spread), std::to_string(m.runs)});
			}
		}
	} catch (const std::exception& e) {
		return fail(e.what());
	}
	return 0;
}
