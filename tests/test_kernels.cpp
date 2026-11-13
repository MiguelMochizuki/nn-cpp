/**
 * test_kernels.cpp
 * Author: Miguel Mochizuki Silva
 * Description: Unit tests for the GEMM kernels against the reference triple loop
 */

#include "doctest/doctest.h"
#include "nn/kernels/gemm.hpp"
#include "nn/kernels/parallel.hpp"
#include "nn/kernels/vmath.hpp"
#include "reference_gemm.hpp"

#include <cmath>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

using namespace nn::kernels;

namespace {

constexpr float kSentinel = 1e30f;

/**
 * Which microkernel this build should pick for sgemm: AVX2/FMA when the
 * compiler targets both, otherwise SSE2 or the unpacked loop
 */
#if defined(__AVX2__) && defined(__FMA__)
constexpr bool kExpectAvx2 = true;
#else
constexpr bool kExpectAvx2 = false;
#endif

/**
 * Signature shared by sgemm, sgemm_portable and reference_sgemm
 */
using Gemm = void (*)(bool, bool, size_t, size_t, size_t, const float*, size_t,
	const float*, size_t, float*, size_t, bool);

/**
 * Holds one GEMM problem: operand storage laid out per the transpose flags
 * with leading dimensions possibly wider than the logical width
 *
 * Attributes:
 * bool ta, tb: Transpose flags
 * size_t m, n, k: Problem dimensions
 * size_t lda, ldb, ldc: Leading dimensions
 * std::vector<float> a, b, c: Storage, padding filled with kSentinel
 */
struct Problem {
	bool ta, tb;
	size_t m, n, k, lda, ldb, ldc;
	std::vector<float> a, b, c;
};

/**
 * Builds a problem with seeded uniform [-1, 1] values and sentinel padding
 *
 * Parameters:
 * bool ta, tb: Transpose flags
 * size_t m, n, k: Problem dimensions
 * size_t pad: Extra columns added to every leading dimension
 * unsigned seed: RNG seed
 *
 * Returns Problem: The filled problem
 */
Problem make_problem(bool ta, bool tb, size_t m, size_t n, size_t k, size_t pad, unsigned seed) {
	Problem p{ta, tb, m, n, k, 0, 0, 0, {}, {}, {}};
	size_t a_rows = ta ? k : m, a_cols = ta ? m : k;
	size_t b_rows = tb ? n : k, b_cols = tb ? k : n;
	p.lda = a_cols + pad;
	p.ldb = b_cols + pad;
	p.ldc = n + pad;
	std::mt19937 gen(seed);
	std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
	auto fill = [&](std::vector<float>& v, size_t rows, size_t cols, size_t ld) {
		v.assign(rows * ld, kSentinel);
		for (size_t r = 0; r < rows; ++r)
			for (size_t col = 0; col < cols; ++col) v[r * ld + col] = dist(gen);
	};
	fill(p.a, a_rows, a_cols, p.lda);
	fill(p.b, b_rows, b_cols, p.ldb);
	fill(p.c, m, n, p.ldc);
	return p;
}

/**
 * Magnitude of the terms summed into C(i, j): sum over p of |op(A)(i,p) *
 * op(B)(p,j)|, plus |C(i, j)| when accumulating. Rounding error of any
 * summation order is bounded by k * eps times this, while |C(i, j)| itself
 * can cancel to near zero, so the relative tolerance is taken against it.
 *
 * Parameters:
 * const Problem& p: The problem
 * bool accumulate: Whether C's prior value is part of the sum
 * size_t i, size_t j: Element of C
 *
 * Returns float: The magnitude
 */
float magnitude(const Problem& p, bool accumulate, size_t i, size_t j) {
	float sum = accumulate ? std::fabs(p.c[i * p.ldc + j]) : 0.0f;
	for (size_t q = 0; q < p.k; ++q) {
		float av = p.ta ? p.a[q * p.lda + i] : p.a[i * p.lda + q];
		float bv = p.tb ? p.b[j * p.ldb + q] : p.b[q * p.ldb + j];
		sum += std::fabs(av * bv);
	}
	return sum;
}

/**
 * Runs got_fn and want_fn on copies of C and checks they agree
 * elementwise (1e-5 relative to magnitude(), 1e-6 absolute floor, NaN/Inf
 * positions equal),
 * that C padding is untouched and that no sentinel leaked into the result
 *
 * Parameters:
 * const Problem& p: The problem
 * bool accumulate: Accumulate flag passed to both
 * Gemm got_fn: Kernel under test
 * Gemm want_fn: Kernel taken as ground truth
 *
 * Returns void
 */
void check(const Problem& p, bool accumulate, Gemm got_fn = sgemm_portable,
		Gemm want_fn = reference_sgemm) {
	std::vector<float> got = p.c, want = p.c;
	got_fn(p.ta, p.tb, p.m, p.n, p.k, p.a.data(), p.lda, p.b.data(), p.ldb,
		got.data(), p.ldc, accumulate);
	want_fn(p.ta, p.tb, p.m, p.n, p.k, p.a.data(), p.lda, p.b.data(), p.ldb,
		want.data(), p.ldc, accumulate);
	size_t bad = 0;
	for (size_t i = 0; i < p.m; ++i) {
		for (size_t j = 0; j < p.ldc; ++j) {
			float g = got[i * p.ldc + j], w = want[i * p.ldc + j];
			if (j >= p.n) {
				if (g != kSentinel) ++bad;
				continue;
			}
			if (std::isnan(w) || std::isnan(g)) {
				if (std::isnan(w) != std::isnan(g)) ++bad;
			} else if (std::isinf(w) || std::isinf(g)) {
				if (g != w) ++bad;
			} else if (std::fabs(g) >= 1e29f
					|| std::fabs(g - w) > std::fmax(1e-5f * magnitude(p, accumulate, i, j), 1e-6f)) {
				++bad;
			}
		}
	}
	INFO("ta=", p.ta, " tb=", p.tb, " acc=", accumulate, " m=", p.m, " n=", p.n, " k=", p.k,
		" ld pad=", p.ldc - p.n);
	CHECK(bad == 0);
}

}  // namespace

TEST_CASE("sgemm_portable matches reference over sizes and flags") {
	const size_t sizes[] = {1, 7, 33, 257};
	unsigned seed = 1;
	for (int ta = 0; ta < 2; ++ta)
		for (int tb = 0; tb < 2; ++tb)
			for (int acc = 0; acc < 2; ++acc)
				for (size_t m : sizes)
					for (size_t n : sizes)
						for (size_t k : sizes)
							check(make_problem(ta, tb, m, n, k, 0, seed++), acc);
}

TEST_CASE("sgemm_portable matches reference past each block size") {
	struct Dims { size_t m, n, k; };
	const Dims cases[] = {{kMC + 5, 7, 33}, {7, 33, kKC + 5}, {7, kNC + 5, 33}, {7, kNC + 5, kKC + 5}};
	unsigned seed = 1000;
	for (int ta = 0; ta < 2; ++ta)
		for (int tb = 0; tb < 2; ++tb)
			for (int acc = 0; acc < 2; ++acc)
				for (const Dims& d : cases)
					check(make_problem(ta, tb, d.m, d.n, d.k, 0, seed++), acc);
}

TEST_CASE("sgemm_portable with zero-size dimensions") {
	struct Dims { size_t m, n, k; };
	const Dims cases[] = {{0, 5, 4}, {3, 0, 4}, {3, 5, 0}};
	for (const Dims& d : cases) {
		for (int acc = 0; acc < 2; ++acc) {
			const size_t ldc = 6;
			std::vector<float> a(16, 1.0f), b(32, 1.0f), c(4 * ldc, kSentinel);
			for (size_t i = 0; i < d.m; ++i)
				for (size_t j = 0; j < d.n; ++j) c[i * ldc + j] = 2.0f;
			sgemm_portable(false, false, d.m, d.n, d.k, a.data(), d.k ? d.k : 1, b.data(),
				d.n ? d.n : 1, c.data(), ldc, acc);
			INFO("m=", d.m, " n=", d.n, " k=", d.k, " acc=", acc);
			for (size_t i = 0; i < 4; ++i) {
				for (size_t j = 0; j < ldc; ++j) {
					bool inside = i < d.m && j < d.n;
					float want = inside ? (acc ? 2.0f : 0.0f) : kSentinel;
					CHECK(c[i * ldc + j] == want);
				}
			}
		}
	}
}

TEST_CASE("sgemm_portable respects leading dimensions wider than the width") {
	unsigned seed = 2000;
	for (int ta = 0; ta < 2; ++ta)
		for (int tb = 0; tb < 2; ++tb)
			for (int acc = 0; acc < 2; ++acc)
				check(make_problem(ta, tb, 7, 33, 33, 3, seed++), acc);
}

TEST_CASE("sgemm_portable propagates Inf and NaN like the reference") {
	const float inf = std::numeric_limits<float>::infinity();
	const float nan = std::numeric_limits<float>::quiet_NaN();
	unsigned seed = 3000;
	for (int ta = 0; ta < 2; ++ta) {
		for (int tb = 0; tb < 2; ++tb) {
			for (int acc = 0; acc < 2; ++acc) {
				Problem p = make_problem(ta, tb, 7, 33, 17, 0, seed++);
				// op(A)(2, 0) is +Inf and row 0 of op(B) cycles +1, -1, 0, so row 2
				// of C holds +Inf, -Inf and NaN (Inf * 0); op(B)(5, 3) is NaN, so
				// column 3 of C is NaN.
				p.a[ta ? 2 : 2 * p.lda] = inf;
				for (size_t j = 0; j < p.n; ++j)
					p.b[tb ? j * p.ldb : j] = j % 3 == 0 ? 1.0f : j % 3 == 1 ? -1.0f : 0.0f;
				p.b[tb ? 3 * p.ldb + 5 : 5 * p.ldb + 3] = nan;
				check(p, acc);
				check(p, acc, sgemm);
			}
		}
	}
}

TEST_CASE("sgemm_unpacked matches reference over sizes and flags") {
	const size_t sizes[] = {1, 7, 33, 257};
	unsigned seed = 6000;
	for (int ta = 0; ta < 2; ++ta)
		for (int tb = 0; tb < 2; ++tb)
			for (int acc = 0; acc < 2; ++acc)
				for (size_t m : sizes)
					for (size_t n : sizes)
						for (size_t k : sizes)
							check(make_problem(ta, tb, m, n, k, 0, seed++), acc, sgemm_unpacked);
}

TEST_CASE("sgemm_unpacked matches reference past each block size") {
	struct Dims { size_t m, n, k; };
	const Dims cases[] = {{kMC + 5, 7, 33}, {7, 33, kKC + 5}, {7, kNC + 5, 33}, {7, kNC + 5, kKC + 5}};
	unsigned seed = 7000;
	for (int ta = 0; ta < 2; ++ta)
		for (int tb = 0; tb < 2; ++tb)
			for (int acc = 0; acc < 2; ++acc)
				for (const Dims& d : cases)
					check(make_problem(ta, tb, d.m, d.n, d.k, 0, seed++), acc, sgemm_unpacked);
}

TEST_CASE("sgemm_unpacked with zero-size dimensions") {
	struct Dims { size_t m, n, k; };
	const Dims cases[] = {{0, 5, 4}, {3, 0, 4}, {3, 5, 0}};
	for (const Dims& d : cases) {
		for (int acc = 0; acc < 2; ++acc) {
			const size_t ldc = 6;
			std::vector<float> a(16, 1.0f), b(32, 1.0f), c(4 * ldc, kSentinel);
			for (size_t i = 0; i < d.m; ++i)
				for (size_t j = 0; j < d.n; ++j) c[i * ldc + j] = 2.0f;
			sgemm_unpacked(false, false, d.m, d.n, d.k, a.data(), d.k ? d.k : 1, b.data(),
				d.n ? d.n : 1, c.data(), ldc, acc);
			INFO("m=", d.m, " n=", d.n, " k=", d.k, " acc=", acc);
			for (size_t i = 0; i < 4; ++i) {
				for (size_t j = 0; j < ldc; ++j) {
					bool inside = i < d.m && j < d.n;
					float want = inside ? (acc ? 2.0f : 0.0f) : kSentinel;
					CHECK(c[i * ldc + j] == want);
				}
			}
		}
	}
}

TEST_CASE("sgemm_unpacked respects leading dimensions wider than the width") {
	unsigned seed = 8000;
	for (int ta = 0; ta < 2; ++ta)
		for (int tb = 0; tb < 2; ++tb)
			for (int acc = 0; acc < 2; ++acc)
				check(make_problem(ta, tb, 7, 33, 33, 3, seed++), acc, sgemm_unpacked);
}

TEST_CASE("sgemm_unpacked propagates Inf and NaN like the reference") {
	const float inf = std::numeric_limits<float>::infinity();
	const float nan = std::numeric_limits<float>::quiet_NaN();
	unsigned seed = 9000;
	for (int ta = 0; ta < 2; ++ta) {
		for (int tb = 0; tb < 2; ++tb) {
			for (int acc = 0; acc < 2; ++acc) {
				Problem p = make_problem(ta, tb, 7, 33, 17, 0, seed++);
				// op(A)(2, 0) is +Inf and row 0 of op(B) cycles +1, -1, 0, so row 2
				// of C holds +Inf, -Inf and NaN (Inf * 0); op(B)(5, 3) is NaN, so
				// column 3 of C is NaN.
				p.a[ta ? 2 : 2 * p.lda] = inf;
				for (size_t j = 0; j < p.n; ++j)
					p.b[tb ? j * p.ldb : j] = j % 3 == 0 ? 1.0f : j % 3 == 1 ? -1.0f : 0.0f;
				p.b[tb ? 3 * p.ldb + 5 : 5 * p.ldb + 3] = nan;
				check(p, acc, sgemm_unpacked);
			}
		}
	}
}

TEST_CASE("sgemm matches reference and portable path") {
	CHECK(sgemm_uses_avx2() == kExpectAvx2);
	const size_t sizes[] = {1, 7, 33, 257};
	unsigned seed = 4000;
	for (int ta = 0; ta < 2; ++ta)
		for (int tb = 0; tb < 2; ++tb)
			for (int acc = 0; acc < 2; ++acc)
				for (size_t m : sizes)
					for (size_t n : sizes)
						for (size_t k : sizes) {
							Problem p = make_problem(ta, tb, m, n, k, 0, seed++);
							check(p, acc, sgemm);
							check(p, acc, sgemm, sgemm_portable);
						}
	struct Dims { size_t m, n, k; };
	const Dims cases[] = {{kMC + 5, 7, 33}, {7, 33, kKC + 5}, {7, kNC + 5, 33}, {7, kNC + 5, kKC + 5}};
	for (int ta = 0; ta < 2; ++ta)
		for (int tb = 0; tb < 2; ++tb)
			for (int acc = 0; acc < 2; ++acc) {
				for (const Dims& d : cases)
					check(make_problem(ta, tb, d.m, d.n, d.k, 0, seed++), acc, sgemm);
				check(make_problem(ta, tb, 7, 33, 33, 3, seed++), acc, sgemm);
			}
}

TEST_CASE("sgemm is deterministic") {
	for (int acc = 0; acc < 2; ++acc) {
		Problem p = make_problem(false, true, 101, 75, 300, 2, 5000);
		std::vector<float> c1 = p.c, c2 = p.c;
		sgemm(p.ta, p.tb, p.m, p.n, p.k, p.a.data(), p.lda, p.b.data(), p.ldb, c1.data(), p.ldc, acc);
		sgemm(p.ta, p.tb, p.m, p.n, p.k, p.a.data(), p.lda, p.b.data(), p.ldb, c2.data(), p.ldc, acc);
		CHECK(std::memcmp(c1.data(), c2.data(), c1.size() * sizeof(float)) == 0);
	}
}

TEST_CASE("sgemm and sgemm_portable are bitwise identical for any thread count") {
	// Restores the thread count when the test scope ends
	struct Restore {
		size_t saved = num_threads();
		~Restore() { set_num_threads(saved); }
	} restore;
	const size_t ms[] = {1, 7, 33, 257, 600};
	const size_t nks[] = {33, 257};
	unsigned seed = 10000;
	for (Gemm fn : {static_cast<Gemm>(sgemm), static_cast<Gemm>(sgemm_portable)})
		for (int ta = 0; ta < 2; ++ta)
			for (int tb = 0; tb < 2; ++tb)
				for (int acc = 0; acc < 2; ++acc)
					for (size_t m : ms)
						for (size_t n : nks)
							for (size_t k : nks) {
								Problem p = make_problem(ta, tb, m, n, k, 2, seed++);
								set_num_threads(1);
								std::vector<float> want = p.c;
								fn(p.ta, p.tb, m, n, k, p.a.data(), p.lda, p.b.data(), p.ldb,
									want.data(), p.ldc, acc);
								for (size_t t : {2, 4, 8}) {
									set_num_threads(t);
									std::vector<float> got = p.c;
									fn(p.ta, p.tb, m, n, k, p.a.data(), p.lda, p.b.data(), p.ldb,
										got.data(), p.ldc, acc);
									INFO("threads=", t, " ta=", ta, " tb=", tb, " acc=", acc,
										" m=", m, " n=", n, " k=", k);
									CHECK(std::memcmp(got.data(), want.data(),
										got.size() * sizeof(float)) == 0);
								}
								set_num_threads(1);
								check(p, acc, fn);
							}
	// A problem past every block size, on 8 threads
	set_num_threads(8);
	check(make_problem(false, false, 2 * kMC + 5, kNC + 5, kKC + 5, 0, seed++), false, sgemm);
}

TEST_CASE("sgemm with a parallel B pack is bitwise identical across thread counts") {
	struct Restore {
		size_t saved = num_threads();
		~Restore() { set_num_threads(saved); }
	} restore;
	// Wide n and k past kKC, so each (jc, pc) block packs many column panels
	// (and a ragged last one) on the workers
	unsigned seed = 20000;
	for (Gemm fn : {static_cast<Gemm>(sgemm), static_cast<Gemm>(sgemm_portable)})
		for (int tb = 0; tb < 2; ++tb) {
			Problem p = make_problem(false, tb, 40, kNC + 37, kKC + 5, 0, seed++);
			set_num_threads(1);
			std::vector<float> want = p.c;
			fn(p.ta, p.tb, p.m, p.n, p.k, p.a.data(), p.lda, p.b.data(), p.ldb, want.data(), p.ldc, false);
			for (size_t t : {3, 8}) {
				set_num_threads(t);
				std::vector<float> got = p.c;
				fn(p.ta, p.tb, p.m, p.n, p.k, p.a.data(), p.lda, p.b.data(), p.ldb, got.data(), p.ldc, false);
				INFO("threads=", t, " tb=", tb);
				CHECK(std::memcmp(got.data(), want.data(), got.size() * sizeof(float)) == 0);
			}
		}
}

TEST_CASE("sgemm cuts a small deep product into pieces: matches the reference, same bits for any thread count") {
	struct Restore {
		size_t saved = num_threads();
		~Restore() { set_num_threads(saved); }
	} restore;
	// Small m x n with a depth of at least two depth blocks, whole and ragged, and the same
	// shapes just under the thresholds, which take the ordinary path
	struct Dims { size_t m, n, k; };
	const Dims cases[] = {{64, 64, 4 * kKC}, {40, 48, 2 * kKC + 77}, {7, 300, 3 * kKC}, {64, 64, 2 * kKC - 1}, {300, 300, 2 * kKC}};
	unsigned seed = 30000;
	for (Gemm fn : {static_cast<Gemm>(sgemm), static_cast<Gemm>(sgemm_portable)})
		for (int ta = 0; ta < 2; ++ta)
			for (int tb = 0; tb < 2; ++tb)
				for (int acc = 0; acc < 2; ++acc)
					for (const Dims& d : cases) {
						Problem p = make_problem(ta, tb, d.m, d.n, d.k, 3, seed++);
						set_num_threads(1);
						std::vector<float> want = p.c;
						fn(p.ta, p.tb, p.m, p.n, p.k, p.a.data(), p.lda, p.b.data(), p.ldb, want.data(), p.ldc, acc);
						for (size_t t : {2, 3, 8}) {
							set_num_threads(t);
							std::vector<float> got = p.c;
							fn(p.ta, p.tb, p.m, p.n, p.k, p.a.data(), p.lda, p.b.data(), p.ldb, got.data(), p.ldc, acc);
							INFO("threads=", t, " ta=", ta, " tb=", tb, " acc=", acc, " m=", d.m, " n=", d.n, " k=", d.k);
							CHECK(std::memcmp(got.data(), want.data(), got.size() * sizeof(float)) == 0);
						}
						set_num_threads(1);
						check(p, acc, fn);
					}
}

TEST_CASE("tanh_inplace gives tanh_one's bits at every position and stays within 2e-7 of std::tanh") {
	std::vector<float> xs;
	for (float x = -12.0f; x <= 12.0f; x += 0.0037f) xs.push_back(x);
	for (float x : {0.0f, -0.0f, 1e-30f, -1e-8f, 9.0f, -9.0f, 50.0f, -50.0f, 1e30f, -1e30f}) xs.push_back(x);
	// Every start offset and a length that is not a multiple of the vector width
	for (size_t off = 0; off < 9; ++off) {
		std::vector<float> got(xs.begin() + off, xs.end() - 3);
		tanh_inplace(got.data(), got.size());
		size_t bad_bits = 0, bad_value = 0;
		for (size_t i = 0; i < got.size(); ++i) {
			const float one = tanh_one(xs[i + off]);
			if (std::memcmp(&one, &got[i], sizeof one) != 0) ++bad_bits;
			if (std::fabs(static_cast<double>(got[i]) - std::tanh(static_cast<double>(xs[i + off]))) > 2e-7) ++bad_value;
		}
		CHECK(bad_bits == 0);
		CHECK(bad_value == 0);
	}
	CHECK(tanh_one(std::numeric_limits<float>::infinity()) == 1.0f);
	CHECK(tanh_one(-std::numeric_limits<float>::infinity()) == -1.0f);
	float nans[9];
	for (float& v : nans) v = std::numeric_limits<float>::quiet_NaN();
	tanh_inplace(nans, 9);
	for (float v : nans) CHECK(std::isnan(v));
}
