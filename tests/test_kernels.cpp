/**
 * test_kernels.cpp
 * Author: Miguel Mochizuki Silva
 * Description: Unit tests for the GEMM kernels against the reference triple loop
 */

#include "doctest/doctest.h"
#include "nn/kernels/gemm.hpp"
#include "reference_gemm.hpp"

#include <cmath>
#include <limits>
#include <random>
#include <vector>

using namespace nn::kernels;

namespace {

constexpr float kSentinel = 1e30f;

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
 * Runs sgemm_portable and the reference on copies of C and checks they agree
 * elementwise (1e-5 relative to magnitude(), 1e-6 absolute floor, NaN/Inf
 * positions equal),
 * that C padding is untouched and that no sentinel leaked into the result
 *
 * Parameters:
 * const Problem& p: The problem
 * bool accumulate: Accumulate flag passed to both
 *
 * Returns void
 */
void check(const Problem& p, bool accumulate) {
	std::vector<float> got = p.c, want = p.c;
	sgemm_portable(p.ta, p.tb, p.m, p.n, p.k, p.a.data(), p.lda, p.b.data(), p.ldb,
		got.data(), p.ldc, accumulate);
	reference_sgemm(p.ta, p.tb, p.m, p.n, p.k, p.a.data(), p.lda, p.b.data(), p.ldb,
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
	const Dims cases[] = {{kMC + 5, 7, 33}, {7, 33, kKC + 5}, {7, kNC + 5, 33}};
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
				// Row 2 of op(A) is +Inf; element (5, 3) of op(B) is NaN.
				for (size_t q = 0; q < p.k; ++q)
					p.a[ta ? q * p.lda + 2 : 2 * p.lda + q] = inf;
				p.b[tb ? 3 * p.ldb + 5 : 5 * p.ldb + 3] = nan;
				check(p, acc);
			}
		}
	}
}
