/**
 * reference_gemm.hpp
 * Author: Miguel Mochizuki Silva
 * Description: Plain triple-loop GEMM used as the ground truth in kernel tests
 */

#pragma once

#include <cstddef>

/**
 * Computes row-major C = op(A) * op(B), or C += op(A) * op(B), with a plain
 * triple loop. Same contract as nn::kernels::sgemm.
 *
 * Parameters:
 * bool trans_a: A is stored k x m and read transposed
 * bool trans_b: B is stored n x k and read transposed
 * size_t m: Rows of op(A) and C
 * size_t n: Columns of op(B) and C
 * size_t k: Columns of op(A), rows of op(B)
 * const float* a: A storage
 * size_t lda: Leading dimension of A storage
 * const float* b: B storage
 * size_t ldb: Leading dimension of B storage
 * float* c: C storage, m x n
 * size_t ldc: Leading dimension of C
 * bool accumulate: Add to C instead of overwriting it
 *
 * Returns void
 */
inline void reference_sgemm(bool trans_a, bool trans_b, size_t m, size_t n, size_t k,
		const float* a, size_t lda, const float* b, size_t ldb,
		float* c, size_t ldc, bool accumulate) {
	for (size_t i = 0; i < m; ++i) {
		for (size_t j = 0; j < n; ++j) {
			float sum = 0.0f;
			for (size_t p = 0; p < k; ++p) {
				float av = trans_a ? a[p * lda + i] : a[i * lda + p];
				float bv = trans_b ? b[j * ldb + p] : b[p * ldb + j];
				sum += av * bv;
			}
			c[i * ldc + j] = accumulate ? c[i * ldc + j] + sum : sum;
		}
	}
}
