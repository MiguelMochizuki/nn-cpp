/**
 * gemm.hpp
 * Author: Miguel Mochizuki Silva
 * Description: Single-precision row-major GEMM kernels (cache-blocked, packed)
 */

#pragma once

#include <cstddef>

namespace nn {
namespace kernels {

/**
 * Microkernel register tile: MR rows of op(A) by NR columns of op(B). 6 x 16
 * matches the AVX2 microkernel (12 ymm accumulators); the portable path uses
 * the same tile so the packed layout is shared.
 */
constexpr size_t kMR = 6;
constexpr size_t kNR = 16;

/**
 * Cache block sizes, sized for an i5-12450HX P-core (L1d 48 KiB, L2 ~1.25 MiB):
 * - kKC: depth of one rank-kc update. A packed B micro-panel (kKC x kNR floats,
 *   16 KiB) plus an A micro-panel (kMR x kKC, 6 KiB) stay in L1d.
 * - kMC: rows of the packed A block (kMC x kKC floats, 96 KiB), resident in L2
 *   while every B micro-panel streams past it. Multiple of kMR.
 * - kNC: columns of the packed B block (kKC x kNC floats, 1 MiB), sized to the
 *   L2/L3 boundary so it is reused across every A block. Multiple of kNR.
 */
constexpr size_t kMC = 96;
constexpr size_t kKC = 256;
constexpr size_t kNC = 1024;

/**
 * Computes row-major C = op(A) * op(B), or C += op(A) * op(B) when
 * accumulate is set, using the best kernel built in. Only the m x n region
 * of C is written. C must not alias A or B.
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
void sgemm(bool trans_a, bool trans_b, size_t m, size_t n, size_t k,
		const float* a, size_t lda, const float* b, size_t ldb,
		float* c, size_t ldc, bool accumulate);

/**
 * Same contract as sgemm, always on the portable scalar microkernel, so
 * tests can exercise it in a native build.
 *
 * Parameters: as sgemm
 *
 * Returns void
 */
void sgemm_portable(bool trans_a, bool trans_b, size_t m, size_t n, size_t k,
		const float* a, size_t lda, const float* b, size_t ldb,
		float* c, size_t ldc, bool accumulate);

/**
 * Reports which microkernel sgemm was compiled with
 *
 * Returns bool: true for the AVX2/FMA microkernel, false for the portable one
 */
bool sgemm_uses_avx2();

}  // namespace kernels
}  // namespace nn
