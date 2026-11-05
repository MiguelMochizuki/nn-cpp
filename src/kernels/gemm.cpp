/**
 * gemm.cpp
 * Author: Miguel Mochizuki Silva
 * Description: Single-precision row-major GEMM kernels (cache-blocked, packed)
 */

#include "nn/kernels/gemm.hpp"

#include <algorithm>
#include <cstdlib>
#include <new>

namespace nn {
namespace kernels {

namespace {

static_assert(kMC % kMR == 0, "kMC must be a multiple of kMR");
static_assert(kNC % kNR == 0, "kNC must be a multiple of kNR");

/**
 * Owns one 64-byte aligned packing buffer, allocated on first use
 *
 * Attributes:
 * float* p: The buffer, or nullptr before first use
 */
struct AlignedBuffer {
	float* p = nullptr;
	~AlignedBuffer() { std::free(p); }
	float* get(size_t floats) {
		if (!p) {
			p = static_cast<float*>(std::aligned_alloc(64, floats * sizeof(float)));
			if (!p) throw std::bad_alloc();
		}
		return p;
	}
};

/**
 * Packs the mc x kc block of op(A) at (ic, pc) into kMR-row panels: panel r
 * holds, for each p, kMR consecutive values of column p. Rows past mc are 0.
 *
 * Parameters:
 * bool trans_a, const float* a, size_t lda: op(A) as passed to sgemm
 * size_t ic, size_t pc: Block origin in op(A)
 * size_t mc, size_t kc: Block size
 * float* dst: Packed output, ceil(mc / kMR) * kMR * kc floats
 *
 * Returns void
 */
void pack_a(bool trans_a, const float* a, size_t lda, size_t ic, size_t pc,
		size_t mc, size_t kc, float* dst) {
	for (size_t ir = 0; ir < mc; ir += kMR) {
		size_t mr = std::min(kMR, mc - ir);
		for (size_t p = 0; p < kc; ++p) {
			for (size_t i = 0; i < kMR; ++i) {
				size_t row = ic + ir + i, col = pc + p;
				*dst++ = i < mr ? (trans_a ? a[col * lda + row] : a[row * lda + col]) : 0.0f;
			}
		}
	}
}

/**
 * Packs the kc x nc block of op(B) at (pc, jc) into kNR-column panels: panel
 * j holds, for each p, kNR consecutive values of row p. Columns past nc are 0.
 *
 * Parameters:
 * bool trans_b, const float* b, size_t ldb: op(B) as passed to sgemm
 * size_t pc, size_t jc: Block origin in op(B)
 * size_t kc, size_t nc: Block size
 * float* dst: Packed output, kc * ceil(nc / kNR) * kNR floats
 *
 * Returns void
 */
void pack_b(bool trans_b, const float* b, size_t ldb, size_t pc, size_t jc,
		size_t kc, size_t nc, float* dst) {
	for (size_t jr = 0; jr < nc; jr += kNR) {
		size_t nr = std::min(kNR, nc - jr);
		for (size_t p = 0; p < kc; ++p) {
			for (size_t j = 0; j < kNR; ++j) {
				size_t row = pc + p, col = jc + jr + j;
				*dst++ = j < nr ? (trans_b ? b[col * ldb + row] : b[row * ldb + col]) : 0.0f;
			}
		}
	}
}

/**
 * How a microkernel tile combines with C: overwrite it, add to it, or start
 * from it (a later depth block continuing the same sum)
 */
enum class Store { kOverwrite, kAdd, kContinue };

/**
 * Scalar microkernel: multiplies one packed A panel by one packed B panel
 * into a kMR x kNR tile and stores its top-left mr x nr corner into C
 *
 * Parameters:
 * size_t kc: Depth
 * const float* ap: Packed A panel, kc * kMR floats
 * const float* bp: Packed B panel, kc * kNR floats
 * float* c: Top-left of the C tile
 * size_t ldc: Leading dimension of C
 * size_t mr, size_t nr: Valid rows and columns of the tile
 * Store mode: How the tile combines with C
 *
 * Returns void
 */
void micro_portable(size_t kc, const float* ap, const float* bp, float* c, size_t ldc,
		size_t mr, size_t nr, Store mode) {
	float acc[kMR][kNR] = {};
	// Continuing a sum from an earlier depth block keeps the naive loop's
	// summation order, so results match it bit for bit when not accumulating.
	if (mode == Store::kContinue)
		for (size_t i = 0; i < mr; ++i)
			for (size_t j = 0; j < nr; ++j) acc[i][j] = c[i * ldc + j];
	for (size_t p = 0; p < kc; ++p, ap += kMR, bp += kNR)
		for (size_t i = 0; i < kMR; ++i)
			for (size_t j = 0; j < kNR; ++j) acc[i][j] += ap[i] * bp[j];
	for (size_t i = 0; i < mr; ++i)
		for (size_t j = 0; j < nr; ++j)
			c[i * ldc + j] = mode == Store::kAdd ? c[i * ldc + j] + acc[i][j] : acc[i][j];
}

}  // namespace

void sgemm_portable(bool trans_a, bool trans_b, size_t m, size_t n, size_t k,
		const float* a, size_t lda, const float* b, size_t ldb,
		float* c, size_t ldc, bool accumulate) {
	if (m == 0 || n == 0) return;
	if (k == 0) {
		if (!accumulate)
			for (size_t i = 0; i < m; ++i) std::fill(c + i * ldc, c + i * ldc + n, 0.0f);
		return;
	}
	thread_local AlignedBuffer a_buf, b_buf;
	float* a_pack = a_buf.get(kMC * kKC);
	float* b_pack = b_buf.get(kKC * kNC);

	for (size_t jc = 0; jc < n; jc += kNC) {
		size_t nc = std::min(kNC, n - jc);
		for (size_t pc = 0; pc < k; pc += kKC) {
			size_t kc = std::min(kKC, k - pc);
			Store mode = pc > 0 ? Store::kContinue : accumulate ? Store::kAdd : Store::kOverwrite;
			pack_b(trans_b, b, ldb, pc, jc, kc, nc, b_pack);
			for (size_t ic = 0; ic < m; ic += kMC) {
				size_t mc = std::min(kMC, m - ic);
				pack_a(trans_a, a, lda, ic, pc, mc, kc, a_pack);
				for (size_t jr = 0; jr < nc; jr += kNR) {
					size_t nr = std::min(kNR, nc - jr);
					for (size_t ir = 0; ir < mc; ir += kMR) {
						size_t mr = std::min(kMR, mc - ir);
						micro_portable(kc, a_pack + ir * kc, b_pack + jr * kc,
							c + (ic + ir) * ldc + jc + jr, ldc, mr, nr, mode);
					}
				}
			}
		}
	}
}

void sgemm(bool trans_a, bool trans_b, size_t m, size_t n, size_t k,
		const float* a, size_t lda, const float* b, size_t ldb,
		float* c, size_t ldc, bool accumulate) {
	sgemm_portable(trans_a, trans_b, m, n, k, a, lda, b, ldb, c, ldc, accumulate);
}

}  // namespace kernels
}  // namespace nn
