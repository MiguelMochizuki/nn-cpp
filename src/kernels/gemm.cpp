/**
 * gemm.cpp
 * Author: Miguel Mochizuki Silva
 * Description: Single-precision row-major GEMM kernels (cache-blocked, packed)
 */

#include "nn/kernels/gemm.hpp"
#include "nn/kernels/parallel.hpp"

#include <algorithm>
#include <new>
#include <vector>

#if defined(__AVX2__) && defined(__FMA__)
#define NN_GEMM_AVX2 1
#include <immintrin.h>
#elif defined(__SSE2__)
#define NN_GEMM_SSE2 1
#include <emmintrin.h>
#endif

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

	/**
	 * Frees the buffer
	 */
	~AlignedBuffer() { ::operator delete(p, std::align_val_t(64)); }

	/**
	 * Returns the buffer, allocating it on the first call. Later calls ignore
	 * floats, so each buffer must always be asked for the same size.
	 *
	 * Parameters:
	 * size_t floats: Buffer size in floats
	 *
	 * Returns float*: The 64-byte aligned buffer
	 */
	float* get(size_t floats) {
		if (!p) {
			p = static_cast<float*>(::operator new(floats * sizeof(float), std::align_val_t(64)));
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

#ifdef NN_GEMM_AVX2
/**
 * AVX2/FMA microkernel, same contract as micro_portable: 12 ymm accumulators
 * hold the kMR x kNR tile. Edge tiles run on a stack copy so nothing outside
 * C's mr x nr corner is read or written.
 *
 * Parameters: as micro_portable
 *
 * Returns void
 */
void micro_avx2(size_t kc, const float* ap, const float* bp, float* c, size_t ldc,
		size_t mr, size_t nr, Store mode) {
	static_assert(kMR == 6 && kNR == 16, "micro_avx2 is written for a 6 x 16 tile");
	alignas(32) float buf[kMR * kNR] = {};
	bool edge = mr < kMR || nr < kNR;
	float* t = c;
	size_t ldt = ldc;
	if (edge) {
		t = buf;
		ldt = kNR;
		if (mode != Store::kOverwrite)
			for (size_t i = 0; i < mr; ++i)
				for (size_t j = 0; j < nr; ++j) buf[i * kNR + j] = c[i * ldc + j];
	}
	__m256 acc[kMR][2];
	for (size_t i = 0; i < kMR; ++i) {
		if (mode == Store::kContinue) {
			acc[i][0] = _mm256_loadu_ps(t + i * ldt);
			acc[i][1] = _mm256_loadu_ps(t + i * ldt + 8);
		} else {
			acc[i][0] = acc[i][1] = _mm256_setzero_ps();
		}
	}
	for (size_t p = 0; p < kc; ++p, ap += kMR, bp += kNR) {
		__m256 b0 = _mm256_load_ps(bp), b1 = _mm256_load_ps(bp + 8);
		for (size_t i = 0; i < kMR; ++i) {
			__m256 av = _mm256_broadcast_ss(ap + i);
			acc[i][0] = _mm256_fmadd_ps(av, b0, acc[i][0]);
			acc[i][1] = _mm256_fmadd_ps(av, b1, acc[i][1]);
		}
	}
	for (size_t i = 0; i < kMR; ++i) {
		if (mode == Store::kAdd) {
			acc[i][0] = _mm256_add_ps(_mm256_loadu_ps(t + i * ldt), acc[i][0]);
			acc[i][1] = _mm256_add_ps(_mm256_loadu_ps(t + i * ldt + 8), acc[i][1]);
		}
		_mm256_storeu_ps(t + i * ldt, acc[i][0]);
		_mm256_storeu_ps(t + i * ldt + 8, acc[i][1]);
	}
	if (edge)
		for (size_t i = 0; i < mr; ++i)
			for (size_t j = 0; j < nr; ++j) c[i * ldc + j] = buf[i * kNR + j];
}
#endif

#ifdef NN_GEMM_SSE2
/**
 * SSE2 microkernel, same contract as micro_portable. A 6 x 16 tile needs 24
 * xmm accumulators but there are only 16 registers, so the tile is computed
 * as two 6 x 8 halves (12 accumulators each), each a full pass over kc that
 * reads its 8 columns of the shared packed B panel. Edge tiles run on a stack
 * copy so nothing outside C's mr x nr corner is read or written.
 *
 * Parameters: as micro_portable
 *
 * Returns void
 */
void micro_sse2(size_t kc, const float* ap, const float* bp, float* c, size_t ldc,
		size_t mr, size_t nr, Store mode) {
	static_assert(kMR == 6 && kNR == 16, "micro_sse2 is written for a 6 x 16 tile");
	alignas(16) float buf[kMR * kNR] = {};
	bool edge = mr < kMR || nr < kNR;
	float* t = c;
	size_t ldt = ldc;
	if (edge) {
		t = buf;
		ldt = kNR;
		if (mode != Store::kOverwrite)
			for (size_t i = 0; i < mr; ++i)
				for (size_t j = 0; j < nr; ++j) buf[i * kNR + j] = c[i * ldc + j];
	}
	for (size_t h = 0; h < kNR && h < nr; h += 8) {
		__m128 acc[kMR][2];
		for (size_t i = 0; i < kMR; ++i) {
			if (mode == Store::kContinue) {
				acc[i][0] = _mm_loadu_ps(t + i * ldt + h);
				acc[i][1] = _mm_loadu_ps(t + i * ldt + h + 4);
			} else {
				acc[i][0] = acc[i][1] = _mm_setzero_ps();
			}
		}
		const float* a = ap;
		const float* b = bp + h;
		for (size_t p = 0; p < kc; ++p, a += kMR, b += kNR) {
			__m128 b0 = _mm_load_ps(b), b1 = _mm_load_ps(b + 4);
			for (size_t i = 0; i < kMR; ++i) {
				__m128 av = _mm_set1_ps(a[i]);
				acc[i][0] = _mm_add_ps(acc[i][0], _mm_mul_ps(av, b0));
				acc[i][1] = _mm_add_ps(acc[i][1], _mm_mul_ps(av, b1));
			}
		}
		for (size_t i = 0; i < kMR; ++i) {
			if (mode == Store::kAdd) {
				acc[i][0] = _mm_add_ps(_mm_loadu_ps(t + i * ldt + h), acc[i][0]);
				acc[i][1] = _mm_add_ps(_mm_loadu_ps(t + i * ldt + h + 4), acc[i][1]);
			}
			_mm_storeu_ps(t + i * ldt + h, acc[i][0]);
			_mm_storeu_ps(t + i * ldt + h + 4, acc[i][1]);
		}
	}
	if (edge)
		for (size_t i = 0; i < mr; ++i)
			for (size_t j = 0; j < nr; ++j) c[i * ldc + j] = buf[i * kNR + j];
}
#endif

/**
 * Microkernel signature shared by micro_portable, micro_sse2 and micro_avx2
 */
using Micro = void (*)(size_t, const float*, const float*, float*, size_t, size_t, size_t, Store);

/**
 * Blocked GEMM driver: packs op(A) and op(B) block by block and runs micro
 * on every tile. Same contract as sgemm.
 *
 * Parameters:
 * Micro micro: Microkernel to run on each tile
 * Others: as sgemm
 *
 * Returns void
 */
void blocked(Micro micro, bool trans_a, bool trans_b, size_t m, size_t n, size_t k,
		const float* a, size_t lda, const float* b, size_t ldb,
		float* c, size_t ldc, bool accumulate) {
	if (m == 0 || n == 0) return;
	if (k == 0) {
		if (!accumulate)
			for (size_t i = 0; i < m; ++i) std::fill(c + i * ldc, c + i * ldc + n, 0.0f);
		return;
	}
	// The B block is packed once per (jc, pc) and shared read-only by the
	// workers; the pack is split by column panels, which write disjoint runs
	// of b_pack, so the packed bytes do not depend on the thread count.
	thread_local AlignedBuffer b_buf;
	float* b_pack = b_buf.get(kKC * kNC);
	const size_t blocks = (m + kMC - 1) / kMC;
	// Below this many multiply-adds (about a 101^3 product) the fork/join cost outweighs the gain
	constexpr size_t kParallelMin = size_t(1) << 20;
	// Work split targets: chunks per thread, and the narrowest column run (in kNR panels)
	constexpr size_t kChunksPerThread = 2;
	constexpr size_t kMinPanels = 4;
	// Fewest kNR panels per thread worth packing in parallel (measured: 4 and 8 tie, 1 loses on cnn_large)
	constexpr size_t kPackMinPanels = 8;
	const size_t threads = num_threads();
	const bool serial = threads == 1 || m * n * k < kParallelMin;

	for (size_t jc = 0; jc < n; jc += kNC) {
		size_t nc = std::min(kNC, n - jc);
		for (size_t pc = 0; pc < k; pc += kKC) {
			size_t kc = std::min(kKC, k - pc);
			Store mode = pc > 0 ? Store::kContinue : accumulate ? Store::kAdd : Store::kOverwrite;
			const size_t panels = (nc + kNR - 1) / kNR;
			auto pack = [&](size_t lo, size_t hi) {
				size_t j0 = lo * kNR;
				pack_b(trans_b, b, ldb, pc, jc + j0, kc, std::min(nc, hi * kNR) - j0, b_pack + lo * kc * kNR);
			};
			if (serial) pack(0, panels);
			else parallel_range(panels, kPackMinPanels, pack);
			// With few row blocks (small m) also split the columns so there are
			// about kChunksPerThread chunks per thread to balance dynamically.
			size_t col_chunks = 1;
			if (!serial && blocks < kChunksPerThread * threads)
				col_chunks = std::max<size_t>(1, std::min((kChunksPerThread * threads + blocks - 1) / blocks,
					panels / kMinPanels));
			const size_t panels_per_chunk = (panels + col_chunks - 1) / col_chunks;
			col_chunks = (panels + panels_per_chunk - 1) / panels_per_chunk;
			// Each chunk owns one kMC row block and a run of whole kNR column
			// panels of C, and packs its own A block, so there is no
			// cross-thread accumulation and every C element sees the same tile
			// sequence whatever the thread count.
			auto chunk = [&](size_t idx) {
				thread_local AlignedBuffer a_buf;
				float* a_pack = a_buf.get(kMC * kKC);
				size_t blk = idx / col_chunks;
				size_t j0 = (idx % col_chunks) * panels_per_chunk * kNR;
				size_t j1 = std::min(nc, j0 + panels_per_chunk * kNR);
				if (j0 >= j1) return;
				size_t ic = blk * kMC;
				size_t mc = std::min(kMC, m - ic);
				pack_a(trans_a, a, lda, ic, pc, mc, kc, a_pack);
				for (size_t jr = j0; jr < j1; jr += kNR) {
					size_t nr = std::min(kNR, nc - jr);
					for (size_t ir = 0; ir < mc; ir += kMR) {
						size_t mr = std::min(kMR, mc - ir);
						micro(kc, a_pack + ir * kc, b_pack + jr * kc,
							c + (ic + ir) * ldc + jc + jr, ldc, mr, nr, mode);
					}
				}
			};
			if (serial) {
				for (size_t idx = 0; idx < blocks * col_chunks; ++idx) chunk(idx);
			} else {
				parallel_for(blocks * col_chunks, chunk);
			}
		}
	}
}

}  // namespace

void sgemm_portable(bool trans_a, bool trans_b, size_t m, size_t n, size_t k,
		const float* a, size_t lda, const float* b, size_t ldb,
		float* c, size_t ldc, bool accumulate) {
	blocked(micro_portable, trans_a, trans_b, m, n, k, a, lda, b, ldb, c, ldc, accumulate);
}

void sgemm_unpacked(bool trans_a, bool trans_b, size_t m, size_t n, size_t k,
		const float* a, size_t lda, const float* b, size_t ldb,
		float* c, size_t ldc, bool accumulate) {
	if (m == 0 || n == 0) return;
	if (!accumulate) {
		for (size_t i = 0; i < m; ++i) std::fill(c + i * ldc, c + i * ldc + n, 0.0f);
	}
	// A transposed B is copied kBlock depth rows at a time into a row-major
	// buffer, so both layouts feed the same unit-stride axpy loop
	constexpr size_t kBlock = 64;
	std::vector<float> bt;
	if (trans_b) bt.resize(std::min(k, kBlock) * n);
	for (size_t q0 = 0; q0 < k; q0 += kBlock) {
		const size_t qn = std::min(kBlock, k - q0);
		const float* bb = b + q0 * ldb;
		size_t bld = ldb;
		if (trans_b) {
			for (size_t j = 0; j < n; ++j)
				for (size_t q = 0; q < qn; ++q) bt[q * n + j] = b[j * ldb + q0 + q];
			bb = bt.data();
			bld = n;
		}
		for (size_t i = 0; i < m; ++i) {
			float* ci = c + i * ldc;
			size_t q = 0;
			// Four depth steps per pass over the C row cut its load/store traffic
			for (; q + 4 <= qn; q += 4) {
				float a0, a1, a2, a3;
				if (trans_a) {
					const float* ap = a + (q0 + q) * lda + i;
					a0 = ap[0]; a1 = ap[lda]; a2 = ap[2 * lda]; a3 = ap[3 * lda];
				} else {
					const float* ap = a + i * lda + q0 + q;
					a0 = ap[0]; a1 = ap[1]; a2 = ap[2]; a3 = ap[3];
				}
				const float* b0 = bb + q * bld;
				const float* b1 = b0 + bld;
				const float* b2 = b1 + bld;
				const float* b3 = b2 + bld;
				for (size_t j = 0; j < n; ++j)
					ci[j] += a0 * b0[j] + a1 * b1[j] + a2 * b2[j] + a3 * b3[j];
			}
			for (; q < qn; ++q) {
				const float av = trans_a ? a[(q0 + q) * lda + i] : a[i * lda + q0 + q];
				const float* bq = bb + q * bld;
				for (size_t j = 0; j < n; ++j) ci[j] += av * bq[j];
			}
		}
	}
}

void sgemm(bool trans_a, bool trans_b, size_t m, size_t n, size_t k,
		const float* a, size_t lda, const float* b, size_t ldb,
		float* c, size_t ldc, bool accumulate) {
#ifdef NN_GEMM_AVX2
	blocked(micro_avx2, trans_a, trans_b, m, n, k, a, lda, b, ldb, c, ldc, accumulate);
#elif defined(NN_GEMM_SSE2)
	blocked(micro_sse2, trans_a, trans_b, m, n, k, a, lda, b, ldb, c, ldc, accumulate);
#else
	sgemm_unpacked(trans_a, trans_b, m, n, k, a, lda, b, ldb, c, ldc, accumulate);
#endif
}

bool sgemm_uses_avx2() {
#ifdef NN_GEMM_AVX2
	return true;
#else
	return false;
#endif
}

}  // namespace kernels
}  // namespace nn
