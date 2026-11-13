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
 * Owns one 64-byte aligned packing buffer that grows on demand
 *
 * Attributes:
 * float* p: The buffer, or nullptr before first use
 * size_t cap: Its size in floats
 */
struct AlignedBuffer {
	float* p = nullptr;
	size_t cap = 0;

	/**
	 * Frees the buffer
	 */
	~AlignedBuffer() { ::operator delete(p, std::align_val_t(64)); }

	/**
	 * Returns a buffer of at least the given size. Growing it discards the old contents, which
	 * is fine for packing: every use fills the buffer before reading it.
	 *
	 * Parameters:
	 * size_t floats: Smallest acceptable size in floats
	 *
	 * Returns float*: The 64-byte aligned buffer
	 */
	float* get(size_t floats) {
		if (floats > cap) {
			::operator delete(p, std::align_val_t(64));
			p = nullptr;
			cap = 0;
			p = static_cast<float*>(::operator new(floats * sizeof(float), std::align_val_t(64)));
			cap = floats;
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
	// Both layouts are walked along the source's unit stride, with no test per element
	for (size_t ir = 0; ir < mc; ir += kMR, dst += kMR * kc) {
		const size_t mr = std::min(kMR, mc - ir);
		if (mr == kMR) {
			// Full tiles, the bulk of the work, with a fixed row count the compiler can unroll
			if (trans_a) {
				for (size_t p = 0; p < kc; ++p) {
					const float* src = a + (pc + p) * lda + ic + ir;
					for (size_t i = 0; i < kMR; ++i) dst[p * kMR + i] = src[i];
				}
			} else {
				const float* src = a + (ic + ir) * lda + pc;
				for (size_t p = 0; p < kc; ++p)
					for (size_t i = 0; i < kMR; ++i) dst[p * kMR + i] = src[i * lda + p];
			}
		} else if (trans_a) {
			for (size_t p = 0; p < kc; ++p) {
				const float* src = a + (pc + p) * lda + ic + ir;
				float* d = dst + p * kMR;
				for (size_t i = 0; i < mr; ++i) d[i] = src[i];
				for (size_t i = mr; i < kMR; ++i) d[i] = 0.0f;
			}
		} else {
			for (size_t i = 0; i < mr; ++i) {
				const float* src = a + (ic + ir + i) * lda + pc;
				for (size_t p = 0; p < kc; ++p) dst[p * kMR + i] = src[p];
			}
			for (size_t i = mr; i < kMR; ++i)
				for (size_t p = 0; p < kc; ++p) dst[p * kMR + i] = 0.0f;
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
	for (size_t jr = 0; jr < nc; jr += kNR, dst += kNR * kc) {
		const size_t nr = std::min(kNR, nc - jr);
		if (nr == kNR && !trans_b) {
			for (size_t p = 0; p < kc; ++p) {
				const float* src = b + (pc + p) * ldb + jc + jr;
				for (size_t j = 0; j < kNR; ++j) dst[p * kNR + j] = src[j];
			}
		} else if (trans_b) {
			for (size_t j = 0; j < nr; ++j) {
				const float* src = b + (jc + jr + j) * ldb + pc;
				for (size_t p = 0; p < kc; ++p) dst[p * kNR + j] = src[p];
			}
			for (size_t j = nr; j < kNR; ++j)
				for (size_t p = 0; p < kc; ++p) dst[p * kNR + j] = 0.0f;
		} else {
			for (size_t p = 0; p < kc; ++p) {
				const float* src = b + (pc + p) * ldb + jc + jr;
				float* d = dst + p * kNR;
				for (size_t j = 0; j < nr; ++j) d[j] = src[j];
				for (size_t j = nr; j < kNR; ++j) d[j] = 0.0f;
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
 * AVX2/FMA microkernel, same contract as micro_portable: 12 ymm accumulators,
 * each a named local so all of them stay in registers, hold the kMR x kNR
 * tile. Edge tiles run on a stack copy so nothing outside C's mr x nr corner
 * is read or written.
 *
 * Parameters: as micro_portable
 *
 * Returns void
 */
void micro_avx2(size_t kc, const float* ap, const float* bp, float* c, size_t ldc,
		size_t mr, size_t nr, Store mode) {
	static_assert(kMR == 6 && kNR == 16, "micro_avx2 is written for a 6 x 16 tile");
	alignas(32) float buf[kMR * kNR];
	const bool edge = mr < kMR || nr < kNR;
	float* t = c;
	size_t ldt = ldc;
	if (edge) {
		for (size_t i = 0; i < kMR * kNR; ++i) buf[i] = 0.0f;
		t = buf;
		ldt = kNR;
		if (mode != Store::kOverwrite)
			for (size_t i = 0; i < mr; ++i)
				for (size_t j = 0; j < nr; ++j) buf[i * kNR + j] = c[i * ldc + j];
	}
	float* r0 = t;
	float* r1 = t + ldt;
	float* r2 = t + 2 * ldt;
	float* r3 = t + 3 * ldt;
	float* r4 = t + 4 * ldt;
	float* r5 = t + 5 * ldt;
	// Twelve named accumulators: as an array the compiler keeps them in memory and stores ten of
	// them back to the stack on every depth step
	__m256 c00, c01, c10, c11, c20, c21, c30, c31, c40, c41, c50, c51;
	if (mode == Store::kContinue) {
		c00 = _mm256_loadu_ps(r0); c01 = _mm256_loadu_ps(r0 + 8);
		c10 = _mm256_loadu_ps(r1); c11 = _mm256_loadu_ps(r1 + 8);
		c20 = _mm256_loadu_ps(r2); c21 = _mm256_loadu_ps(r2 + 8);
		c30 = _mm256_loadu_ps(r3); c31 = _mm256_loadu_ps(r3 + 8);
		c40 = _mm256_loadu_ps(r4); c41 = _mm256_loadu_ps(r4 + 8);
		c50 = _mm256_loadu_ps(r5); c51 = _mm256_loadu_ps(r5 + 8);
	} else {
		c00 = c01 = c10 = c11 = c20 = c21 = c30 = c31 = c40 = c41 = c50 = c51 = _mm256_setzero_ps();
	}
	for (size_t p = 0; p < kc; ++p, ap += kMR, bp += kNR) {
		const __m256 b0 = _mm256_load_ps(bp), b1 = _mm256_load_ps(bp + 8);
		__m256 a;
		a = _mm256_broadcast_ss(ap + 0); c00 = _mm256_fmadd_ps(a, b0, c00); c01 = _mm256_fmadd_ps(a, b1, c01);
		a = _mm256_broadcast_ss(ap + 1); c10 = _mm256_fmadd_ps(a, b0, c10); c11 = _mm256_fmadd_ps(a, b1, c11);
		a = _mm256_broadcast_ss(ap + 2); c20 = _mm256_fmadd_ps(a, b0, c20); c21 = _mm256_fmadd_ps(a, b1, c21);
		a = _mm256_broadcast_ss(ap + 3); c30 = _mm256_fmadd_ps(a, b0, c30); c31 = _mm256_fmadd_ps(a, b1, c31);
		a = _mm256_broadcast_ss(ap + 4); c40 = _mm256_fmadd_ps(a, b0, c40); c41 = _mm256_fmadd_ps(a, b1, c41);
		a = _mm256_broadcast_ss(ap + 5); c50 = _mm256_fmadd_ps(a, b0, c50); c51 = _mm256_fmadd_ps(a, b1, c51);
	}
	if (mode == Store::kAdd) {
		c00 = _mm256_add_ps(_mm256_loadu_ps(r0), c00); c01 = _mm256_add_ps(_mm256_loadu_ps(r0 + 8), c01);
		c10 = _mm256_add_ps(_mm256_loadu_ps(r1), c10); c11 = _mm256_add_ps(_mm256_loadu_ps(r1 + 8), c11);
		c20 = _mm256_add_ps(_mm256_loadu_ps(r2), c20); c21 = _mm256_add_ps(_mm256_loadu_ps(r2 + 8), c21);
		c30 = _mm256_add_ps(_mm256_loadu_ps(r3), c30); c31 = _mm256_add_ps(_mm256_loadu_ps(r3 + 8), c31);
		c40 = _mm256_add_ps(_mm256_loadu_ps(r4), c40); c41 = _mm256_add_ps(_mm256_loadu_ps(r4 + 8), c41);
		c50 = _mm256_add_ps(_mm256_loadu_ps(r5), c50); c51 = _mm256_add_ps(_mm256_loadu_ps(r5 + 8), c51);
	}
	_mm256_storeu_ps(r0, c00); _mm256_storeu_ps(r0 + 8, c01);
	_mm256_storeu_ps(r1, c10); _mm256_storeu_ps(r1 + 8, c11);
	_mm256_storeu_ps(r2, c20); _mm256_storeu_ps(r2 + 8, c21);
	_mm256_storeu_ps(r3, c30); _mm256_storeu_ps(r3 + 8, c31);
	_mm256_storeu_ps(r4, c40); _mm256_storeu_ps(r4 + 8, c41);
	_mm256_storeu_ps(r5, c50); _mm256_storeu_ps(r5 + 8, c51);
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
 * bool force_serial: Never use the pool, whatever the size
 * Others: as sgemm
 *
 * Returns void
 */
void blocked_flat(Micro micro, bool trans_a, bool trans_b, size_t m, size_t n, size_t k,
		const float* a, size_t lda, const float* b, size_t ldb,
		float* c, size_t ldc, bool accumulate, bool force_serial) {
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
	// Work split targets: chunks per thread, and the fewest multiply-adds worth a chunk of its own
	constexpr size_t kChunksPerThread = 2;
	constexpr size_t kMinChunkWork = size_t(1) << 17;
	// Fewest kNR panels per thread worth packing in parallel (measured: 4 and 8 tie, 1 loses on cnn_large)
	constexpr size_t kPackMinPanels = 8;
	const size_t threads = num_threads();
	const bool serial = force_serial || threads == 1 || m * n * k < kParallelMin;

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
			// With few row blocks (small m) also split the columns, down to one kNR panel per
			// chunk, so there are about kChunksPerThread chunks per thread to balance
			// dynamically: a small deep product (a Linear weight gradient, out x in x rows)
			// would otherwise be one or two chunks however many threads there are.
			size_t col_chunks = 1;
			if (!serial && blocks < kChunksPerThread * threads)
				col_chunks = std::max<size_t>(1, std::min({(kChunksPerThread * threads + blocks - 1) / blocks,
					panels, std::min(m, kMC) * nc * kc / kMinChunkWork}));
			const size_t panels_per_chunk = (panels + col_chunks - 1) / col_chunks;
			col_chunks = (panels + panels_per_chunk - 1) / panels_per_chunk;
			// Several chunks then share each row block: op(A) is packed once, a block per
			// thread, instead of once per chunk. That only happens below kChunksPerThread *
			// threads blocks, which bounds the shared buffer.
			const bool shared_a = col_chunks > 1;
			float* a_shared = nullptr;
			if (shared_a) {
				thread_local AlignedBuffer a_all;
				a_shared = a_all.get(blocks * kMC * kKC);
				auto pack_block = [&](size_t blk) {
					size_t ic = blk * kMC;
					pack_a(trans_a, a, lda, ic, pc, std::min(kMC, m - ic), kc, a_shared + ic * kc);
				};
				if (blocks > 1) parallel_for(blocks, pack_block);
				else pack_block(0);
			}
			// Each chunk owns one kMC row block and a run of whole kNR column
			// panels of C, so there is no cross-thread accumulation and every C
			// element sees the same tile sequence whatever the thread count.
			auto chunk = [&](size_t idx) {
				size_t blk = idx / col_chunks;
				size_t j0 = (idx % col_chunks) * panels_per_chunk * kNR;
				size_t j1 = std::min(nc, j0 + panels_per_chunk * kNR);
				if (j0 >= j1) return;
				size_t ic = blk * kMC;
				size_t mc = std::min(kMC, m - ic);
				const float* a_pack = a_shared + ic * kc;
				if (!shared_a) {
					thread_local AlignedBuffer a_buf;
					float* own = a_buf.get(kMC * kKC);
					pack_a(trans_a, a, lda, ic, pc, mc, kc, own);
					a_pack = own;
				}
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

/**
 * GEMM driver, same contract as sgemm. A product with a small output and a long depth (a Linear
 * weight gradient: out x in, summed over every row of the batch) has too few tiles of C to give
 * each thread one, and its depth blocks, which is where the work is, would run one after another.
 * Such a product is instead cut along the depth into pieces of kKC / 2, each piece is computed into
 * its own m x n buffer by one thread, and the pieces are added in depth order. The cut depends
 * only on m, n and k, never on the thread count, and so does every sum.
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
	// Largest output, in elements, handled piecewise, and the fewest multiply-adds worth it
	constexpr size_t kDepthMaxOut = size_t(1) << 16;
	constexpr size_t kDepthMinWork = size_t(1) << 20;
	// Depth of one piece. Half a depth block, so that the 1024 rows of a typical batch give eight
	// pieces rather than four (measured: 28-35 us against 43-50 us for a 64 x 64 x 1024 product)
	constexpr size_t kPiece = kKC / 2;
	if (m == 0 || n == 0 || m * n > kDepthMaxOut || k < 4 * kPiece || m * n * k < kDepthMinWork) {
		blocked_flat(micro, trans_a, trans_b, m, n, k, a, lda, b, ldb, c, ldc, accumulate, false);
		return;
	}
	const size_t pieces = (k + kPiece - 1) / kPiece;
	thread_local AlignedBuffer part_buf;
	float* part = part_buf.get(pieces * m * n);
	parallel_for(pieces, [&](size_t p) {
		const size_t pc = p * kPiece, kc = std::min(kPiece, k - pc);
		// Depth range [pc, pc + kc) of op(A) and op(B) is a column or row offset into the stored matrix
		blocked_flat(micro, trans_a, trans_b, m, n, kc, trans_a ? a + pc * lda : a + pc, lda,
			trans_b ? b + pc : b + pc * ldb, ldb, part + p * m * n, n, false, true);
	});
	for (size_t i = 0; i < m; ++i) {
		float* acc = part + i * n;
		for (size_t p = 1; p < pieces; ++p) {
			const float* q = part + p * m * n + i * n;
			for (size_t j = 0; j < n; ++j) acc[j] += q[j];
		}
		float* ci = c + i * ldc;
		if (accumulate) for (size_t j = 0; j < n; ++j) ci[j] += acc[j];
		else for (size_t j = 0; j < n; ++j) ci[j] = acc[j];
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
