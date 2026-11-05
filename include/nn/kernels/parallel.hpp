/**
 * parallel.hpp
 * Author: Miguel Mochizuki Silva
 * Description: Fork/join thread pool with a deterministic-chunk parallel_for
 */

#pragma once

#include <algorithm>
#include <cstddef>
#include <functional>

namespace nn {
namespace kernels {

/**
 * Number of threads parallel_for may use, the caller included. Defaults to
 * the NN_NUM_THREADS environment variable when it is a positive integer,
 * otherwise std::thread::hardware_concurrency().
 *
 * Returns size_t: The thread count, at least 1
 */
size_t num_threads();

/**
 * Sets the thread count. The worker pool is resized on the next parallel_for
 * that needs it. Must not be called from inside a parallel_for chunk.
 *
 * Parameters:
 * size_t n: New thread count, 0 is treated as 1
 *
 * Returns void
 */
void set_num_threads(size_t n);

/**
 * Runs fn(0) .. fn(n_chunks - 1), each exactly once, and returns when all
 * have finished. The calling thread takes part; chunks are claimed from a
 * shared counter, so which thread runs which chunk varies between runs and
 * fn must not depend on it. Runs as a plain loop, without touching the pool,
 * when n_chunks <= 1, num_threads() == 1, or when called from a pool worker
 * (nested calls). An exception escaping fn terminates the process. Do not
 * call from static destructors or atexit handlers (the pool may already be
 * gone). NN_NUM_THREADS has no upper bound; a failed thread start terminates
 * the process.
 *
 * Parameters:
 * size_t n_chunks: Number of chunks
 * const std::function<void(size_t)>& fn: Work for one chunk index
 *
 * Returns void
 */
void parallel_for(size_t n_chunks, const std::function<void(size_t)>& fn);

/**
 * Splits [0, n) into contiguous ranges and runs fn(begin, end) on each, in parallel when each
 * range would hold at least min_per_chunk elements. fn must write only to elements in its own
 * range, so the result does not depend on the thread count.
 *
 * Parameters:
 * size_t n: Number of elements
 * size_t min_per_chunk: Smallest range worth a thread; below 2 * this the loop runs serially
 * Fn fn: Callable taking (size_t begin, size_t end)
 *
 * Returns void
 */
template <typename Fn>
void parallel_range(size_t n, size_t min_per_chunk, Fn fn) {
	size_t chunks = std::min(num_threads(), n / min_per_chunk);
	if (chunks <= 1) {
		fn(size_t{0}, n);
		return;
	}
	parallel_for(chunks, [&](size_t c) { fn(n * c / chunks, n * (c + 1) / chunks); });
}

}  // namespace kernels
}  // namespace nn
