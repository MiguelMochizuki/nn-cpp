/**
 * parallel.cpp
 * Author: Miguel Mochizuki Silva
 * Description: Fork/join thread pool with a deterministic-chunk parallel_for
 */

#include "nn/kernels/parallel.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
#include <immintrin.h>
#define NN_CPU_RELAX() _mm_pause()
#else
#define NN_CPU_RELAX() std::this_thread::yield()
#endif

namespace nn {
namespace kernels {

namespace {

// 0 means not yet resolved from the environment
std::atomic<size_t> g_threads{0};

// Spin budget in microseconds; negative means not yet resolved from the environment
std::atomic<long> g_spin_us{-1};
constexpr long kDefaultSpinUs = 100;
constexpr long kMaxSpinUs = 10000;

// True on pool workers and on a caller inside a job, so nested parallel_for runs serially
thread_local bool t_in_worker = false;

/**
 * Parses NN_NUM_THREADS
 *
 * Returns size_t: The positive value, or 0 when unset or invalid
 */
size_t env_threads() {
	const char* s = std::getenv("NN_NUM_THREADS");
	if (!s) return 0;
	char* end = nullptr;
	long v = std::strtol(s, &end, 10);
	return (end != s && *end == '\0' && v > 0) ? static_cast<size_t>(v) : 0;
}

/**
 * Parses NN_SPIN_US
 *
 * Returns long: The value clamped to [0, kMaxSpinUs], or kDefaultSpinUs when unset or invalid
 */
long env_spin_us() {
	const char* s = std::getenv("NN_SPIN_US");
	if (!s) return kDefaultSpinUs;
	char* end = nullptr;
	long v = std::strtol(s, &end, 10);
	if (end == s || *end != '\0' || v < 0) return kDefaultSpinUs;
	return std::min(v, kMaxSpinUs);
}

/**
 * Busy-waits until done() holds or the budget runs out
 *
 * Parameters:
 * Pred done: Condition to wait for; must be cheap and safe to call without a lock
 * long budget_us: Longest time to spin, in microseconds; 0 only tests the condition once
 *
 * Returns bool: The last value of done()
 */
template <typename Pred>
bool spin_until(Pred done, long budget_us) {
	if (done()) return true;
	if (budget_us <= 0) return false;
	const auto deadline = std::chrono::steady_clock::now() + std::chrono::microseconds(budget_us);
	for (;;) {
		// The clock is read once per batch of pauses, so the wait reacts within tens of nanoseconds
		for (int i = 0; i < 16; ++i) {
			if (done()) return true;
			NN_CPU_RELAX();
		}
		if (std::chrono::steady_clock::now() >= deadline) return done();
	}
}

/**
 * Workers that spin briefly, then sleep, between jobs. run() is serialised by the caller.
 * A training step issues hundreds of short jobs back to back; waking a sleeping worker costs
 * several microseconds per job per thread, so workers first spin for spin_microseconds() and
 * only then block on the condition variable. The caller waits for the job the same way.
 */
class Pool {
public:
	/**
	 * Starts n worker threads
	 */
	explicit Pool(size_t n) {
		for (size_t i = 0; i < n; ++i) workers_.emplace_back([this] { loop(); });
	}

	/**
	 * Stops and joins the workers
	 */
	~Pool() {
		{
			std::lock_guard<std::mutex> lk(m_);
			stop_ = true;
		}
		cv_.notify_all();
		for (std::thread& t : workers_) t.join();
	}

	size_t size() const { return workers_.size(); }

	/**
	 * Runs one job on the workers plus the calling thread and waits for it
	 */
	void run(size_t n, const std::function<void(size_t)>& fn) {
		{
			std::lock_guard<std::mutex> lk(m_);
			fn_ = &fn;
			n_ = n;
			next_.store(0, std::memory_order_relaxed);
			active_.store(workers_.size(), std::memory_order_relaxed);
			gen_.store(gen_.load(std::memory_order_relaxed) + 1, std::memory_order_release);
		}
		// gen_ changed under the mutex, so a worker that has not yet gone to sleep sees it in its
		// wait predicate; only workers already asleep need the wake-up
		if (sleepers_.load(std::memory_order_acquire) > 0) cv_.notify_all();
		t_in_worker = true;
		drain();
		t_in_worker = false;
		auto idle = [this] { return active_.load(std::memory_order_acquire) == 0; };
		if (spin_until(idle, static_cast<long>(spin_microseconds()))) return;
		std::unique_lock<std::mutex> lk(m_);
		done_cv_.wait(lk, idle);
	}

private:
	/**
	 * Claims and runs chunks until none are left. A throwing chunk would leave
	 * the other threads running on a dead job, so it terminates instead.
	 */
	void drain() {
		try {
			for (size_t i; (i = next_.fetch_add(1, std::memory_order_relaxed)) < n_;) (*fn_)(i);
		} catch (...) {
			std::terminate();
		}
	}

	void loop() {
		t_in_worker = true;
		uint64_t seen = 0;
		for (;;) {
			auto posted = [&] { return gen_.load(std::memory_order_acquire) != seen || stop_.load(std::memory_order_acquire); };
			if (!spin_until(posted, static_cast<long>(spin_microseconds()))) {
				std::unique_lock<std::mutex> lk(m_);
				sleepers_.fetch_add(1, std::memory_order_release);
				cv_.wait(lk, posted);
				sleepers_.fetch_sub(1, std::memory_order_release);
			}
			if (stop_.load(std::memory_order_acquire)) return;
			seen = gen_.load(std::memory_order_acquire);
			drain();
			// The caller may be asleep in run(): the last worker wakes it under the mutex
			if (active_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
				std::lock_guard<std::mutex> lk(m_);
				done_cv_.notify_one();
			}
		}
	}

	std::vector<std::thread> workers_;
	std::mutex m_;
	std::condition_variable cv_, done_cv_;
	std::atomic<uint64_t> gen_{0};
	std::atomic<bool> stop_{false};
	const std::function<void(size_t)>* fn_ = nullptr;
	size_t n_ = 0;
	std::atomic<size_t> next_{0};
	std::atomic<size_t> active_{0};
	std::atomic<size_t> sleepers_{0};
};

std::mutex g_call_mutex;  // one job at a time; also guards g_pool
std::unique_ptr<Pool> g_pool;

}  // namespace

size_t num_threads() {
	size_t t = g_threads.load();
	if (t) return t;
	t = env_threads();
	if (!t) t = std::thread::hardware_concurrency();
	if (!t) t = 1;
	size_t expected = 0;
	g_threads.compare_exchange_strong(expected, t);
	return g_threads.load();
}

void set_num_threads(size_t n) {
	g_threads.store(n ? n : 1);
}

size_t spin_microseconds() {
	long s = g_spin_us.load(std::memory_order_relaxed);
	if (s >= 0) return static_cast<size_t>(s);
	long expected = -1;
	g_spin_us.compare_exchange_strong(expected, env_spin_us());
	return static_cast<size_t>(g_spin_us.load());
}

void set_spin_microseconds(size_t us) {
	g_spin_us.store(static_cast<long>(std::min<size_t>(us, static_cast<size_t>(kMaxSpinUs))));
}

void parallel_for(size_t n_chunks, const std::function<void(size_t)>& fn) {
	size_t t = num_threads();
	if (n_chunks <= 1 || t == 1 || t_in_worker) {
		for (size_t i = 0; i < n_chunks; ++i) fn(i);
		return;
	}
	std::lock_guard<std::mutex> lk(g_call_mutex);
	if (!g_pool || g_pool->size() != t - 1) {
		g_pool.reset();
		g_pool.reset(new Pool(t - 1));
	}
	g_pool->run(n_chunks, fn);
}

}  // namespace kernels
}  // namespace nn
