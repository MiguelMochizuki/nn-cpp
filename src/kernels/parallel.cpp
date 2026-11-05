/**
 * parallel.cpp
 * Author: Miguel Mochizuki Silva
 * Description: Fork/join thread pool with a deterministic-chunk parallel_for
 */

#include "nn/kernels/parallel.hpp"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace nn {
namespace kernels {

namespace {

// 0 means not yet resolved from the environment
std::atomic<size_t> g_threads{0};

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
 * Workers that sleep between jobs. run() is serialised by the caller.
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
			active_ = workers_.size();
			++gen_;
		}
		cv_.notify_all();
		t_in_worker = true;
		drain();
		t_in_worker = false;
		std::unique_lock<std::mutex> lk(m_);
		done_cv_.wait(lk, [this] { return active_ == 0; });
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
		std::unique_lock<std::mutex> lk(m_);
		for (;;) {
			cv_.wait(lk, [&] { return stop_ || gen_ != seen; });
			if (stop_) return;
			seen = gen_;
			lk.unlock();
			drain();
			lk.lock();
			if (--active_ == 0) done_cv_.notify_one();
		}
	}

	std::vector<std::thread> workers_;
	std::mutex m_;
	std::condition_variable cv_, done_cv_;
	uint64_t gen_ = 0;
	bool stop_ = false;
	const std::function<void(size_t)>* fn_ = nullptr;
	size_t n_ = 0;
	std::atomic<size_t> next_{0};
	size_t active_ = 0;
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
