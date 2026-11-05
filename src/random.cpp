/**
 * random.cpp
 * Author: Miguel Mochizuki Silva
 * Description: Library-wide seeding and per-component derived random generators
 */

#include "nn/random.hpp"

#include <cstdint>
#include <map>
#include <mutex>

namespace nn {

namespace {

std::mutex g_mutex;
bool g_seeded = false;
bool g_has_base = false;
std::uint64_t g_base = 0;
std::uint64_t g_epoch = 0;
std::map<std::string, std::uint64_t> g_counters;

}  // namespace

namespace detail {

std::uint64_t splitmix64(std::uint64_t x) {
	x += 0x9e3779b97f4a7c15ULL;
	x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
	x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
	return x ^ (x >> 31);
}

// FNV-1a: fixed across platforms, unlike std::hash
std::uint64_t fnv1a(const std::string& s) {
	std::uint64_t h = 0xcbf29ce484222325ULL;
	for (unsigned char c : s) {
		h = (h ^ c) * 0x100000001b3ULL;
	}
	return h;
}

}  // namespace detail

using detail::fnv1a;
using detail::splitmix64;

void manual_seed(std::uint64_t seed) {
	std::lock_guard<std::mutex> lock(g_mutex);
	g_seeded = true;
	g_has_base = true;
	g_base = seed;
	g_epoch++;
	g_counters.clear();
}

bool is_seeded() {
	std::lock_guard<std::mutex> lock(g_mutex);
	return g_seeded;
}

std::uint64_t seed_epoch() {
	std::lock_guard<std::mutex> lock(g_mutex);
	return g_epoch;
}

std::mt19937 derive_generator(const std::string& stream) {
	std::lock_guard<std::mutex> lock(g_mutex);
	if (!g_has_base) {
		std::random_device rd;
		g_base = (static_cast<std::uint64_t>(rd()) << 32) ^ rd();
		g_has_base = true;
	}
	std::uint64_t k = g_counters[stream]++;
	std::uint64_t s = splitmix64(g_base ^ fnv1a(stream) ^ (k * 0x9e3779b97f4a7c15ULL));
	std::seed_seq seq{static_cast<std::uint32_t>(s), static_cast<std::uint32_t>(s >> 32)};
	return std::mt19937(seq);
}

}  // namespace nn
