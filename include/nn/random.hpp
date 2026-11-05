/**
 * random.hpp
 * Author: Miguel Mochizuki Silva
 * Description: Library-wide seeding and per-component derived random generators
 */

#pragma once

#include <cstdint>
#include <random>
#include <string>

namespace nn {

/**
 * Seeds the library and resets every derived-stream counter, so the same seed replays
 * the same sequence of derive_generator calls
 *
 * Parameters:
 * std::uint64_t seed: base seed
 *
 * Returns void: none
 */
void manual_seed(std::uint64_t seed);

/**
 * Tells whether manual_seed has been called
 *
 * Parameters: none
 *
 * Returns bool: true after manual_seed
 */
bool is_seeded();

/**
 * Counts seedings, so a component that caches a generator can notice a re-seed.
 * For internal use by library components; user code has no need for it
 *
 * Parameters: none
 *
 * Returns std::uint64_t: increases on every manual_seed call
 */
std::uint64_t seed_epoch();

/**
 * Hands out a generator for a named stream. The k-th call with the same name after a
 * manual_seed is distinct and reproducible; other names do not affect it. Unseeded, the
 * base comes from std::random_device once, so results are not deterministic. Thread-safe.
 *
 * Parameters:
 * const std::string& stream: component name, e.g. "dataloader"
 *
 * Returns std::mt19937: a freshly seeded generator
 */
std::mt19937 derive_generator(const std::string& stream);

namespace detail {

/**
 * Internal, exposed so tests can pin the values. FNV-1a 64-bit hash of a stream name
 *
 * Parameters:
 * const std::string& s: stream name
 *
 * Returns std::uint64_t: the hash
 */
std::uint64_t fnv1a(const std::string& s);

/**
 * Internal, exposed so tests can pin the values. One splitmix64 mixing step
 *
 * Parameters:
 * std::uint64_t x: input
 *
 * Returns std::uint64_t: mixed output
 */
std::uint64_t splitmix64(std::uint64_t x);

}  // namespace detail

}  // namespace nn
