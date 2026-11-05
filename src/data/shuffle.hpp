/**
 * shuffle.hpp
 * Author: Miguel Mochizuki Silva
 * Description: Internal portable Fisher-Yates shuffle over raw mt19937 output (not public API)
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <random>
#include <utility>
#include <vector>

namespace nn {

/*
 * Shuffles in place with Fisher-Yates. Uses only mt19937 output and rejection sampling, never
 * std::*_distribution or std::shuffle, so the permutation for a seed is identical on every
 * standard library.
 *
 * Parameters:
 * std::vector<size_t>& v: values to permute
 * std::mt19937& rng: generator, advanced by at least v.size() - 1 draws
 *
 * Returns void
 */
inline void portable_shuffle(std::vector<size_t>& v, std::mt19937& rng) {
	for (size_t i = v.size(); i > 1; i--) {
		std::uint32_t bound = static_cast<std::uint32_t>(i);
		// reject draws in the incomplete top bucket so every residue is equally likely
		std::uint64_t limit = (std::uint64_t{1} << 32) - ((std::uint64_t{1} << 32) % bound);
		std::uint32_t r = static_cast<std::uint32_t>(rng());
		while (r >= limit) {
			r = static_cast<std::uint32_t>(rng());
		}
		std::swap(v[i - 1], v[r % bound]);
	}
}

}  // namespace nn
