/**
 * subset.cpp
 * Author: Miguel Mochizuki Silva
 * Description: Index-selecting view over a Dataset, and random_split built on it
 */

#include "nn/data/subset.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>

namespace nn {

Subset::Subset(const Dataset& dataset, std::vector<size_t> indices) : dataset_(&dataset), indices_(std::move(indices)) {
	for (size_t i : indices_) {
		if (i >= dataset.size()) {
			throw TensorIndexError();
		}
	}
}

std::pair<Tensor, Tensor> Subset::get(size_t index) const {
	if (index >= indices_.size()) {
		throw TensorIndexError();
	}
	return dataset_->get(indices_[index]);
}

std::vector<Subset> random_split(const Dataset& dataset, const std::vector<double>& fractions, std::uint64_t seed) {
	if (fractions.empty()) {
		throw std::invalid_argument("random_split: no fractions given");
	}
	double total = 0.0;
	for (double f : fractions) {
		if (!(f > 0.0)) {
			throw std::invalid_argument("random_split: fractions must be positive");
		}
		total += f;
	}
	if (std::abs(total - 1.0) > 1e-9) {
		throw std::invalid_argument("random_split: fractions must sum to 1");
	}

	size_t n = dataset.size();
	std::vector<size_t> sizes;
	size_t assigned = 0;
	for (double f : fractions) {
		sizes.push_back(static_cast<size_t>(std::floor(f * static_cast<double>(n))));
		assigned += sizes.back();
	}
	// only reachable if fractions sum slightly above 1 and rounding overshoots
	if (assigned > n) {
		throw std::invalid_argument("random_split: fractions exceed the dataset size");
	}
	size_t last = sizes.size() - 1;
	while (last > 0 && sizes[last] == 0) {
		last--;
	}
	sizes[last] += n - assigned;
	for (size_t s : sizes) {
		if (s == 0) {
			throw std::invalid_argument("random_split: a split would be empty");
		}
	}

	std::vector<size_t> order(n);
	std::iota(order.begin(), order.end(), 0);
	std::mt19937 rng(static_cast<std::mt19937::result_type>(seed));
	std::shuffle(order.begin(), order.end(), rng);

	std::vector<Subset> out;
	size_t pos = 0;
	for (size_t s : sizes) {
		out.emplace_back(dataset, std::vector<size_t>(order.begin() + pos, order.begin() + pos + s));
		pos += s;
	}
	return out;
}

}  // namespace nn
