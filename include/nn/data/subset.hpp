/**
 * subset.hpp
 * Author: Miguel Mochizuki Silva
 * Description: Index-selecting view over a Dataset, and random_split built on it
 */

#pragma once

#include "nn/data/dataset.hpp"

#include <cstdint>
#include <vector>

namespace nn {

/*
 * View of a Dataset restricted to, and ordered by, a list of indices.
 *
 * Attributes:
 * const Dataset* dataset_: underlying dataset, not owned
 * vector<size_t> indices_: indices into the underlying dataset
 */
class Subset : public Dataset {
public:
	/* Public, build a view
	 *
	 * Parameters:
	 * const Dataset& dataset: underlying dataset; must outlive this Subset
	 * vector<size_t> indices: indices into dataset, each < dataset.size()
	 *
	 * Returns Subset: none (constructor); throws TensorIndexError if an index is out of range
	 */
	Subset(const Dataset& dataset, std::vector<size_t> indices);

	size_t size() const override { return indices_.size(); }

	/* Public, fetch the sample at position index of the view
	 *
	 * Parameters:
	 * size_t index: position in the view
	 *
	 * Returns pair<Tensor, Tensor>: the underlying sample; throws TensorIndexError if index >= size()
	 */
	std::pair<Tensor, Tensor> get(size_t index) const override;

private:
	const Dataset* dataset_;
	std::vector<size_t> indices_;
};

/**
 * Splits a dataset into disjoint random subsets that cover it
 *
 * Parameters:
 * const Dataset& dataset: dataset to split; must outlive the subsets
 * const std::vector<double>& fractions: positive fractions summing to 1 within 1e-9
 * std::uint64_t seed: seeds the std::mt19937 that shuffles the indices
 *
 * Returns vector<Subset>: one subset per fraction, sizes floor(fraction * N) with the remainder
 * added to the last non-empty split; throws std::invalid_argument on bad fractions or an empty split
 */
std::vector<Subset> random_split(const Dataset& dataset, const std::vector<double>& fractions, std::uint64_t seed);

}  // namespace nn
