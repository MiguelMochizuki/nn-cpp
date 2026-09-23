/**
 * dataloader.hpp
 * Author: Miguel Mochizuki Silva
 * Description: Batching and shuffling iterator over any Dataset
 */

#pragma once

#include "nn/data/dataset.hpp"
#include "nn/autograd/value.hpp"

#include <random>
#include <utility>
#include <vector>

namespace nn {

/*
 * Iterable batcher over any Dataset implementation.
 *
 * Attributes:
 * const Dataset& dataset_: source data, not owned
 * size_t batch_size_: samples per batch
 * bool shuffle_: whether to reshuffle sample order each epoch
 * vector<size_t> indices_: current sample order
 * mt19937 rng_: random engine used when shuffle_ is true
 */
class DataLoader {
public:
	/* Public, construct a batching loader over a dataset
	 *
	 * Parameters:
	 * const Dataset& dataset: data to iterate; must outlive this DataLoader
	 * size_t batch_size: samples per batch
	 * bool shuffle: reshuffle order at the start of each epoch
	 *
	 * Returns DataLoader: none (constructor)
	 */
	DataLoader(const Dataset& dataset, size_t batch_size, bool shuffle);

	/* Public, number of batches per epoch, including a shorter final batch if the
	 * dataset size isn't a multiple of batch_size
	 *
	 * Parameters: none
	 *
	 * Returns size_t: ceil(dataset.size() / batch_size)
	 */
	size_t num_batches() const;

	/* Public, reshuffle sample order; no-op if shuffle was disabled at construction
	 *
	 * Parameters: none
	 *
	 * Returns void: none
	 */
	void reset_epoch();

	/* Public, fetch one batch by index within the current epoch
	 *
	 * Parameters:
	 * size_t batch_index: which batch to fetch, 0-based; the final batch may be
	 * smaller than batch_size
	 *
	 * Returns pair<Value, Tensor>: stacked input batch as a leaf Value
	 * (requires_grad() false), stacked target batch
	 */
	std::pair<Value, Tensor> get_batch(size_t batch_index) const;

private:
	const Dataset& dataset_;
	size_t batch_size_;
	bool shuffle_;
	std::vector<size_t> indices_;
	std::mt19937 rng_;
};

}  // namespace nn
