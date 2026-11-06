/**
 * dataloader.hpp
 * Author: Miguel Mochizuki Silva
 * Description: Batching and shuffling iterator over any Dataset
 */

#pragma once

#include "nn/data/dataset.hpp"
#include "nn/autograd/value.hpp"

#include <random>
#include <type_traits>
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
 * bool drop_last_: whether a final batch smaller than batch_size_ is skipped
 * vector<size_t> indices_: current sample order
 * vector<double> cumulative_: running sums of the sample weights; empty unless the loader is weighted
 * size_t num_samples_: samples per epoch (the dataset size unless the loader is weighted)
 * mt19937 rng_: random engine used when shuffle_ is true or the loader is weighted
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

	/* Public, construct a batching loader that can skip the short final batch
	 *
	 * Parameters:
	 * const Dataset& dataset: data to iterate; must outlive this DataLoader
	 * size_t batch_size: samples per batch
	 * bool shuffle: reshuffle order at the start of each epoch
	 * bool drop_last: skip the final batch if it has fewer than batch_size samples
	 *
	 * Returns DataLoader: none (constructor)
	 */
	DataLoader(const Dataset& dataset, size_t batch_size, bool shuffle, bool drop_last);

	/* Public, construct a loader that draws each epoch's samples at random with the given weights,
	 * with replacement (like PyTorch's WeightedRandomSampler). A sample with twice the weight is
	 * drawn twice as often; the draws are portable, so a seed gives the same epoch on every platform.
	 *
	 * Parameters:
	 * const Dataset& dataset: data to iterate; must outlive this DataLoader
	 * size_t batch_size: samples per batch
	 * const std::vector<double>& sample_weights: one non-negative weight per dataset sample, not all zero
	 * size_t num_samples: samples drawn per epoch, a count and not a flag; 0 means the dataset size
	 * bool drop_last: skip the final batch if it has fewer than batch_size samples
	 *
	 * Returns DataLoader: none (constructor); throws std::invalid_argument for a wrong number of weights,
	 * a negative or non-finite weight, or weights that sum to zero. Passing a bool as num_samples (a mix-up
	 * with the unweighted (dataset, batch_size, shuffle, drop_last) form) is rejected at compile time.
	 * A single-element braced list such as {1.0} selects the bool overload instead; pass a named std::vector<double>.
	 */
	DataLoader(const Dataset& dataset, size_t batch_size, const std::vector<double>& sample_weights,
	           size_t num_samples = 0, bool drop_last = false);

	/* Public, deleted guard: a bool in the num_samples position is almost surely a misplaced drop_last
	 *
	 * Parameters:
	 * const Dataset&, size_t, const std::vector<double>&: as the weighted constructor
	 * T: deduced as bool, the only type this overload accepts
	 *
	 * Returns DataLoader: none; any use fails to compile
	 */
	template <class T, std::enable_if_t<std::is_same<T, bool>::value, int> = 0>
	DataLoader(const Dataset&, size_t, const std::vector<double>&, T) = delete;

	/* Public, number of batches per epoch, including a shorter final batch if the
	 * samples per epoch aren't a multiple of batch_size
	 *
	 * Parameters: none
	 *
	 * Returns size_t: ceil(samples per epoch / batch_size), or the floor when drop_last is set
	 */
	size_t num_batches() const;

	/* Public, reshuffle the sample order, or redraw the samples of a weighted loader; no-op for an unshuffled,
	 * unweighted loader
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
	bool drop_last_;
	std::vector<size_t> indices_;
	std::vector<double> cumulative_;
	size_t num_samples_;
	std::mt19937 rng_;
};

/*
 * Per-sample weights that make every class equally likely to be drawn by the weighted DataLoader:
 * a sample of class c gets weight 1 / count(c).
 *
 * Parameters:
 * const std::vector<size_t>& labels: class of every sample
 * size_t num_classes: number of classes; every label must be below it
 *
 * Returns std::vector<double>: one weight per sample; throws std::invalid_argument for an out-of-range label
 */
std::vector<double> balanced_sample_weights(const std::vector<size_t>& labels, size_t num_classes);

}  // namespace nn
