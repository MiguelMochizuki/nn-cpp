/**
 * dataloader.cpp
 * Author: Miguel Mochizuki Silva
 * Description: Batching and shuffling iterator over any Dataset
 */

#include "nn/data/dataloader.hpp"
#include "nn/random.hpp"
#include "shuffle.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <stdexcept>

namespace nn {

DataLoader::DataLoader(const Dataset& dataset, size_t batch_size, bool shuffle)
	: DataLoader(dataset, batch_size, shuffle, false) {}

DataLoader::DataLoader(const Dataset& dataset, size_t batch_size, bool shuffle, bool drop_last)
	: dataset_(dataset), batch_size_(batch_size), shuffle_(shuffle), drop_last_(drop_last), num_samples_(dataset.size()), rng_(derive_generator("dataloader")) {
	if (batch_size_ == 0) {
		throw TensorShapeError();
	}
	indices_.resize(dataset_.size());
	std::iota(indices_.begin(), indices_.end(), 0);
	reset_epoch();
}

DataLoader::DataLoader(const Dataset& dataset, size_t batch_size, const std::vector<double>& sample_weights,
                       size_t num_samples, bool drop_last)
	: dataset_(dataset), batch_size_(batch_size), shuffle_(false), drop_last_(drop_last),
	  num_samples_(num_samples ? num_samples : dataset.size()), rng_(derive_generator("dataloader")) {
	if (batch_size_ == 0) {
		throw TensorShapeError();
	}
	if (sample_weights.size() != dataset.size()) {
		throw std::invalid_argument("DataLoader: need one sample weight per dataset sample");
	}
	double total = 0.0;
	cumulative_.reserve(sample_weights.size());
	for (double w : sample_weights) {
		if (!std::isfinite(w) || w < 0.0) throw std::invalid_argument("DataLoader: sample weights must be finite and non-negative");
		total += w;
		cumulative_.push_back(total);
	}
	if (!(total > 0.0)) throw std::invalid_argument("DataLoader: sample weights sum to zero");
	reset_epoch();
}

size_t DataLoader::num_batches() const {
	return drop_last_ ? num_samples_ / batch_size_ : (num_samples_ + batch_size_ - 1) / batch_size_;
}

void DataLoader::reset_epoch() {
	if (!cumulative_.empty()) {
		// 53 random bits from two mt19937 outputs: a uniform double in [0, 1) that is the same on every platform
		indices_.resize(num_samples_);
		for (size_t& index : indices_) {
			const std::uint64_t a = rng_() >> 5, b = rng_() >> 6;
			const double u = (static_cast<double>(a) * 67108864.0 + static_cast<double>(b)) / 9007199254740992.0;
			// first sample whose running sum exceeds the target; zero-weight samples are never chosen
			index = static_cast<size_t>(std::upper_bound(cumulative_.begin(), cumulative_.end(), u * cumulative_.back()) - cumulative_.begin());
			if (index >= cumulative_.size()) index = cumulative_.size() - 1;
		}
	} else if (shuffle_) {
		portable_shuffle(indices_, rng_);
	}
}

std::pair<Value, Tensor> DataLoader::get_batch(size_t batch_index) const {
	if (batch_index >= num_batches()) {
		throw TensorIndexError();
	}
	size_t start = batch_index * batch_size_;
	size_t end = std::min(start + batch_size_, indices_.size());

	std::vector<Tensor> x_items;
	std::vector<Tensor> y_items;
	x_items.reserve(end - start);
	y_items.reserve(end - start);
	for (size_t i = start; i < end; i++) {
		auto [x, y] = dataset_.get(indices_[i]);
		x_items.push_back(x);
		y_items.push_back(y);
	}

	Tensor x_batch = Tensor::stack(x_items);
	Tensor y_batch = Tensor::stack(y_items);
	return {Value::leaf(x_batch, false), y_batch};
}

std::vector<double> balanced_sample_weights(const std::vector<size_t>& labels, size_t num_classes) {
	std::vector<size_t> counts(num_classes, 0);
	for (size_t l : labels) {
		if (l >= num_classes) throw std::invalid_argument("balanced_sample_weights: label out of range");
		counts[l]++;
	}
	std::vector<double> w(labels.size());
	for (size_t i = 0; i < labels.size(); i++) w[i] = 1.0 / static_cast<double>(counts[labels[i]]);
	return w;
}

}  // namespace nn
