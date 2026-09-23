/**
 * dataloader.cpp
 * Author: Miguel Mochizuki Silva
 * Description: Batching and shuffling iterator over any Dataset
 */

#include "nn/data/dataloader.hpp"

#include <algorithm>
#include <numeric>

namespace nn {

DataLoader::DataLoader(const Dataset& dataset, size_t batch_size, bool shuffle)
	: dataset_(dataset), batch_size_(batch_size), shuffle_(shuffle), rng_(std::random_device{}()) {
	if (batch_size_ == 0) {
		throw TensorShapeError();
	}
	indices_.resize(dataset_.size());
	std::iota(indices_.begin(), indices_.end(), 0);
	reset_epoch();
}

size_t DataLoader::num_batches() const { return (dataset_.size() + batch_size_ - 1) / batch_size_; }

void DataLoader::reset_epoch() {
	if (shuffle_) {
		std::shuffle(indices_.begin(), indices_.end(), rng_);
	}
}

std::pair<Value, Tensor> DataLoader::get_batch(size_t batch_index) const {
	if (batch_index >= num_batches()) {
		throw TensorIndexError();
	}
	size_t start = batch_index * batch_size_;
	size_t end = std::min(start + batch_size_, dataset_.size());

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

}  // namespace nn
