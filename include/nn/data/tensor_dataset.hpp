/**
 * tensor_dataset.hpp
 * Author: Miguel Mochizuki Silva
 * Description: In-memory Dataset backed by an inputs tensor and a targets tensor
 */

#pragma once

#include "nn/data/dataset.hpp"

#include <algorithm>
#include <utility>

namespace nn {

/*
 * Dataset whose sample i is row i of two tensors. Sample shapes are the tensor shapes
 * without the first dimension, so 1-D targets give scalar ({}) targets, as MnistDataset does.
 *
 * Attributes:
 * Tensor inputs_: all inputs, first dimension is the sample index
 * Tensor targets_: all targets, same first dimension
 */
class TensorDataset : public Dataset {
public:
	/* Public, build a dataset from two tensors
	 *
	 * Parameters:
	 * Tensor inputs: shape {N, ...}, N >= 1 and rank >= 1
	 * Tensor targets: shape {N, ...}
	 *
	 * Returns TensorDataset: none (constructor); throws TensorShapeError if the first dimensions differ or a tensor is rank 0
	 */
	TensorDataset(Tensor inputs, Tensor targets) : inputs_(std::move(inputs)), targets_(std::move(targets)) {
		if (inputs_.shape().empty() || targets_.shape().empty() || inputs_.shape()[0] != targets_.shape()[0]) {
			throw TensorShapeError();
		}
	}

	size_t size() const override { return inputs_.shape()[0]; }

	/* Public, fetch row index of both tensors
	 *
	 * Parameters:
	 * size_t index: sample index, 0-based
	 *
	 * Returns pair<Tensor, Tensor>: input row, target row; throws TensorIndexError if index >= size()
	 */
	std::pair<Tensor, Tensor> get(size_t index) const override {
		if (index >= size()) {
			throw TensorIndexError();
		}
		return {row(inputs_, index), row(targets_, index)};
	}

private:
	static Tensor row(const Tensor& t, size_t index) {
		std::vector<size_t> shape(t.shape().begin() + 1, t.shape().end());
		size_t n = t.size() / t.shape()[0];
		Tensor out(shape);
		const float* src = t.data().data() + index * n;
		std::copy(src, src + n, out.data().data());
		return out;
	}

	Tensor inputs_;
	Tensor targets_;
};

}  // namespace nn
