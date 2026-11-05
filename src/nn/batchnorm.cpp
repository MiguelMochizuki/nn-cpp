/**
 * batchnorm.cpp
 * Author: Miguel Mochizuki Silva
 * Description: Batch normalisation layers for {N, C} and {N, C, H, W} inputs
 */

#include "nn/nn/batchnorm.hpp"

#include "nn/autograd/ops.hpp"

namespace nn {

BatchNormBase::BatchNormBase(size_t num_features, size_t rank, float eps, float momentum)
	: num_features_(num_features),
	  rank_(rank),
	  eps_(eps),
	  momentum_(momentum),
	  weight_(Value::leaf(Tensor({num_features}, 1.0f), true)),
	  bias_(Value::leaf(Tensor::zeros({num_features}), true)),
	  running_mean_({num_features}, 0.0f),
	  running_var_({num_features}, 1.0f) {}

Value BatchNormBase::forward(const Value& input) {
	const std::vector<size_t>& shape = input.data().shape();
	if (shape.size() != rank_ || shape[1] != num_features_) throw TensorShapeError();

	// Per-channel statistics keep every other axis as size 1 so they broadcast against the input
	std::vector<size_t> stat_shape(rank_, 1);
	stat_shape[1] = num_features_;

	if (training_) {
		Value mu = input;
		for (int axis = static_cast<int>(rank_) - 1; axis >= 0; axis--) {
			if (axis != 1) mu = mean(mu, axis, true);
		}
		Value centered = sub(input, mu);
		Value var = mul(centered, centered);
		for (int axis = static_cast<int>(rank_) - 1; axis >= 0; axis--) {
			if (axis != 1) var = mean(var, axis, true);
		}
		size_t n = input.data().size() / num_features_;
		if (n > 1) {
			float unbiased = static_cast<float>(n) / static_cast<float>(n - 1);
			for (size_t c = 0; c < num_features_; c++) {
				running_mean_.data()[c] = (1.0f - momentum_) * running_mean_.data()[c] + momentum_ * mu.data().data()[c];
				running_var_.data()[c] =
					(1.0f - momentum_) * running_var_.data()[c] + momentum_ * var.data().data()[c] * unbiased;
			}
		} else {
			for (size_t c = 0; c < num_features_; c++) {
				running_mean_.data()[c] = (1.0f - momentum_) * running_mean_.data()[c] + momentum_ * mu.data().data()[c];
			}
		}
		Value inv_std = rsqrt(add(var, Value::leaf(Tensor(stat_shape, eps_))));
		Value normed = mul(centered, inv_std);
		return add(mul(normed, reshape(weight_, stat_shape)), reshape(bias_, stat_shape));
	}

	Value run_mean = Value::leaf(Tensor(stat_shape, running_mean_.data()));
	Value run_var = Value::leaf(Tensor(stat_shape, running_var_.data()));
	Value inv_std = rsqrt(add(run_var, Value::leaf(Tensor(stat_shape, eps_))));
	Value normed = mul(sub(input, run_mean), inv_std);
	return add(mul(normed, reshape(weight_, stat_shape)), reshape(bias_, stat_shape));
}

std::vector<Value> BatchNormBase::parameters() const { return {weight_, bias_}; }

std::vector<std::pair<std::string, Value>> BatchNormBase::named_parameters() const {
	return {{"weight", weight_}, {"bias", bias_}};
}

std::vector<std::pair<std::string, Tensor*>> BatchNormBase::named_buffers() const {
	return {{"running_mean", &running_mean_}, {"running_var", &running_var_}};
}

}  // namespace nn
