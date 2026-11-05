/**
 * batchnorm.hpp
 * Author: Miguel Mochizuki Silva
 * Description: Batch normalisation layers for {N, C} and {N, C, H, W} inputs
 */

#pragma once

#include "nn/nn/module.hpp"

namespace nn {

/*
 * Shared implementation of batch normalisation over the channel axis (axis 1). In training
 * mode it normalises with the batch mean and biased variance and updates the running
 * statistics; in evaluation mode it uses the running statistics. Use BatchNorm1d or
 * BatchNorm2d rather than this class directly.
 *
 * Attributes:
 * size_t num_features_: channel count C
 * size_t rank_: required input rank (2 or 4)
 * float eps_: added to the variance before the reciprocal square root
 * float momentum_: weight of the new batch statistic in the running average
 * Value weight_: learnable scale, shape {C}, initialised to 1
 * Value bias_: learnable shift, shape {C}, initialised to 0
 * Tensor running_mean_: shape {C}, initialised to 0; a buffer, saved by checkpoints
 * Tensor running_var_: shape {C}, initialised to 1; a buffer, saved by checkpoints
 */
class BatchNormBase : public Module {
public:
	/* Public, normalise the input per channel
	 *
	 * Parameters:
	 * Value input: shape {N, C} or {N, C, H, W} matching the layer; other ranks or a channel
	 * count different from num_features throw TensorShapeError
	 *
	 * Returns Value: same shape as input, weight * normalised + bias. Train mode with one
	 * element per channel stays finite (zero variance plus eps). Running variance is updated
	 * with the unbiased batch variance, and left unchanged when only one element per channel
	 * is seen
	 */
	Value forward(const Value& input) override;

	/* Public, list the learnable parameters
	 *
	 * Parameters: none
	 *
	 * Returns vector<Value>: {weight, bias}
	 */
	std::vector<Value> parameters() const override;

	/* Public, list the learnable parameters with names
	 *
	 * Parameters: none
	 *
	 * Returns vector<pair<string, Value>>: {"weight", ...}, {"bias", ...}
	 */
	std::vector<std::pair<std::string, Value>> named_parameters() const override;

	/* Public, list the running statistics
	 *
	 * Parameters: none
	 *
	 * Returns vector<pair<string, Tensor*>>: {"running_mean", ...}, {"running_var", ...}
	 */
	std::vector<std::pair<std::string, Tensor*>> named_buffers() const override;

protected:
	/* Protected, construct the shared state
	 *
	 * Parameters:
	 * size_t num_features: channel count C
	 * size_t rank: required input rank
	 * float eps: variance stabiliser
	 * float momentum: running-average weight of the new batch statistic
	 *
	 * Returns BatchNormBase: none (constructor)
	 */
	BatchNormBase(size_t num_features, size_t rank, float eps, float momentum);

private:
	size_t num_features_;
	size_t rank_;
	float eps_;
	float momentum_;
	Value weight_;
	Value bias_;
	mutable Tensor running_mean_;
	mutable Tensor running_var_;
};

/* Batch normalisation for inputs of shape {N, C} */
class BatchNorm1d : public BatchNormBase {
public:
	/* Public, construct the layer
	 *
	 * Parameters:
	 * size_t num_features: channel count C
	 * float eps: variance stabiliser
	 * float momentum: running-average weight of the new batch statistic
	 *
	 * Returns BatchNorm1d: none (constructor)
	 */
	explicit BatchNorm1d(size_t num_features, float eps = 1e-5f, float momentum = 0.1f)
		: BatchNormBase(num_features, 2, eps, momentum) {}
};

/* Batch normalisation for inputs of shape {N, C, H, W} */
class BatchNorm2d : public BatchNormBase {
public:
	/* Public, construct the layer
	 *
	 * Parameters:
	 * size_t num_features: channel count C
	 * float eps: variance stabiliser
	 * float momentum: running-average weight of the new batch statistic
	 *
	 * Returns BatchNorm2d: none (constructor)
	 */
	explicit BatchNorm2d(size_t num_features, float eps = 1e-5f, float momentum = 0.1f)
		: BatchNormBase(num_features, 4, eps, momentum) {}
};

}  // namespace nn
