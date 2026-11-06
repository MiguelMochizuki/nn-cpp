/**
 * layernorm.hpp
 * Author: Miguel Mochizuki Silva
 * Description: Layer normalisation over the last axis
 */

#pragma once

#include "nn/nn/module.hpp"

namespace nn {

/*
 * Normalises each row (last axis) to zero mean and unit biased variance, then applies a
 * learnable scale and shift. Composed from differentiable ops only.
 *
 * Attributes:
 * size_t dim_: size of the last axis
 * float eps_: added to the variance before the reciprocal square root
 * Value gamma_: scale, shape {dim}, initialised to 1
 * Value beta_: shift, shape {dim}, initialised to 0
 */
class LayerNorm : public Module {
public:
	/* Public, construct a layer norm
	 *
	 * Parameters:
	 * size_t dim: size of the last axis
	 * float eps: variance stabiliser
	 *
	 * Returns LayerNorm: none (constructor)
	 */
	explicit LayerNorm(size_t dim, float eps = 1e-5f);

	/* Public, normalise the last axis
	 *
	 * Parameters:
	 * Value input: any shape whose last dimension equals dim; otherwise throws TensorShapeError
	 *
	 * Returns Value: same shape as input, (x - mean) * rsqrt(var + eps) * gamma + beta
	 */
	Value forward(const Value& input) override;
	/* Public, list the trainable parameters
	 *
	 * Parameters: none
	 *
	 * Returns vector<Value>: {gamma, beta}, same order as named_parameters()
	 */
	std::vector<Value> parameters() const override;
	/* Public, list the trainable parameters with stable names
	 *
	 * Parameters: none
	 *
	 * Returns vector<pair<string, Value>>: {"gamma", "beta"} handles
	 */
	std::vector<std::pair<std::string, Value>> named_parameters() const override;

private:
	size_t dim_;
	float eps_;
	Value gamma_;
	Value beta_;
};

}  // namespace nn
