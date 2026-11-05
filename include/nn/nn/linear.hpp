/**
 * linear.hpp
 * Author: Miguel Mochizuki Silva
 * Description: Fully-connected layer
 */

#pragma once

#include "nn/nn/module.hpp"

namespace nn {

/*
 * Fully-connected layer: output = input @ weight^T + bias.
 *
 * Attributes:
 * Value weight_: shape {out_features, in_features}, requires_grad() true
 * Value bias_: shape {out_features}, requires_grad() true
 */
class Linear : public Module {
public:
	/* Public, construct a layer with randomly-initialized weight and zero bias
	 *
	 * Parameters:
	 * size_t in_features: input feature count
	 * size_t out_features: output feature count
	 *
	 * Returns Linear: none (constructor)
	 */
	Linear(size_t in_features, size_t out_features);

	/* Public, compute input @ weight^T + bias
	 *
	 * Parameters:
	 * Value input: input tensor wrapped in the autograd graph
	 *
	 * Returns Value: input @ weight^T + bias, with out_features as the last dimension
	 */
	Value forward(const Value& input) override;
	/* Public, list the trainable parameters
	 *
	 * Parameters: none
	 *
	 * Returns vector<Value>: the parameters, in the same order as named_parameters()
	 */
	std::vector<Value> parameters() const override;
	/* Public, list the trainable parameters with stable names
	 *
	 * Parameters: none
	 *
	 * Returns vector<pair<string, Value>>: (name, handle) pairs, in the same order as parameters()
	 */
	std::vector<std::pair<std::string, Value>> named_parameters() const override;

private:
	Value weight_;
	Value bias_;
};

}  // namespace nn
