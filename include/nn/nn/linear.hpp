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

	Value forward(const Value& input) override;
	std::vector<Value> parameters() const override;

private:
	Value weight_;
	Value bias_;
};

}  // namespace nn
