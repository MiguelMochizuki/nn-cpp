/**
 * module.hpp
 * Author: Miguel Mochizuki Silva
 * Description: Base interface for neural network layers and containers
 */

#pragma once

#include "nn/autograd/value.hpp"

#include <string>
#include <utility>
#include <vector>

namespace nn {

/*
 * Base interface for anything that maps a Value to a Value and may own parameters.
 *
 * Attributes: none (stateless interface)
 */
class Module {
public:
	virtual ~Module() = default;

	/* Public, compute layer output from input
	 *
	 * Parameters:
	 * Value input: input tensor wrapped in the autograd graph
	 *
	 * Returns Value: output, graph-connected to input and this layer's parameters
	 */
	virtual Value forward(const Value& input) = 0;

	/* Public, list this layer's trainable parameters
	 *
	 * Parameters: none
	 *
	 * Returns vector<Value>: leaf Values with requires_grad() true; empty if this layer
	 * owns no parameters
	 */
	virtual std::vector<Value> parameters() const { return {}; }

	/* Public, list this layer's trainable parameters with stable names, in the same order
	 * as parameters(); used to label tensors in checkpoints
	 *
	 * Parameters: none
	 *
	 * Returns vector<pair<string, Value>>: (name, handle) pairs; empty if this layer owns no
	 * parameters
	 */
	virtual std::vector<std::pair<std::string, Value>> named_parameters() const { return {}; }
};

}  // namespace nn
