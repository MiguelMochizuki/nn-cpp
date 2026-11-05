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
 * Attributes:
 * bool training_: true in training mode, false in evaluation mode
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

	/* Public, list this layer's non-trainable state tensors (e.g. batch-norm running
	 * statistics) with stable names; saved and restored by checkpoints. Names must not
	 * collide with parameter names. Not const so the pointers can be written on load
	 *
	 * Parameters: none
	 *
	 * Returns vector<pair<string, Tensor*>>: (name, pointer to the live tensor) pairs; empty
	 * by default
	 */
	virtual std::vector<std::pair<std::string, Tensor*>> named_buffers() { return {}; }

	/* Public, switch between training and evaluation behaviour (layers such as dropout and
	 * batch norm act differently); containers override this to reach their children
	 *
	 * Parameters:
	 * bool mode: true for training, false for evaluation
	 *
	 * Returns void: none
	 */
	virtual void train(bool mode = true) { training_ = mode; }

	/* Public, switch to evaluation behaviour; same as train(false)
	 *
	 * Parameters: none
	 *
	 * Returns void: none
	 */
	void eval() { train(false); }

	/* Public, tell whether the module is in training mode (the default)
	 *
	 * Parameters: none
	 *
	 * Returns bool: true in training mode
	 */
	bool is_training() const { return training_; }

protected:
	bool training_ = true;
};

}  // namespace nn
