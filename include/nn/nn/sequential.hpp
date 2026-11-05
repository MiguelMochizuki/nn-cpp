/**
 * sequential.hpp
 * Author: Miguel Mochizuki Silva
 * Description: Container that chains layers in order
 */

#pragma once

#include "nn/nn/module.hpp"

#include <memory>

namespace nn {

/*
 * Chains owned layers in order: forward feeds each layer's output to the next;
 * parameters() concatenates every layer's parameters.
 *
 * Attributes:
 * vector<unique_ptr<Module>> layers_: layers in application order
 */
class Sequential : public Module {
public:
	/* Public, append a layer to the end of the sequence
	 *
	 * Parameters:
	 * unique_ptr<Module> layer: layer to own and append
	 *
	 * Returns void: none
	 */
	void add(std::unique_ptr<Module> layer);

	/* Public, feed input through every layer in order
	 *
	 * Parameters:
	 * Value input: input tensor wrapped in the autograd graph
	 *
	 * Returns Value: output of the last layer; input unchanged if the sequence is empty
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
	std::vector<std::unique_ptr<Module>> layers_;
};

}  // namespace nn
