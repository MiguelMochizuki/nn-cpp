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

	Value forward(const Value& input) override;
	std::vector<Value> parameters() const override;

private:
	std::vector<std::unique_ptr<Module>> layers_;
};

}  // namespace nn
