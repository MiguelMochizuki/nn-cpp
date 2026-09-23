/**
 * flatten.hpp
 * Author: Miguel Mochizuki Silva
 * Description: Layer that collapses every dimension after the batch dimension into one
 */

#pragma once

#include "nn/nn/module.hpp"
#include "nn/autograd/ops.hpp"

namespace nn {

/*
 * Reshapes {N, d1, d2, ...} to {N, d1*d2*...}. No parameters.
 *
 * Attributes: none
 */
class Flatten : public Module {
public:
	Value forward(const Value& input) override {
		size_t n = input.data().shape()[0];
		size_t rest = input.data().size() / n;
		return reshape(input, {n, rest});
	}
};

}  // namespace nn
