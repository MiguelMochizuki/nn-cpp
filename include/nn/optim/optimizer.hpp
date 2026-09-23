/**
 * optimizer.hpp
 * Author: Miguel Mochizuki Silva
 * Description: Base interface for parameter update rules
 */

#pragma once

#include "nn/autograd/value.hpp"

#include <vector>

namespace nn {

/*
 * Base interface for optimizers. Holds the parameter set it updates; operates
 * directly on Value data/grad, outside the autograd graph.
 *
 * Attributes:
 * vector<Value> params_: parameters this optimizer updates
 */
class Optimizer {
public:
	/* Public, construct an optimizer over a fixed parameter set
	 *
	 * Parameters:
	 * vector<Value> params: parameters to update on each step()
	 *
	 * Returns Optimizer: none (constructor)
	 */
	explicit Optimizer(std::vector<Value> params);

	virtual ~Optimizer() = default;

	/* Public, apply one update step to every parameter using its current grad
	 *
	 * Parameters: none
	 *
	 * Returns void: none
	 */
	virtual void step() = 0;

	/* Public, reset every tracked parameter's grad to zero
	 *
	 * Parameters: none
	 *
	 * Returns void: none
	 */
	void zero_grad();

protected:
	std::vector<Value> params_;
};

}  // namespace nn
