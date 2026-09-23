/**
 * sgd.hpp
 * Author: Miguel Mochizuki Silva
 * Description: Stochastic gradient descent with optional momentum
 */

#pragma once

#include "nn/optim/optimizer.hpp"

namespace nn {

/*
 * SGD update rule: param -= lr * (velocity, if momentum > 0, else grad directly).
 *
 * Attributes:
 * float lr_: learning rate
 * float momentum_: momentum coefficient, 0 disables momentum
 * vector<Tensor> velocity_: per-parameter momentum buffer, same shape as each parameter,
 * indexed in the same order as params_
 */
class SGD : public Optimizer {
public:
	/* Public, construct an SGD optimizer
	 *
	 * Parameters:
	 * vector<Value> params: parameters to update
	 * float lr: learning rate
	 * float momentum: momentum coefficient, default 0
	 *
	 * Returns SGD: none (constructor)
	 */
	SGD(std::vector<Value> params, float lr, float momentum = 0.0f);

	void step() override;

private:
	float lr_;
	float momentum_;
	std::vector<Tensor> velocity_;
};

}  // namespace nn
