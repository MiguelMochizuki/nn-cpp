/**
 * sgd.hpp
 * Author: Miguel Mochizuki Silva
 * Description: Stochastic gradient descent with optional momentum
 */

#pragma once

#include "nn/optim/optimizer.hpp"

namespace nn {

/*
 * SGD update rule (PyTorch convention): g = grad + weight_decay * param; if momentum > 0,
 * velocity = momentum * velocity + g and step = (nesterov ? g + momentum * velocity : velocity),
 * else step = g; then param -= lr * step. Weight decay is plain L2 added to the gradient and
 * applies to every parameter, including biases and BatchNorm scales.
 *
 * Attributes:
 * float lr_: learning rate
 * float momentum_: momentum coefficient, 0 disables momentum
 * bool nesterov_: use the Nesterov look-ahead step, requires momentum > 0
 * float weight_decay_: L2 coefficient added to the gradient, 0 disables it
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
	 * bool nesterov: use Nesterov momentum, default false
	 * float weight_decay: L2 coefficient added to every parameter's gradient, default 0
	 *
	 * Returns SGD: none (constructor); throws std::invalid_argument if nesterov is set with momentum <= 0 or weight_decay is negative
	 */
	SGD(std::vector<Value> params, float lr, float momentum = 0.0f, bool nesterov = false, float weight_decay = 0.0f);

	/* Public, apply one update to every parameter from its current grad, updating velocity_ when momentum is on
	 *
	 * Parameters: none
	 *
	 * Returns void: none
	 */
	void step() override;
	float lr() const override { return lr_; }
	void set_lr(float lr) override { lr_ = lr; }
	/* Public, snapshot the momentum buffers for checkpointing
	 *
	 * Parameters: none
	 *
	 * Returns OptimizerState: named tensors that load_state() accepts
	 */
	OptimizerState state() const override;
	/* Public, restore a snapshot produced by state(); all-or-nothing
	 *
	 * Parameters:
	 * OptimizerState snapshot: snapshot to restore
	 *
	 * Returns void: none; a snapshot without "lr" keeps the current lr; throws std::runtime_error and leaves this optimizer unchanged on a name or shape mismatch
	 */
	void load_state(const OptimizerState& snapshot) override;

private:
	float lr_;
	float momentum_;
	bool nesterov_;
	float weight_decay_;
	std::vector<Tensor> velocity_;
};

}  // namespace nn
