/**
 * adam.hpp
 * Author: Miguel Mochizuki Silva
 * Description: Adam optimizer with bias-corrected moment estimates
 */

#pragma once

#include "nn/optim/optimizer.hpp"

namespace nn {

/*
 * Adam update rule (Kingma & Ba): per parameter, m = b1*m + (1-b1)*g, v = b2*v + (1-b2)*g^2,
 * param -= lr * (m / (1-b1^t)) / (sqrt(v / (1-b2^t)) + eps).
 *
 * Attributes:
 * float lr_, beta1_, beta2_, eps_: hyperparameters
 * int64_t t_: number of steps taken
 * vector<Tensor> m_, v_: per-parameter first and second moment, same shape as each parameter,
 * indexed in the same order as params_
 */
class Adam : public Optimizer {
public:
	/* Public, construct an Adam optimizer
	 *
	 * Parameters:
	 * vector<Value> params: parameters to update
	 * float lr: learning rate, default 1e-3
	 * float beta1: first-moment decay, default 0.9
	 * float beta2: second-moment decay, default 0.999
	 * float eps: denominator stabilizer, default 1e-8
	 *
	 * Returns Adam: none (constructor)
	 */
	Adam(std::vector<Value> params, float lr = 1e-3f, float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f);

	/* Public, apply one update to every parameter from its current grad, advancing t_
	 *
	 * Parameters: none
	 *
	 * Returns void: none
	 */
	void step() override;
	float lr() const override { return lr_; }
	void set_lr(float lr) override { lr_ = lr; }
	/* Public, snapshot m_ and v_ moments and the step count t_ for checkpointing
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

protected:
	float lr_, beta1_, beta2_, eps_;
	int64_t t_ = 0;
	std::vector<Tensor> m_;
	std::vector<Tensor> v_;
};

}  // namespace nn
