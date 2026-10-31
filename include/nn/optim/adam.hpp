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

	void step() override;
	OptimizerState state() const override;
	void load_state(const OptimizerState& state) override;

private:
	float lr_, beta1_, beta2_, eps_;
	int64_t t_ = 0;
	std::vector<Tensor> m_;
	std::vector<Tensor> v_;
};

}  // namespace nn
