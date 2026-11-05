/**
 * adamw.hpp
 * Author: Miguel Mochizuki Silva
 * Description: Adam with decoupled weight decay
 */

#pragma once

#include "nn/optim/adam.hpp"

namespace nn {

/*
 * AdamW (Loshchilov & Hutter): before each Adam update, param -= lr * weight_decay * param.
 * The decay is applied to the weights directly, not added to the gradient. Checkpoint state
 * has the same entries as Adam.
 *
 * Attributes:
 * float weight_decay_: decoupled decay coefficient
 */
class AdamW : public Adam {
public:
	/* Public, construct an AdamW optimizer
	 *
	 * Parameters:
	 * vector<Value> params: parameters to update
	 * float lr: learning rate, default 1e-3
	 * float beta1: first-moment decay, default 0.9
	 * float beta2: second-moment decay, default 0.999
	 * float eps: denominator stabilizer, default 1e-8
	 * float weight_decay: decoupled decay coefficient, default 1e-2
	 *
	 * Returns AdamW: none (constructor)
	 */
	AdamW(std::vector<Value> params, float lr = 1e-3f, float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f,
		float weight_decay = 1e-2f);

	/* Public, decay every parameter, then apply one Adam update from its current grad
	 *
	 * Parameters: none
	 *
	 * Returns void: none
	 */
	void step() override;

private:
	float weight_decay_;
};

}  // namespace nn
