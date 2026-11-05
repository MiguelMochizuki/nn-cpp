/**
 * adamw.cpp
 * Author: Miguel Mochizuki Silva
 * Description: Adam with decoupled weight decay
 */

#include "nn/optim/adamw.hpp"

namespace nn {

AdamW::AdamW(std::vector<Value> params, float lr, float beta1, float beta2, float eps, float weight_decay)
	: Adam(std::move(params), lr, beta1, beta2, eps), weight_decay_(weight_decay) {}

void AdamW::step() {
	const float shrink = 1.0f - lr_ * weight_decay_;
	for (Value& p : params_) {
		for (float& w : p.data().data()) w *= shrink;
	}
	Adam::step();
}

}  // namespace nn
