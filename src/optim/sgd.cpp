/**
 * sgd.cpp
 * Author: Miguel Mochizuki Silva
 * Description: Stochastic gradient descent with optional momentum
 */

#include "nn/optim/sgd.hpp"
#include "nn/autograd/ops.hpp"

namespace nn {

Optimizer::Optimizer(std::vector<Value> params) : params_(std::move(params)) {}

void Optimizer::zero_grad() {
	for (Value& p : params_) p.zero_grad();
}

SGD::SGD(std::vector<Value> params, float lr, float momentum)
	: Optimizer(std::move(params)), lr_(lr), momentum_(momentum) {
	velocity_.reserve(params_.size());
	for (const Value& p : params_) {
		velocity_.push_back(Tensor::zeros(p.data().shape()));
	}
}

void SGD::step() {
	for (size_t i = 0; i < params_.size(); i++) {
		Tensor update = params_[i].grad();
		if (momentum_ > 0.0f) {
			velocity_[i] = add(scale(velocity_[i], momentum_), params_[i].grad());
			update = velocity_[i];
		}
		params_[i].data() = sub(params_[i].data(), scale(update, lr_));
	}
}

}  // namespace nn
