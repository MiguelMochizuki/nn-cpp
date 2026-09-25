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
		if (momentum_ > 0.0f) {
			velocity_[i] *= momentum_;
			velocity_[i] += params_[i].grad();
			params_[i].data() -= velocity_[i] * lr_;
		} else {
			params_[i].data() -= params_[i].grad() * lr_;
		}
	}
}

}  // namespace nn
