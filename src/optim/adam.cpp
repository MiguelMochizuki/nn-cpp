/**
 * adam.cpp
 * Author: Miguel Mochizuki Silva
 * Description: Adam optimizer with bias-corrected moment estimates
 */

#include "nn/optim/adam.hpp"

#include "nn/kernels/parallel.hpp"

#include <cmath>
#include <string>

namespace nn {

Adam::Adam(std::vector<Value> params, float lr, float beta1, float beta2, float eps)
	: Optimizer(std::move(params)), lr_(lr), beta1_(beta1), beta2_(beta2), eps_(eps) {
	for (const Value& p : params_) {
		m_.push_back(Tensor::zeros(p.data().shape()));
		v_.push_back(Tensor::zeros(p.data().shape()));
	}
}

void Adam::step() {
	t_++;
	const float c1 = 1.0f - std::pow(beta1_, static_cast<float>(t_));
	const float c2 = 1.0f - std::pow(beta2_, static_cast<float>(t_));
	for (size_t i = 0; i < params_.size(); i++) {
		float* w = params_[i].data().data().data();
		const float* g = params_[i].grad().data().data();
		float* m = m_[i].data().data();
		float* v = v_[i].data().data();
		const float b1 = beta1_, b2 = beta2_, lr = lr_, eps = eps_;
		// Elements are independent, so ranges run on different threads without changing any result
		kernels::parallel_range(params_[i].data().size(), size_t(1) << 13, [=](size_t lo, size_t hi) {
			for (size_t k = lo; k < hi; k++) {
				m[k] = b1 * m[k] + (1.0f - b1) * g[k];
				v[k] = b2 * v[k] + (1.0f - b2) * g[k] * g[k];
				w[k] -= lr * (m[k] / c1) / (std::sqrt(v[k] / c2) + eps);
			}
		});
	}
}

// ponytail: t is stored as a float tensor, exact only to 2^24 steps; bias correction is saturated long before
OptimizerState Adam::state() const {
	OptimizerState out;
	out.emplace_back("lr", Tensor({1}, lr_));
	out.emplace_back("beta1", Tensor({1}, beta1_));
	out.emplace_back("beta2", Tensor({1}, beta2_));
	out.emplace_back("eps", Tensor({1}, eps_));
	out.emplace_back("t", Tensor({1}, static_cast<float>(t_)));
	for (size_t i = 0; i < m_.size(); i++) out.emplace_back("m." + std::to_string(i), m_[i]);
	for (size_t i = 0; i < v_.size(); i++) out.emplace_back("v." + std::to_string(i), v_[i]);
	return out;
}

void Adam::load_state(const OptimizerState& snapshot) {
	const OptimizerState state = with_lr(snapshot, lr_);
	check_state(state, this->state());
	for (const auto& entry : state) {
		const std::string& name = entry.first;
		if (name == "lr") lr_ = entry.second.data()[0];
		else if (name == "beta1") beta1_ = entry.second.data()[0];
		else if (name == "beta2") beta2_ = entry.second.data()[0];
		else if (name == "eps") eps_ = entry.second.data()[0];
		else if (name == "t") t_ = static_cast<int64_t>(entry.second.data()[0]);
		else if (name[0] == 'm') m_[std::stoul(name.substr(2))] = entry.second;
		else v_[std::stoul(name.substr(2))] = entry.second;
	}
}

}  // namespace nn
