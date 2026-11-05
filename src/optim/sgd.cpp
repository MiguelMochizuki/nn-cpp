/**
 * sgd.cpp
 * Author: Miguel Mochizuki Silva
 * Description: Stochastic gradient descent with optional momentum
 */

#include "nn/optim/sgd.hpp"
#include "nn/autograd/ops.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace nn {

Optimizer::Optimizer(std::vector<Value> params) : params_(std::move(params)) {}

void Optimizer::zero_grad() {
	for (Value& p : params_) p.zero_grad();
}

float Optimizer::lr() const { throw std::logic_error("this optimizer has no learning rate"); }

void Optimizer::set_lr(float) { throw std::logic_error("this optimizer has no learning rate"); }

OptimizerState Optimizer::with_lr(const OptimizerState& snapshot, float current_lr) {
	OptimizerState out = snapshot;
	bool has = std::any_of(out.begin(), out.end(), [](const auto& e) { return e.first == "lr"; });
	if (!has) out.emplace_back("lr", Tensor({1}, current_lr));
	return out;
}

void Optimizer::load_state(const OptimizerState& state) { check_state(state, {}); }

void Optimizer::check_state(const OptimizerState& got, const OptimizerState& expected) {
	std::string problems;
	for (const auto& want : expected) {
		auto it = std::find_if(got.begin(), got.end(), [&](const auto& g) { return g.first == want.first; });
		if (it == got.end()) problems += " missing:" + want.first;
		else if (it->second.shape() != want.second.shape()) problems += " shape-mismatch:" + want.first;
	}
	for (const auto& g : got) {
		bool known = std::any_of(expected.begin(), expected.end(), [&](const auto& w) { return w.first == g.first; });
		if (!known) problems += " unexpected:" + g.first;
	}
	if (problems.empty() && got.size() != expected.size()) problems = " duplicate entries";
	if (!problems.empty()) throw std::runtime_error("optimizer state mismatch:" + problems);
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

OptimizerState SGD::state() const {
	OptimizerState out;
	out.emplace_back("lr", Tensor({1}, lr_));
	out.emplace_back("momentum", Tensor({1}, momentum_));
	for (size_t i = 0; i < velocity_.size(); i++) out.emplace_back("velocity." + std::to_string(i), velocity_[i]);
	return out;
}

void SGD::load_state(const OptimizerState& snapshot) {
	const OptimizerState state = with_lr(snapshot, lr_);
	check_state(state, this->state());
	for (const auto& entry : state) {
		if (entry.first == "lr") lr_ = entry.second.data()[0];
		else if (entry.first == "momentum") momentum_ = entry.second.data()[0];
		else velocity_[std::stoul(entry.first.substr(9))] = entry.second;
	}
}

}  // namespace nn
