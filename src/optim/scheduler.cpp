/**
 * scheduler.cpp
 * Author: Miguel Mochizuki Silva
 * Description: Learning-rate schedulers that drive Optimizer::set_lr
 */

#include "nn/optim/scheduler.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace nn {

LRScheduler::LRScheduler(Optimizer& opt) : opt_(opt), base_lr_(opt.lr()) {}

void LRScheduler::step() {
	step_++;
	opt_.set_lr(compute(step_));
}

void LRScheduler::set_step(size_t n) {
	step_ = n;
	opt_.set_lr(compute(n));
}

StepLR::StepLR(Optimizer& opt, size_t step_size, float gamma) : LRScheduler(opt), step_size_(step_size), gamma_(gamma) {
	if (step_size == 0) throw std::invalid_argument("StepLR: step_size must be positive");
}

float StepLR::compute(size_t step) const {
	return base_lr_ * std::pow(gamma_, static_cast<float>(step / step_size_));
}

CosineAnnealingLR::CosineAnnealingLR(Optimizer& opt, size_t t_max, float eta_min)
	: LRScheduler(opt), t_max_(t_max), eta_min_(eta_min) {
	if (t_max == 0) throw std::invalid_argument("CosineAnnealingLR: t_max must be positive");
}

float CosineAnnealingLR::compute(size_t step) const {
	if (step >= t_max_) return eta_min_;  // exact endpoint, no cos rounding
	const float pi = std::acos(-1.0f);
	const float frac = static_cast<float>(step) / static_cast<float>(t_max_);
	return eta_min_ + (base_lr_ - eta_min_) * (1.0f + std::cos(pi * frac)) * 0.5f;
}

LinearWarmup::LinearWarmup(Optimizer& opt, std::unique_ptr<LRScheduler> inner, size_t warmup_steps)
	: LRScheduler(opt), inner_(std::move(inner)), warmup_steps_(warmup_steps) {
	if (!inner_) throw std::invalid_argument("LinearWarmup: inner scheduler is null");
	opt_.set_lr(compute(0));
}

float LinearWarmup::compute(size_t step) const {
	if (step < warmup_steps_) return base_lr_ * static_cast<float>(step) / static_cast<float>(warmup_steps_);
	return inner_->compute(step - warmup_steps_);
}

}  // namespace nn
