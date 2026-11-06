/**
 * plateau.cpp
 * Author: Miguel Mochizuki Silva
 * Description: ReduceLROnPlateau implementation
 */

#include "nn/optim/plateau.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace nn {

ReduceLROnPlateau::ReduceLROnPlateau(Optimizer& opt, PlateauMode mode, float factor, size_t patience,
                                     float threshold, size_t cooldown, float min_lr)
	: opt_(opt), mode_(mode), factor_(factor), patience_(patience), threshold_(threshold),
	  cooldown_(cooldown), min_lr_(min_lr),
	  best_(mode == PlateauMode::Min ? std::numeric_limits<float>::infinity()
	                                 : -std::numeric_limits<float>::infinity()) {
	// Negated comparisons so NaN is rejected too
	if (!(factor > 0.0f && factor < 1.0f)) throw std::invalid_argument("ReduceLROnPlateau: factor must be in (0, 1)");
	if (!(threshold >= 0.0f)) throw std::invalid_argument("ReduceLROnPlateau: threshold must be non-negative");
	if (!(min_lr >= 0.0f)) throw std::invalid_argument("ReduceLROnPlateau: min_lr must be non-negative");
}

void ReduceLROnPlateau::step(float metric) {
	// Comparisons with NaN are false, so a NaN metric is never an improvement
	bool improved = mode_ == PlateauMode::Min ? metric < best_ * (1.0f - threshold_)
	                                          : metric > best_ * (1.0f + threshold_);
	if (improved) {
		best_ = metric;
		bad_epochs_ = 0;
	} else {
		++bad_epochs_;
	}
	if (cooldown_left_ > 0) {
		--cooldown_left_;
		bad_epochs_ = 0;
	}
	if (bad_epochs_ > patience_) {
		float old_lr = opt_.lr();
		float new_lr = std::max(old_lr * factor_, min_lr_);
		if (old_lr - new_lr > 1e-8f) opt_.set_lr(new_lr);
		cooldown_left_ = cooldown_;
		bad_epochs_ = 0;
	}
}

}
