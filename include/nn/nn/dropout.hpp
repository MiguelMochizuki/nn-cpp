/**
 * dropout.hpp
 * Author: Miguel Mochizuki Silva
 * Description: Inverted dropout layer
 */

#pragma once

#include "nn/nn/module.hpp"

#include <random>

namespace nn {

/*
 * Inverted dropout: in training mode each element is zeroed with probability p and the rest
 * are scaled by 1/(1-p), so the expected value is unchanged. In evaluation mode, or with
 * p == 0, the input passes through untouched and no random numbers are drawn.
 *
 * Attributes:
 * float p_: drop probability, in [0, 1]
 * mt19937 gen_: this layer's own stream, derive_generator("dropout") taken at construction
 */
class Dropout : public Module {
public:
	/* Public, construct the layer and take its random stream
	 *
	 * Parameters:
	 * float p: drop probability; 0 <= p <= 1, otherwise std::invalid_argument is thrown.
	 * p == 1 outputs zeros
	 *
	 * Returns Dropout: none (constructor)
	 */
	explicit Dropout(float p = 0.5f);

	/* Public, whether forward() currently changes its input
	 *
	 * Parameters: none
	 *
	 * Returns bool: true in training mode with p > 0
	 */
	bool active() const { return training_ && p_ > 0.0f; }

	/* Public, apply dropout
	 *
	 * Parameters:
	 * Value input: input of any shape
	 *
	 * Returns Value: input itself in eval mode or when p == 0; otherwise input times a fresh
	 * 0 / (1/(1-p)) mask (all zeros when p == 1)
	 */
	Value forward(const Value& input) override;

private:
	float p_;
	std::mt19937 gen_;
};

}  // namespace nn
