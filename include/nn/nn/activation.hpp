/**
 * activation.hpp
 * Author: Miguel Mochizuki Silva
 * Description: Elementwise activation layers
 */

#pragma once

#include "nn/nn/module.hpp"

namespace nn {

/*
 * Rectified linear unit activation. No parameters.
 *
 * Attributes: none
 */
class ReLU : public Module {
public:
	Value forward(const Value& input) override;
};

/*
 * Logistic sigmoid activation. No parameters.
 *
 * Attributes: none
 */
class Sigmoid : public Module {
public:
	Value forward(const Value& input) override;
};

/*
 * Hyperbolic tangent activation. No parameters.
 *
 * Attributes: none
 */
class Tanh : public Module {
public:
	Value forward(const Value& input) override;
};

/*
 * Softmax activation along a fixed axis. No parameters.
 *
 * Attributes:
 * int axis_: dimension to normalize over
 */
class Softmax : public Module {
public:
	/* Public, construct a softmax layer over a fixed axis
	 *
	 * Parameters:
	 * int axis: dimension to normalize over
	 *
	 * Returns Softmax: none (constructor)
	 */
	explicit Softmax(int axis);

	Value forward(const Value& input) override;

private:
	int axis_;
};

}  // namespace nn
