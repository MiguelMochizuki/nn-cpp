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
	/* Public, apply ReLU elementwise: max(x, 0)
	 *
	 * Parameters:
	 * Value input: input tensor wrapped in the autograd graph
	 *
	 * Returns Value: activation output, same shape as input
	 */
	Value forward(const Value& input) override;
};

/*
 * Logistic sigmoid activation. No parameters.
 *
 * Attributes: none
 */
class Sigmoid : public Module {
public:
	/* Public, apply the logistic sigmoid elementwise
	 *
	 * Parameters:
	 * Value input: input tensor wrapped in the autograd graph
	 *
	 * Returns Value: activation output, same shape as input
	 */
	Value forward(const Value& input) override;
};

/*
 * Hyperbolic tangent activation. No parameters.
 *
 * Attributes: none
 */
class Tanh : public Module {
public:
	/* Public, apply tanh elementwise
	 *
	 * Parameters:
	 * Value input: input tensor wrapped in the autograd graph
	 *
	 * Returns Value: activation output, same shape as input
	 */
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

	/* Public, apply softmax along the configured axis
	 *
	 * Parameters:
	 * Value input: input tensor wrapped in the autograd graph
	 *
	 * Returns Value: activation output, same shape as input
	 */
	Value forward(const Value& input) override;

private:
	int axis_;
};

}  // namespace nn
