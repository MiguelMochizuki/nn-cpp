/**
 * conv2d.hpp
 * Author: Miguel Mochizuki Silva
 * Description: 2D convolutional layer
 */

#pragma once

#include "nn/nn/module.hpp"

namespace nn {

/*
 * 2D convolutional layer, implemented via the autograd conv2d op.
 *
 * Attributes:
 * Value weight_: shape {out_channels, in_channels, kernel_size, kernel_size}, requires_grad() true
 * Value bias_: shape {out_channels}, requires_grad() true
 * size_t stride_: convolution stride
 * size_t padding_: zero-padding applied before convolving
 */
class Conv2d : public Module {
public:
	/* Public, construct a layer with randomly-initialized weight and zero bias
	 *
	 * Parameters:
	 * size_t in_channels: input channel count
	 * size_t out_channels: output channel count
	 * size_t kernel_size: convolution kernel side length
	 * size_t stride: convolution stride, default 1
	 * size_t padding: zero-padding applied before convolving, default 0
	 *
	 * Returns Conv2d: none (constructor)
	 */
	Conv2d(size_t in_channels, size_t out_channels, size_t kernel_size, size_t stride = 1, size_t padding = 0);

	Value forward(const Value& input) override;
	std::vector<Value> parameters() const override;

private:
	Value weight_;
	Value bias_;
	size_t stride_;
	size_t padding_;
};

}  // namespace nn
