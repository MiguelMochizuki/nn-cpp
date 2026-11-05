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

	/* Public, convolve the input with weight_ and add bias_
	 *
	 * Parameters:
	 * Value input: input tensor wrapped in the autograd graph
	 *
	 * Returns Value: shape {N, out_channels, H_out, W_out}; input must be {N, in_channels, H, W}
	 */
	Value forward(const Value& input) override;
	/* Public, list the trainable parameters
	 *
	 * Parameters: none
	 *
	 * Returns vector<Value>: the parameters, in the same order as named_parameters()
	 */
	std::vector<Value> parameters() const override;
	/* Public, list the trainable parameters with stable names
	 *
	 * Parameters: none
	 *
	 * Returns vector<pair<string, Value>>: (name, handle) pairs, in the same order as parameters()
	 */
	std::vector<std::pair<std::string, Value>> named_parameters() const override;

private:
	Value weight_;
	Value bias_;
	size_t stride_;
	size_t padding_;
};

}  // namespace nn
