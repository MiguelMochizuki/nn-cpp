/**
 * conv2d.hpp
 * Author: Miguel Mochizuki Silva
 * Description: 2D convolutional layer
 */

#pragma once

#include "nn/nn/module.hpp"

#include <utility>

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

	/* Public, construct a layer with a rectangular kernel, dilation and groups
	 *
	 * Parameters:
	 * size_t in_channels: input channel count
	 * size_t out_channels: output channel count
	 * pair<size_t,size_t> kernel: kernel (height, width)
	 * size_t stride: convolution stride, default 1
	 * size_t padding: zero-padding applied before convolving, default 0
	 * size_t dilation: spacing between kernel taps, default 1; zero throws invalid_argument
	 * size_t groups: channel groups, default 1; must divide in_channels and out_channels,
	 * else invalid_argument. The weight is {out_channels, in_channels/groups, kH, kW}, initialized
	 * uniformly in +-1/sqrt(in_channels/groups * kH * kW)
	 *
	 * Returns Conv2d: none (constructor)
	 */
	Conv2d(size_t in_channels, size_t out_channels, std::pair<size_t, size_t> kernel, size_t stride = 1,
		size_t padding = 0, size_t dilation = 1, size_t groups = 1);

	/* Public, convolve the input with weight_ and add bias_
	 *
	 * Parameters:
	 * Value input: input tensor wrapped in the autograd graph
	 *
	 * Returns Value: shape {N, out_channels, H_out, W_out}; input must be {N, in_channels, H, W}
	 */
	Value forward(const Value& input) override;

	/* Public, convolution and ReLU in one op
	 *
	 * Parameters: none
	 *
	 * Returns bool: true when dilation and groups are both 1 (the plain im2col path)
	 */
	bool can_fuse_relu() const override { return dilation_ == 1 && groups_ == 1; }

	/* Public, relu(forward(input)) as one op
	 *
	 * Parameters:
	 * Value input: input {N, C_in, H, W}
	 *
	 * Returns Value: the activated output, same shape as forward(input)
	 */
	Value forward_relu(const Value& input) override;
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
	size_t dilation_ = 1;
	size_t groups_ = 1;
};

}  // namespace nn
