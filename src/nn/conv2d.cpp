/**
 * conv2d.cpp
 * Author: Miguel Mochizuki Silva
 * Description: 2D convolutional layer
 */

#include "nn/nn/conv2d.hpp"
#include "nn/autograd/ops.hpp"

#include <cmath>

namespace nn {

namespace {
float conv2d_init_limit(size_t in_channels, size_t kernel_size) {
	float fan_in = static_cast<float>(in_channels * kernel_size * kernel_size);
	return 1.0f / std::sqrt(fan_in);
}
}  // namespace

Conv2d::Conv2d(size_t in_channels, size_t out_channels, size_t kernel_size, size_t stride, size_t padding)
	: weight_(Value::leaf(Tensor::random({out_channels, in_channels, kernel_size, kernel_size},
	                                      -conv2d_init_limit(in_channels, kernel_size),
	                                      conv2d_init_limit(in_channels, kernel_size)),
	                       true)),
	  bias_(Value::leaf(Tensor::zeros({out_channels}), true)),
	  stride_(stride),
	  padding_(padding) {}

Value Conv2d::forward(const Value& input) {
	return conv2d(input, weight_, bias_, static_cast<int>(stride_), static_cast<int>(padding_));
}

std::vector<Value> Conv2d::parameters() const { return {weight_, bias_}; }

}  // namespace nn
