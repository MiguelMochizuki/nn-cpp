/**
 * conv2d.cpp
 * Author: Miguel Mochizuki Silva
 * Description: 2D convolutional layer
 */

#include "nn/nn/conv2d.hpp"
#include "nn/autograd/ops.hpp"

#include <cmath>
#include <stdexcept>

namespace nn {

namespace {
/**
 * Bound of the uniform weight initialization, 1 / sqrt(fan_in)
 *
 * Parameters:
 * size_t in_channels: Input channels
 * size_t kernel_size: Side of the square kernel
 *
 * Returns float: Weights are drawn from [-limit, limit]
 */
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

namespace {
// Validates groups/dilation (as invalid_argument, before any division by them) and builds the weight
Value grouped_weight(size_t in_channels, size_t out_channels, std::pair<size_t, size_t> kernel, size_t dilation,
	size_t groups) {
	if (dilation < 1) throw std::invalid_argument("Conv2d: dilation must be >= 1");
	if (groups < 1 || in_channels % groups != 0 || out_channels % groups != 0) {
		throw std::invalid_argument("Conv2d: groups must divide in_channels and out_channels");
	}
	if (kernel.first == 0 || kernel.second == 0) throw std::invalid_argument("Conv2d: kernel must be non-empty");
	size_t cg = in_channels / groups;
	float limit = 1.0f / std::sqrt(static_cast<float>(cg * kernel.first * kernel.second));
	return Value::leaf(Tensor::random({out_channels, cg, kernel.first, kernel.second}, -limit, limit), true);
}
}  // namespace

Conv2d::Conv2d(size_t in_channels, size_t out_channels, std::pair<size_t, size_t> kernel, size_t stride,
	size_t padding, size_t dilation, size_t groups)
	: weight_(grouped_weight(in_channels, out_channels, kernel, dilation, groups)),
	  bias_(Value::leaf(Tensor::zeros({out_channels}), true)),
	  stride_(stride),
	  padding_(padding),
	  dilation_(dilation),
	  groups_(groups) {}

Value Conv2d::forward(const Value& input) {
	if (dilation_ != 1 || groups_ != 1) {
		return conv2d(input, weight_, bias_, static_cast<int>(stride_), static_cast<int>(padding_),
			static_cast<int>(dilation_), static_cast<int>(groups_));
	}
	return conv2d(input, weight_, bias_, static_cast<int>(stride_), static_cast<int>(padding_));
}

std::vector<Value> Conv2d::parameters() const { return {weight_, bias_}; }

std::vector<std::pair<std::string, Value>> Conv2d::named_parameters() const {
	return {{"weight", weight_}, {"bias", bias_}};
}

}  // namespace nn
