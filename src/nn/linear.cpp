/**
 * linear.cpp
 * Author: Miguel Mochizuki Silva
 * Description: Fully-connected layer
 */

#include "nn/nn/linear.hpp"
#include "nn/autograd/ops.hpp"

#include <cmath>

namespace nn {

namespace {
float linear_init_limit(size_t in_features) { return 1.0f / std::sqrt(static_cast<float>(in_features)); }
}  // namespace

Linear::Linear(size_t in_features, size_t out_features)
	: weight_(Value::leaf(Tensor::random({out_features, in_features}, -linear_init_limit(in_features),
	                                      linear_init_limit(in_features)),
	                       true)),
	  bias_(Value::leaf(Tensor::zeros({out_features}), true)) {}

Value Linear::forward(const Value& input) {
	Value weight_t = transpose(weight_, {1, 0});
	Value out = matmul(input, weight_t);
	return add(out, bias_);
}

std::vector<Value> Linear::parameters() const { return {weight_, bias_}; }

}  // namespace nn
