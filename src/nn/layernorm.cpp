/**
 * layernorm.cpp
 * Author: Miguel Mochizuki Silva
 * Description: Layer normalisation over the last axis
 */

#include "nn/nn/layernorm.hpp"
#include "nn/autograd/ops.hpp"

namespace nn {

LayerNorm::LayerNorm(size_t dim, float eps)
	: dim_(dim), eps_(eps), gamma_(Value::leaf(Tensor({dim}, 1.0f), true)), beta_(Value::leaf(Tensor::zeros({dim}), true)) {}

Value LayerNorm::forward(const Value& input) {
	const std::vector<size_t>& shape = input.data().shape();
	if (shape.empty() || shape.back() != dim_) throw TensorShapeError();
	int axis = static_cast<int>(shape.size()) - 1;
	std::vector<size_t> stat_shape(shape.size(), 1);

	Value centered = sub(input, mean(input, axis, true));
	Value var = mean(mul(centered, centered), axis, true);
	Value normed = mul(centered, rsqrt(add(var, Value::leaf(Tensor(stat_shape, eps_)))));
	return add(mul(normed, gamma_), beta_);
}

std::vector<Value> LayerNorm::parameters() const { return {gamma_, beta_}; }

std::vector<std::pair<std::string, Value>> LayerNorm::named_parameters() const {
	return {{"gamma", gamma_}, {"beta", beta_}};
}

}  // namespace nn
