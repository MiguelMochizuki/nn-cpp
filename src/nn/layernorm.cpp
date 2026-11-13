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
	return layernorm(input, gamma_, beta_, eps_);
}

std::vector<Value> LayerNorm::parameters() const { return {gamma_, beta_}; }

std::vector<std::pair<std::string, Value>> LayerNorm::named_parameters() const {
	return {{"gamma", gamma_}, {"beta", beta_}};
}

}  // namespace nn
