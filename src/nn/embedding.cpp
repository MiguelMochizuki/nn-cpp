/**
 * embedding.cpp
 * Author: Miguel Mochizuki Silva
 * Description: Token embedding lookup table
 */

#include "nn/nn/embedding.hpp"
#include "nn/autograd/ops.hpp"

#include <cmath>

namespace nn {

Embedding::Embedding(size_t num_embeddings, size_t dim)
	: weight_(Value::leaf(Tensor::random({num_embeddings, dim}, -1.0f / std::sqrt(static_cast<float>(dim)),
	                                      1.0f / std::sqrt(static_cast<float>(dim))),
	                       true)) {}

Value Embedding::forward(const Value& ids) { return embedding(weight_, ids.data()); }

std::vector<Value> Embedding::parameters() const { return {weight_}; }

std::vector<std::pair<std::string, Value>> Embedding::named_parameters() const { return {{"weight", weight_}}; }

}  // namespace nn
