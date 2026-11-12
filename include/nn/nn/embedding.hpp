/**
 * embedding.hpp
 * Author: Miguel Mochizuki Silva
 * Description: Token embedding lookup table
 */

#pragma once

#include "nn/nn/module.hpp"

namespace nn {

/*
 * Lookup table mapping integer ids to learned vectors.
 *
 * Attributes:
 * Value weight_: shape {num_embeddings, dim}, requires_grad() true
 */
class Embedding : public Module {
public:
	/* Public, construct a table with weights uniform in [-1/sqrt(dim), 1/sqrt(dim)]
	 *
	 * Parameters:
	 * size_t num_embeddings: number of distinct ids
	 * size_t dim: vector length per id
	 *
	 * Returns Embedding: none (constructor)
	 */
	Embedding(size_t num_embeddings, size_t dim);

	/* Public, look up the rows for a tensor of ids
	 *
	 * Parameters:
	 * Value ids: float-valued integer ids in [0, num_embeddings), any shape; ids are not differentiable, so any gradient for them is discarded
	 *
	 * Returns Value: shape ids.shape() + {dim}; throws TensorShapeError for an invalid id
	 */
	Value forward(const Value& ids) override;
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
};

}  // namespace nn
