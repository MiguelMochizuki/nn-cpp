/**
 * attention.hpp
 * Author: Miguel Mochizuki Silva
 * Description: Multi-head scaled dot-product attention
 */

#pragma once

#include "nn/nn/dropout.hpp"
#include "nn/nn/linear.hpp"
#include "nn/nn/module.hpp"

namespace nn {

/*
 * Multi-head attention over {N, T, D} sequences, composed from differentiable ops only.
 * Scores are softmax(Q K^T / sqrt(D / heads)) per head. Masked scores are replaced by -1e9
 * before the softmax, so a query row whose keys are all masked yields the uniform average of
 * V (finite, never NaN).
 *
 * Attributes:
 * size_t dim_: model width D
 * size_t heads_: number of heads; dim_ % heads_ == 0
 * bool causal_: mask keys j > i
 * Linear q_, k_, v_, o_: query, key, value and output projections, each Linear(dim, dim)
 * Dropout drop_: attention-weight dropout, active in training mode only
 */
class MultiHeadAttention : public Module {
public:
	/* Public, construct the attention block
	 *
	 * Parameters:
	 * size_t dim: model width; must be divisible by heads, otherwise std::invalid_argument
	 * size_t heads: number of heads
	 * bool causal: forbid attending to later positions (requires Tq == Tk in attend)
	 * float dropout: drop probability on the attention weights, in [0, 1]
	 *
	 * Returns MultiHeadAttention: none (constructor)
	 */
	MultiHeadAttention(size_t dim, size_t heads, bool causal = false, float dropout = 0.0f);

	/* Public, self-attention
	 *
	 * Parameters:
	 * Value x: shape {N, T, D}
	 *
	 * Returns Value: attend(x, x), shape {N, T, D}
	 */
	Value forward(const Value& x) override;

	/* Public, attention from a query sequence over a context sequence
	 *
	 * Parameters:
	 * Value query: shape {N, Tq, D}
	 * Value context: shape {N, Tk, D}
	 * Tensor key_padding_mask: optional, shape {N, Tk}, nonzero = real token, 0 = padding;
	 * the default zero-element Tensor({0}) means no padding. Padded keys get zero weight
	 *
	 * Returns Value: shape {N, Tq, D}. Throws TensorShapeError on bad shapes, or if causal
	 * and Tq != Tk. A row with all keys masked averages V uniformly
	 */
	Value attend(const Value& query, const Value& context, const Tensor& key_padding_mask = Tensor({0}));

	/* Public, list the trainable parameters
	 *
	 * Parameters: none
	 *
	 * Returns vector<Value>: q, k, v, o parameters, same order as named_parameters()
	 */
	std::vector<Value> parameters() const override;
	/* Public, list the trainable parameters with stable names
	 *
	 * Parameters: none
	 *
	 * Returns vector<pair<string, Value>>: q.weight, q.bias, k.weight, ..., o.bias
	 */
	std::vector<std::pair<std::string, Value>> named_parameters() const override;
	/* Public, switch training / evaluation mode, forwarded to the dropout
	 *
	 * Parameters:
	 * bool mode: true for training, false for evaluation
	 *
	 * Returns void: none
	 */
	void train(bool mode = true) override;

private:
	size_t dim_;
	size_t heads_;
	bool causal_;
	Linear q_, k_, v_, o_;
	Dropout drop_;
};

}  // namespace nn
