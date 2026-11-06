/**
 * transformer.hpp
 * Author: Miguel Mochizuki Silva
 * Description: Pre-norm transformer block and a token-id transformer stack
 */

#pragma once

#include "nn/nn/activation.hpp"
#include "nn/nn/attention.hpp"
#include "nn/nn/dropout.hpp"
#include "nn/nn/embedding.hpp"
#include "nn/nn/layernorm.hpp"
#include "nn/nn/linear.hpp"
#include "nn/nn/module.hpp"
#include "nn/nn/positional.hpp"

#include <memory>

namespace nn {

/*
 * Hyperparameters of a Transformer.
 *
 * Attributes:
 * size_t vocab_size: number of token ids and of output logits
 * size_t dim: model width
 * size_t heads: attention heads, must divide dim
 * size_t layers: number of TransformerBlocks
 * size_t max_len: longest supported sequence
 * size_t mlp_ratio: hidden width of each block MLP is mlp_ratio * dim
 * float dropout: dropout probability in attention and the block MLP
 * bool causal: true for decoder-only, false for encoder-only
 * bool learned_positions: learned instead of sinusoidal positions
 */
struct TransformerConfig {
	size_t vocab_size;
	size_t dim;
	size_t heads;
	size_t layers;
	size_t max_len;
	size_t mlp_ratio = 4;
	float dropout = 0.0f;
	bool causal = true;
	bool learned_positions = false;
};

/*
 * Pre-norm transformer block: x + attn(ln1(x)), then + mlp(ln2(.)), where
 * mlp = Linear(D, r*D), GELU, Linear(r*D, D), Dropout.
 *
 * Attributes: owned LayerNorms, attention, MLP Linears, GELU and Dropout
 */
class TransformerBlock : public Module {
public:
	/* Public, construct a block
	 *
	 * Parameters:
	 * size_t dim: model width
	 * size_t heads: attention heads, must divide dim
	 * size_t mlp_ratio: MLP hidden width multiplier
	 * float dropout: dropout probability in attention and the MLP
	 * bool causal: mask attention to earlier positions
	 *
	 * Returns TransformerBlock: none (constructor); throws std::invalid_argument on a zero size
	 * or dim not divisible by heads
	 */
	TransformerBlock(size_t dim, size_t heads, size_t mlp_ratio, float dropout, bool causal);

	/* Public, apply the block
	 *
	 * Parameters:
	 * Value x: activations {N, T, dim}
	 *
	 * Returns Value: activations, same shape
	 */
	Value forward(const Value& x) override;

	/* Public, list parameters in the order of named_parameters()
	 *
	 * Parameters: none
	 *
	 * Returns vector<Value>: ln1, attn, ln2, fc1, fc2 parameters
	 */
	std::vector<Value> parameters() const override;

	/* Public, list parameters named ln1., attn., ln2., fc1., fc2. plus the child's own name
	 *
	 * Parameters: none
	 *
	 * Returns vector<pair<string, Value>>: (name, handle) pairs
	 */
	std::vector<std::pair<std::string, Value>> named_parameters() const override;

	/* Public, switch training mode on this block and all children
	 *
	 * Parameters:
	 * bool mode: true for training, false for evaluation
	 *
	 * Returns void: none
	 */
	void train(bool mode = true) override;

private:
	LayerNorm ln1_;
	MultiHeadAttention attn_;
	LayerNorm ln2_;
	Linear fc1_;
	GELU act_;
	Linear fc2_;
	Dropout drop_;
};

/*
 * Token-id transformer: Embedding, PositionalEncoding, blocks, final LayerNorm, Linear head.
 *
 * Attributes: owned embedding, positions, blocks, final norm and head
 */
class Transformer : public Module {
public:
	/* Public, construct a transformer
	 *
	 * Parameters:
	 * TransformerConfig config: hyperparameters
	 *
	 * Returns Transformer: none (constructor); throws std::invalid_argument on a zero size or
	 * dim not divisible by heads
	 */
	explicit Transformer(const TransformerConfig& config);

	/* Public, map token ids to logits
	 *
	 * Parameters:
	 * Value ids: float token ids {N, T}, T <= max_len
	 *
	 * Returns Value: logits {N, T, vocab_size}, no softmax
	 */
	Value forward(const Value& ids) override;

	/* Public, list parameters in the order of named_parameters()
	 *
	 * Parameters: none
	 *
	 * Returns vector<Value>: all parameters
	 */
	std::vector<Value> parameters() const override;

	/* Public, list parameters named embed., pos., blocks.<i>., ln_f., head.
	 *
	 * Parameters: none
	 *
	 * Returns vector<pair<string, Value>>: (name, handle) pairs
	 */
	std::vector<std::pair<std::string, Value>> named_parameters() const override;

	/* Public, switch training mode on every child
	 *
	 * Parameters:
	 * bool mode: true for training, false for evaluation
	 *
	 * Returns void: none
	 */
	void train(bool mode = true) override;

private:
	Embedding embed_;
	PositionalEncoding pos_;
	std::vector<std::unique_ptr<TransformerBlock>> blocks_;
	LayerNorm ln_f_;
	Linear head_;
};

}  // namespace nn
