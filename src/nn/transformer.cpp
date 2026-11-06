/**
 * transformer.cpp
 * Author: Miguel Mochizuki Silva
 * Description: Pre-norm transformer block and a token-id transformer stack
 */

#include "nn/nn/transformer.hpp"
#include "nn/autograd/ops.hpp"

#include <stdexcept>

namespace nn {

namespace {

// Helper, throws std::invalid_argument on zero sizes or dim % heads != 0; returns dim
size_t checked_dim(size_t dim, size_t heads, size_t mlp_ratio) {
	if (dim == 0 || heads == 0 || mlp_ratio == 0) throw std::invalid_argument("Transformer: sizes must be non-zero");
	if (dim % heads != 0) throw std::invalid_argument("Transformer: dim must be divisible by heads");
	return dim;
}

// Helper, validates a whole config; returns its vocab_size
size_t checked_vocab(const TransformerConfig& c) {
	checked_dim(c.dim, c.heads, c.mlp_ratio);
	if (c.vocab_size == 0 || c.layers == 0 || c.max_len == 0) throw std::invalid_argument("Transformer: sizes must be non-zero");
	return c.vocab_size;
}

// Helper, appends child parameters under a name prefix
void append(std::vector<std::pair<std::string, Value>>& out, const std::string& prefix, const Module& m) {
	for (const auto& p : m.named_parameters()) out.emplace_back(prefix + p.first, p.second);
}

}  // namespace

TransformerBlock::TransformerBlock(size_t dim, size_t heads, size_t mlp_ratio, float dropout, bool causal)
	: ln1_(checked_dim(dim, heads, mlp_ratio)), attn_(dim, heads, causal, dropout), ln2_(dim),
	  fc1_(dim, mlp_ratio * dim), fc2_(mlp_ratio * dim, dim), drop_(dropout) {}

Value TransformerBlock::forward(const Value& x) {
	Value h = add(x, attn_.forward(ln1_.forward(x)));
	return add(h, drop_.forward(fc2_.forward(act_.forward(fc1_.forward(ln2_.forward(h))))));
}

std::vector<Value> TransformerBlock::parameters() const {
	std::vector<Value> out;
	for (const auto& p : named_parameters()) out.push_back(p.second);
	return out;
}

std::vector<std::pair<std::string, Value>> TransformerBlock::named_parameters() const {
	std::vector<std::pair<std::string, Value>> out;
	append(out, "ln1.", ln1_);
	append(out, "attn.", attn_);
	append(out, "ln2.", ln2_);
	append(out, "fc1.", fc1_);
	append(out, "fc2.", fc2_);
	return out;
}

void TransformerBlock::train(bool mode) {
	attn_.train(mode);
	drop_.train(mode);
	training_ = mode;
}

Transformer::Transformer(const TransformerConfig& c)
	: embed_(checked_vocab(c), c.dim), pos_(c.max_len, c.dim, c.learned_positions), ln_f_(c.dim), head_(c.dim, c.vocab_size) {
	for (size_t i = 0; i < c.layers; i++)
		blocks_.push_back(std::make_unique<TransformerBlock>(c.dim, c.heads, c.mlp_ratio, c.dropout, c.causal));
}

Value Transformer::forward(const Value& ids) {
	Value h = pos_.forward(embed_.forward(ids));
	for (auto& b : blocks_) h = b->forward(h);
	return head_.forward(ln_f_.forward(h));
}

std::vector<Value> Transformer::parameters() const {
	std::vector<Value> out;
	for (const auto& p : named_parameters()) out.push_back(p.second);
	return out;
}

std::vector<std::pair<std::string, Value>> Transformer::named_parameters() const {
	std::vector<std::pair<std::string, Value>> out;
	append(out, "embed.", embed_);
	append(out, "pos.", pos_);
	for (size_t i = 0; i < blocks_.size(); i++) append(out, "blocks." + std::to_string(i) + ".", *blocks_[i]);
	append(out, "ln_f.", ln_f_);
	append(out, "head.", head_);
	return out;
}

void Transformer::train(bool mode) {
	for (auto& b : blocks_) b->train(mode);
	training_ = mode;
}

}  // namespace nn
