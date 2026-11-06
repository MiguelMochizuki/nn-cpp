/**
 * attention.cpp
 * Author: Miguel Mochizuki Silva
 * Description: Multi-head scaled dot-product attention
 */

#include "nn/nn/attention.hpp"
#include "nn/autograd/ops.hpp"

#include <cmath>
#include <stdexcept>

namespace nn {

MultiHeadAttention::MultiHeadAttention(size_t dim, size_t heads, bool causal, float dropout)
	: dim_(dim), heads_(heads), causal_(causal), q_(dim, dim), k_(dim, dim), v_(dim, dim), o_(dim, dim), drop_(dropout) {
	if (heads == 0 || dim % heads != 0) throw std::invalid_argument("MultiHeadAttention: dim must be divisible by heads");
}

Value MultiHeadAttention::forward(const Value& x) { return attend(x, x); }

Value MultiHeadAttention::attend(const Value& query, const Value& context, const Tensor& key_padding_mask) {
	const std::vector<size_t>& qs = query.data().shape();
	const std::vector<size_t>& cs = context.data().shape();
	if (qs.size() != 3 || cs.size() != 3 || qs[2] != dim_ || cs[2] != dim_ || qs[0] != cs[0]) throw TensorShapeError();
	size_t N = qs[0], Tq = qs[1], Tk = cs[1], Dh = dim_ / heads_;
	if (causal_ && Tq != Tk) throw TensorShapeError();
	bool padded = key_padding_mask.size() != 0;
	if (padded && key_padding_mask.shape() != std::vector<size_t>{N, Tk}) throw TensorShapeError();

	// {N, T, D} -> {N, H, T, Dh}; matmul batches over the leading N, H axes
	auto split_heads = [&](const Value& x, size_t T) { return transpose(reshape(x, {N, T, heads_, Dh}), {0, 2, 1, 3}); };
	Value q = split_heads(q_.forward(query), Tq);
	Value k = split_heads(k_.forward(context), Tk);
	Value v = split_heads(v_.forward(context), Tk);

	Value scores = scale(matmul_bt(q, k), 1.0f / std::sqrt(static_cast<float>(Dh)));
	if (causal_ || padded) {
		Tensor mask = Tensor::zeros({N, 1, Tq, Tk});
		for (size_t n = 0; n < N; n++)
			for (size_t i = 0; i < Tq; i++)
				for (size_t j = 0; j < Tk; j++)
					if ((causal_ && j > i) || (padded && key_padding_mask.data()[n * Tk + j] == 0.0f))
						mask.data()[(n * Tq + i) * Tk + j] = 1.0f;
		scores = masked_fill(scores, mask, -1e9f);
	}
	Value weights = drop_.forward(softmax(scores, 3));
	Value merged = transpose(matmul(weights, v), {0, 2, 1, 3});
	return o_.forward(reshape(merged, {N, Tq, dim_}));
}

std::vector<Value> MultiHeadAttention::parameters() const {
	std::vector<Value> out;
	for (const auto& p : named_parameters()) out.push_back(p.second);
	return out;
}

std::vector<std::pair<std::string, Value>> MultiHeadAttention::named_parameters() const {
	std::vector<std::pair<std::string, Value>> out;
	const std::pair<const char*, const Linear*> subs[] = {{"q.", &q_}, {"k.", &k_}, {"v.", &v_}, {"o.", &o_}};
	for (const auto& s : subs)
		for (const auto& p : s.second->named_parameters()) out.emplace_back(s.first + p.first, p.second);
	return out;
}

void MultiHeadAttention::train(bool mode) {
	drop_.train(mode);
	training_ = mode;
}

}  // namespace nn
