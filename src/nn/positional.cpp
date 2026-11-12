/**
 * positional.cpp
 * Author: Miguel Mochizuki Silva
 * Description: Sinusoidal or learned positional encoding for {N, T, D} sequences
 */

#include "nn/nn/positional.hpp"
#include "nn/autograd/ops.hpp"

#include <cmath>

namespace nn {

namespace {
Tensor sinusoidal_table(size_t max_len, size_t dim) {
	Tensor t = Tensor::zeros({max_len, dim});
	for (size_t pos = 0; pos < max_len; pos++) {
		for (size_t c = 0; c < dim; c++) {
			double angle = static_cast<double>(pos) / std::pow(10000.0, static_cast<double>(c - c % 2) / static_cast<double>(dim));
			t.data()[pos * dim + c] = static_cast<float>(c % 2 == 0 ? std::sin(angle) : std::cos(angle));
		}
	}
	return t;
}
}  // namespace

PositionalEncoding::PositionalEncoding(size_t max_len, size_t dim, bool learned)
	: max_len_(max_len),
	  dim_(dim),
	  learned_(learned),
	  pe_(learned ? Value::leaf(Tensor::random({max_len, dim}, -0.02f, 0.02f), true)
	              : Value::leaf(sinusoidal_table(max_len, dim), false)) {}

Value PositionalEncoding::forward(const Value& input) {
	const std::vector<size_t>& shape = input.data().shape();
	if (shape.size() != 3 || shape[1] > max_len_ || shape[2] != dim_) throw TensorShapeError();
	return add(input, slice(pe_, 0, 0, shape[1]));
}

std::vector<Value> PositionalEncoding::parameters() const {
	return learned_ ? std::vector<Value>{pe_} : std::vector<Value>{};
}

std::vector<std::pair<std::string, Value>> PositionalEncoding::named_parameters() const {
	return learned_ ? std::vector<std::pair<std::string, Value>>{{"pe", pe_}}
	                : std::vector<std::pair<std::string, Value>>{};
}

}  // namespace nn
