/**
 * positional.hpp
 * Author: Miguel Mochizuki Silva
 * Description: Sinusoidal or learned positional encoding for {N, T, D} sequences
 */

#pragma once

#include "nn/nn/module.hpp"

namespace nn {

/*
 * Adds a position-dependent vector to every sequence in a batch. The sinusoidal table is a
 * fixed buffer (no parameters); the learned table is a trainable parameter.
 *
 * Attributes:
 * size_t max_len_: longest supported sequence
 * size_t dim_: model width D
 * bool learned_: true when the table is trainable
 * Value pe_: table of shape {max_len, dim}
 */
class PositionalEncoding : public Module {
public:
	/* Public, construct the encoding table
	 *
	 * Parameters:
	 * size_t max_len: longest supported sequence
	 * size_t dim: model width D
	 * bool learned: false for sinusoidal (pe[t][2i] = sin(t / 10000^(2i/D)), pe[t][2i+1] = cos of the
	 * same angle), true for a trainable table initialised uniform in [-0.02, 0.02]
	 *
	 * Returns PositionalEncoding: none (constructor)
	 */
	PositionalEncoding(size_t max_len, size_t dim, bool learned = false);

	/* Public, add the encoding to the input
	 *
	 * Parameters:
	 * Value input: shape {N, T, D}; T > max_len or a last dimension != D throws TensorShapeError
	 *
	 * Returns Value: same shape as input, input + pe[0:T] broadcast over N
	 */
	Value forward(const Value& input) override;
	/* Public, list the trainable parameters
	 *
	 * Parameters: none
	 *
	 * Returns vector<Value>: {pe} when learned, otherwise empty
	 */
	std::vector<Value> parameters() const override;
	/* Public, list the trainable parameters with stable names
	 *
	 * Parameters: none
	 *
	 * Returns vector<pair<string, Value>>: {"pe", pe} when learned, otherwise empty
	 */
	std::vector<std::pair<std::string, Value>> named_parameters() const override;

private:
	size_t max_len_;
	size_t dim_;
	bool learned_;
	Value pe_;
};

}  // namespace nn
