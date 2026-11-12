/**
 * text.hpp
 * Author: Miguel Mochizuki Silva
 * Description: Byte-level character vocabulary and sliding-window sequence dataset for language modelling
 */

#pragma once

#include "nn/data/dataset.hpp"

#include <string>
#include <utility>
#include <vector>

namespace nn {

/*
 * Maps the distinct bytes of a text to ids 0..V-1 in sorted byte order. Bytes are treated as
 * unsigned char; there is no UTF-8 awareness, so a multi-byte character is several symbols.
 *
 * Attributes:
 * std::string chars_: the sorted distinct bytes, index = id
 * int ids_[256]: byte -> id, -1 when absent
 */
class CharVocab {
public:
	/* Public, build the vocabulary from a text
	 *
	 * Parameters:
	 * const std::string& text: corpus whose distinct bytes become the vocabulary
	 *
	 * Returns CharVocab: none (constructor)
	 */
	explicit CharVocab(const std::string& text);

	/* Public, number of distinct symbols
	 *
	 * Parameters: none
	 *
	 * Returns size_t: vocabulary size V
	 */
	size_t size() const;

	/* Public, convert a string to token ids
	 *
	 * Parameters:
	 * const std::string& s: text to encode
	 *
	 * Returns std::vector<float>: one id per byte; throws std::invalid_argument on a byte not in the vocabulary
	 */
	std::vector<float> encode(const std::string& s) const;

	/* Public, convert token ids back to a string
	 *
	 * Parameters:
	 * const std::vector<float>& ids: token ids, each an integer in [0, V)
	 *
	 * Returns std::string: decoded text; throws std::invalid_argument on an invalid id
	 */
	std::string decode(const std::vector<float>& ids) const;

private:
	std::string chars_;
	int ids_[256];
};

/*
 * Sliding windows over a token stream: sample i is input tokens[i, i+T) and target tokens[i+1, i+T+1),
 * both shaped {T}.
 *
 * Attributes:
 * std::vector<float> tokens_: the token stream
 * size_t block_size_: window length T
 */
class SequenceDataset : public Dataset {
public:
	/* Public, build a dataset over a token stream
	 *
	 * Parameters:
	 * std::vector<float> tokens: the token stream
	 * size_t block_size: window length T, >= 1 and < tokens.size()
	 *
	 * Returns SequenceDataset: none (constructor); throws std::invalid_argument if block_size is 0 or tokens.size() <= block_size
	 */
	SequenceDataset(std::vector<float> tokens, size_t block_size);

	/* Public, number of windows
	 *
	 * Parameters: none
	 *
	 * Returns size_t: tokens.size() - block_size
	 */
	size_t size() const override;

	/* Public, fetch one window
	 *
	 * Parameters:
	 * size_t index: window start, 0-based
	 *
	 * Returns pair<Tensor, Tensor>: input {T}, target {T} shifted by one; throws TensorIndexError if index >= size()
	 */
	std::pair<Tensor, Tensor> get(size_t index) const override;

private:
	std::vector<float> tokens_;
	size_t block_size_;
};

}  // namespace nn
