/**
 * text.cpp
 * Author: Miguel Mochizuki Silva
 * Description: CharVocab and SequenceDataset implementations
 */

#include "nn/data/text.hpp"

#include <algorithm>
#include <stdexcept>

namespace nn {

CharVocab::CharVocab(const std::string& text) {
	std::fill(ids_, ids_ + 256, -1);
	for (unsigned char c : text) {
		ids_[c] = 0;
	}
	for (int c = 0; c < 256; ++c) {
		if (ids_[c] == 0) {
			ids_[c] = static_cast<int>(chars_.size());
			chars_.push_back(static_cast<char>(c));
		}
	}
}

size_t CharVocab::size() const { return chars_.size(); }

std::vector<float> CharVocab::encode(const std::string& s) const {
	std::vector<float> out;
	out.reserve(s.size());
	for (unsigned char c : s) {
		if (ids_[c] < 0) {
			throw std::invalid_argument("CharVocab::encode: character not in vocabulary");
		}
		out.push_back(static_cast<float>(ids_[c]));
	}
	return out;
}

std::string CharVocab::decode(const std::vector<float>& ids) const {
	std::string out;
	out.reserve(ids.size());
	for (float f : ids) {
		if (!(f >= 0.0f && f < static_cast<float>(chars_.size())) || f != static_cast<float>(static_cast<size_t>(f))) {
			throw std::invalid_argument("CharVocab::decode: invalid id");
		}
		out.push_back(chars_[static_cast<size_t>(f)]);
	}
	return out;
}

SequenceDataset::SequenceDataset(std::vector<float> tokens, size_t block_size)
	: tokens_(std::move(tokens)), block_size_(block_size) {
	if (block_size_ == 0 || tokens_.size() <= block_size_) {
		throw std::invalid_argument("SequenceDataset: need block_size >= 1 and tokens.size() > block_size");
	}
}

size_t SequenceDataset::size() const { return tokens_.size() - block_size_; }

std::pair<Tensor, Tensor> SequenceDataset::get(size_t index) const {
	if (index >= size()) {
		throw TensorIndexError();
	}
	Tensor x({block_size_});
	Tensor y({block_size_});
	std::copy(tokens_.begin() + index, tokens_.begin() + index + block_size_, x.data().begin());
	std::copy(tokens_.begin() + index + 1, tokens_.begin() + index + 1 + block_size_, y.data().begin());
	return {x, y};
}

}  // namespace nn
