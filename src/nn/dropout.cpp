/**
 * dropout.cpp
 * Author: Miguel Mochizuki Silva
 * Description: Inverted dropout layer
 */

#include "nn/nn/dropout.hpp"

#include "nn/autograd/ops.hpp"
#include "nn/random.hpp"

#include <stdexcept>

namespace nn {

Dropout::Dropout(float p) : p_(p), gen_(derive_generator("dropout")) {
	if (!(p >= 0.0f && p <= 1.0f)) throw std::invalid_argument("Dropout: p must be in [0, 1]");
}

Value Dropout::forward(const Value& input) {
	if (!training_ || p_ == 0.0f) return input;
	Tensor mask(input.data().shape());
	// p == 1 drops everything, so no random numbers are drawn (mask stays zero-initialised)
	if (p_ == 1.0f) return mul(input, Value::leaf(std::move(mask)));
	float keep_scale = 1.0f / (1.0f - p_);
	for (float& m : mask.data()) {
		// 24 high bits give a uniform in [0, 1) that is identical on every standard library
		float u = static_cast<float>(gen_() >> 8) * (1.0f / 16777216.0f);
		m = u < p_ ? 0.0f : keep_scale;
	}
	return mul(input, Value::leaf(std::move(mask)));
}

}  // namespace nn
