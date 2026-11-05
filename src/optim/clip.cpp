/**
 * clip.cpp
 * Author: Miguel Mochizuki Silva
 * Description: Gradient clipping by global norm and by value
 */

#include "nn/optim/clip.hpp"

#include <algorithm>
#include <cmath>

namespace nn {

float clip_grad_norm(const std::vector<Value>& params, float max_norm) {
	double sum = 0.0;  // double: many small squares lose precision in float
	for (const Value& p : params) {
		if (!p.has_grad()) continue;
		for (float g : p.grad().data()) sum += static_cast<double>(g) * g;
	}
	const float norm = static_cast<float>(std::sqrt(sum));
	if (norm > max_norm) {
		const float scale = max_norm / (norm + 1e-6f);
		for (const Value& p : params) {
			if (!p.has_grad()) continue;
			for (float& g : p.grad().data()) g *= scale;
		}
	}
	return norm;
}

void clip_grad_value(const std::vector<Value>& params, float clip) {
	for (const Value& p : params) {
		if (!p.has_grad()) continue;
		for (float& g : p.grad().data()) g = std::min(std::max(g, -clip), clip);
	}
}

}  // namespace nn
