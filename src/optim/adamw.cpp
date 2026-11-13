/**
 * adamw.cpp
 * Author: Miguel Mochizuki Silva
 * Description: Adam with decoupled weight decay
 */

#include "nn/optim/adamw.hpp"
#include "nn/kernels/parallel.hpp"

namespace nn {

AdamW::AdamW(std::vector<Value> params, float lr, float beta1, float beta2, float eps, float weight_decay)
	: Adam(std::move(params), lr, beta1, beta2, eps), weight_decay_(weight_decay) {}

void AdamW::step() {
	const float shrink = 1.0f - lr_ * weight_decay_;
	for (Value& p : params_) {
		float* w = p.data().data().data();
		kernels::parallel_range(p.data().size(), size_t(1) << 13, [=](size_t lo, size_t hi) {
			for (size_t k = lo; k < hi; k++) w[k] *= shrink;
		});
	}
	Adam::step();
}

}  // namespace nn
