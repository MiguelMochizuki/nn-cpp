/**
 * workloads.cpp
 * Author: Miguel Mochizuki Silva
 * Description: Builds the seeded synthetic workloads used by the benchmark harness
 */

#include "workloads.hpp"

#include "nn/nn/activation.hpp"
#include "nn/nn/conv2d.hpp"
#include "nn/nn/flatten.hpp"
#include "nn/nn/linear.hpp"
#include "nn/nn/pool2d.hpp"
#include "nn/optim/adam.hpp"
#include "nn/optim/sgd.hpp"

#include <cmath>
#include <random>
#include <stdexcept>

namespace bench {

using namespace nn;

void reseed_parameters(Module& model, unsigned seed) {
	std::mt19937 gen(seed);
	for (Value p : model.parameters()) {
		Tensor& t = p.data();
		std::vector<float>& d = t.data();
		if (t.ndim() < 2) {
			std::fill(d.begin(), d.end(), 0.0f);
			continue;
		}
		float bound = 1.0f / std::sqrt(static_cast<float>(t.size() / t.shape()[0]));
		std::uniform_real_distribution<float> dist(-bound, bound);
		for (float& v : d) v = dist(gen);
	}
}

std::vector<std::string> workload_names() {
	return {"xor", "mlp_wide", "cnn_mnist", "cnn_large", "mlp_adam"};
}

/**
 * Builds the 784-1024-1024-10 MLP shared by mlp_wide and mlp_adam
 *
 * Parameters: none
 *
 * Returns shared_ptr<Sequential>: model with library-initialized (unseeded) weights
 */
static std::shared_ptr<Sequential> build_mlp() {
	auto m = std::make_shared<Sequential>();
	m->add(std::make_unique<Linear>(784, 1024));
	m->add(std::make_unique<ReLU>());
	m->add(std::make_unique<Linear>(1024, 1024));
	m->add(std::make_unique<ReLU>());
	m->add(std::make_unique<Linear>(1024, 10));
	return m;
}

/**
 * Builds a CNN: conv(3x3, pad 1) -> ReLU -> pool 2x2, repeated per channel entry, then a classifier head
 *
 * Parameters:
 * vector<size_t> channels: channel counts, input first
 * size_t side: input height and width
 * size_t hidden: hidden Linear width, 0 for none
 *
 * Returns shared_ptr<Sequential>: model with library-initialized (unseeded) weights
 */
static std::shared_ptr<Sequential> build_cnn(const std::vector<size_t>& channels, size_t side, size_t hidden) {
	auto m = std::make_shared<Sequential>();
	for (size_t i = 0; i + 1 < channels.size(); i++) {
		m->add(std::make_unique<Conv2d>(channels[i], channels[i + 1], 3, 1, 1));
		m->add(std::make_unique<ReLU>());
		m->add(std::make_unique<MaxPool2d>(2, 2));
		side /= 2;
	}
	m->add(std::make_unique<Flatten>());
	size_t flat = channels.back() * side * side;
	if (hidden > 0) {
		m->add(std::make_unique<Linear>(flat, hidden));
		m->add(std::make_unique<ReLU>());
		flat = hidden;
	}
	m->add(std::make_unique<Linear>(flat, 10));
	return m;
}

Workload make_workload(const std::string& name, unsigned seed) {
	Workload w{name, nullptr, nullptr, Tensor({1}), Tensor({1}), false};
	std::vector<size_t> xshape;
	size_t out = 10;
	if (name == "xor") {
		w.model = std::make_shared<Sequential>();
		w.model->add(std::make_unique<Linear>(2, 8));
		w.model->add(std::make_unique<ReLU>());
		w.model->add(std::make_unique<Linear>(8, 1));
		w.optimizer = std::make_shared<SGD>(w.model->parameters(), 0.1f);
		w.classification = false;
		xshape = {16, 2};
		out = 1;
	} else if (name == "mlp_wide" || name == "mlp_adam") {
		w.model = build_mlp();
		if (name == "mlp_wide") w.optimizer = std::make_shared<SGD>(w.model->parameters(), 0.05f);
		else w.optimizer = std::make_shared<Adam>(w.model->parameters(), 1e-3f);
		w.classification = true;
		xshape = {256, 784};
	} else if (name == "cnn_mnist") {
		w.model = build_cnn({1, 8, 16}, 28, 128);
		w.optimizer = std::make_shared<SGD>(w.model->parameters(), 0.05f);
		w.classification = true;
		xshape = {32, 1, 28, 28};
	} else if (name == "cnn_large") {
		w.model = build_cnn({3, 32, 64}, 64, 0);
		w.optimizer = std::make_shared<SGD>(w.model->parameters(), 0.05f);
		w.classification = true;
		xshape = {64, 3, 64, 64};
	} else {
		throw std::invalid_argument("unknown workload: " + name);
	}
	reseed_parameters(*w.model, seed);

	// Inputs and a fixed random linear teacher, each from its own stream of the seed
	std::mt19937 gen(seed + 1);
	std::uniform_real_distribution<float> unit(0.0f, 1.0f), sym(-1.0f, 1.0f);
	w.x = Tensor(xshape);
	for (float& v : w.x.data()) v = unit(gen);
	size_t n = xshape[0], feat = w.x.size() / n;
	std::vector<float> teacher(feat * out);
	for (float& v : teacher) v = sym(gen);

	w.y = Tensor(w.classification ? std::vector<size_t>{n} : std::vector<size_t>{n, out});
	const std::vector<float>& xd = w.x.data();
	for (size_t i = 0; i < n; i++) {
		std::vector<float> o(out, 0.0f);
		for (size_t f = 0; f < feat; f++)
			for (size_t k = 0; k < out; k++) o[k] += xd[i * feat + f] * teacher[f * out + k];
		if (w.classification) {
			size_t best = 0;
			for (size_t k = 1; k < out; k++) if (o[k] > o[best]) best = k;
			w.y.data()[i] = static_cast<float>(best);
		} else {
			for (size_t k = 0; k < out; k++) w.y.data()[i * out + k] = o[k];
		}
	}
	return w;
}

}  // namespace bench
