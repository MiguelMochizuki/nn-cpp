/**
 * workloads.cpp
 * Author: Miguel Mochizuki Silva
 * Description: Builds the seeded synthetic workloads used by the benchmark harness
 */

#include "workloads.hpp"

#include "nn/nn/activation.hpp"
#include "nn/nn/batchnorm.hpp"
#include "nn/nn/conv2d.hpp"
#include "nn/nn/flatten.hpp"
#include "nn/nn/linear.hpp"
#include "nn/nn/pool2d.hpp"
#include "nn/nn/transformer.hpp"
#include "nn/autograd/ops.hpp"
#include "nn/optim/adam.hpp"
#include "nn/optim/adamw.hpp"
#include "nn/optim/sgd.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>

namespace bench {

using namespace nn;

void reseed_parameters(Module& model, unsigned seed, bool keep_ones_rank1) {
	std::mt19937 gen(seed);
	for (Value p : model.parameters()) {
		Tensor& t = p.data();
		std::vector<float>& d = t.data();
		if (t.ndim() < 2) {
			if (keep_ones_rank1 && std::all_of(d.begin(), d.end(), [](float v) { return v == 1.0f; })) continue;
			std::fill(d.begin(), d.end(), 0.0f);
			continue;
		}
		float bound = 1.0f / std::sqrt(static_cast<float>(t.size() / t.shape()[0]));
		std::uniform_real_distribution<float> dist(-bound, bound);
		for (float& v : d) v = dist(gen);
	}
}

std::vector<std::string> workload_names() {
	return {"xor", "mlp_wide", "cnn_mnist", "cnn_large", "mlp_adam", "cnn_fer", "transformer_tiny"};
}

/**
 * Wraps a Transformer and flattens its {N, T, V} logits to {N*T, V} so the Sequential and
 * cross_entropy_loss path of the harness applies unchanged
 *
 * Attributes: inner_ the wrapped transformer
 */
class FlatTransformer : public Module {
public:
	explicit FlatTransformer(const TransformerConfig& c) : inner_(c) {}
	Value forward(const Value& ids) override {
		Value o = inner_.forward(ids);
		const std::vector<size_t>& s = o.data().shape();
		return reshape(o, {s[0] * s[1], s[2]});
	}
	std::vector<Value> parameters() const override { return inner_.parameters(); }
	std::vector<std::pair<std::string, Value>> named_parameters() const override { return inner_.named_parameters(); }
	void train(bool mode = true) override { Module::train(mode); inner_.train(mode); }

private:
	Transformer inner_;
};

/**
 * Builds the transformer_tiny workload: random token ids, random target ids, AdamW
 *
 * Parameters:
 * unsigned seed: seed for parameters, ids and targets
 *
 * Returns Workload: x {16, 64} float ids in [0, 64), y {16*64} float ids
 */
static Workload make_transformer_tiny(unsigned seed) {
	Workload w{"transformer_tiny", std::make_shared<Sequential>(), nullptr, Tensor({16, 64}), Tensor({16 * 64}), true};
	auto net = std::make_unique<FlatTransformer>(TransformerConfig{64, 64, 4, 2, 64});
	reseed_parameters(*net, seed, true);
	w.model->add(std::move(net));
	w.optimizer = std::make_shared<AdamW>(w.model->parameters(), 1e-3f);
	std::mt19937 gen(seed + 1);
	std::uniform_int_distribution<int> id(0, 63);
	for (float& v : w.x.data()) v = static_cast<float>(id(gen));
	for (float& v : w.y.data()) v = static_cast<float>(id(gen));
	return w;
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
	if (name == "transformer_tiny") return make_transformer_tiny(seed);
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
	} else if (name == "cnn_fer") {
		// FER-2013-shaped: Conv-BN-ReLU stacks, global average pooling and a 7-class head
		w.model = std::make_shared<Sequential>();
		size_t in = 1;
		for (size_t c : {16, 32, 64}) {
			w.model->add(std::make_unique<Conv2d>(in, c, 3, 1, 1));
			w.model->add(std::make_unique<BatchNorm2d>(c));
			w.model->add(std::make_unique<ReLU>());
			if (c != 64) w.model->add(std::make_unique<MaxPool2d>(2, 2));
			in = c;
		}
		w.model->add(std::make_unique<GlobalAvgPool2d>());
		w.model->add(std::make_unique<Linear>(64, 7));
		w.optimizer = std::make_shared<AdamW>(w.model->parameters(), 1e-3f);
		w.classification = true;
		xshape = {64, 1, 48, 48};
		out = 7;
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
