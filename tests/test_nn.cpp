/**
 * test_nn.cpp
 * Author: Miguel Mochizuki Silva
 * Description: Unit tests for nn::Module layers
 */

#include "doctest/doctest.h"
#include "gradcheck.hpp"
#include "nn/nn/linear.hpp"
#include "nn/nn/conv2d.hpp"
#include "nn/nn/pool2d.hpp"
#include "nn/nn/flatten.hpp"
#include "nn/nn/dropout.hpp"
#include "nn/nn/batchnorm.hpp"
#include "nn/random.hpp"
#include "nn/nn/activation.hpp"
#include "nn/nn/sequential.hpp"
#include "nn/autograd/ops.hpp"

#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace nn;

TEST_CASE("Linear forward produces the expected shape") {
	Linear layer(3, 2);
	Value input = Value::leaf(Tensor({4, 3}, 1.0f), false);
	Value output = layer.forward(input);
	CHECK(output.data().shape() == std::vector<size_t>{4, 2});
}

TEST_CASE("Linear forward matches a hand-set weight and bias") {
	Linear layer(2, 1);
	for (Value& p : layer.parameters()) {
		if (p.data().shape().size() == 2) {
			p.data().at({0, 0}) = 1.0f;
			p.data().at({0, 1}) = 2.0f;
		} else {
			p.data().at({0}) = 0.5f;
		}
	}
	Value input = Value::leaf(Tensor({1, 2}, std::vector<float>{3.0f, 4.0f}), false);
	Value output = layer.forward(input);
	CHECK(output.data().get({0, 0}) == doctest::Approx(1.0f * 3.0f + 2.0f * 4.0f + 0.5f));
}

TEST_CASE("Linear forward matches x * W^T + b") {
	Linear layer(3, 2);
	std::vector<Value> params = layer.parameters();
	params[0].data() = Tensor({2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6});
	params[1].data() = Tensor({2}, std::vector<float>{0.5f, -1.0f});
	Value input = Value::leaf(Tensor({2, 3}, std::vector<float>{1, 0, 1, 0, 1, 2}), false);
	Tensor out = layer.forward(input).data();
	CHECK(out.shape() == std::vector<size_t>{2, 2});
	CHECK(out.get({0, 0}) == doctest::Approx(4.5f));
	CHECK(out.get({0, 1}) == doctest::Approx(9.0f));
	CHECK(out.get({1, 0}) == doctest::Approx(8.5f));
	CHECK(out.get({1, 1}) == doctest::Approx(16.0f));
}

TEST_CASE("Linear named_parameters keeps weight {out,in} and bias {out}") {
	Linear layer(3, 2);
	auto named = layer.named_parameters();
	REQUIRE(named.size() == 2);
	CHECK(named[0].first == "weight");
	CHECK(named[0].second.data().shape() == std::vector<size_t>{2, 3});
	CHECK(named[1].first == "bias");
	CHECK(named[1].second.data().shape() == std::vector<size_t>{2});
}

TEST_CASE("Linear parameters gradcheck through the layer") {
	Linear layer(2, 2);
	Value input = Value::leaf(Tensor({3, 2}, std::vector<float>{1, 2, 3, 4, 5, 6}), false);
	std::vector<Value> params = layer.parameters();
	CHECK(params.size() == 2);
	for (Value& p : params) {
		auto f = [&layer, &input](Value) { return sum(layer.forward(input)); };
		CHECK(gradcheck(f, p));
	}
}

TEST_CASE("Conv2d forward produces the expected shape") {
	Conv2d layer(1, 4, 3, 1, 1);
	Value input = Value::leaf(Tensor({2, 1, 8, 8}, 1.0f), false);
	Value output = layer.forward(input);
	CHECK(output.data().shape() == std::vector<size_t>{2, 4, 8, 8});
}

TEST_CASE("Conv2d parameters gradcheck through the layer") {
	Conv2d layer(1, 2, 3, 1, 1);
	Value input = Value::leaf(Tensor({1, 1, 5, 5}, 1.0f), false);
	for (Value& p : layer.parameters()) {
		auto f = [&layer, &input](Value) { return sum(layer.forward(input)); };
		CHECK(gradcheck(f, p, 1e-3f, 0.1f));
	}
}

TEST_CASE("MaxPool2d forward produces the expected shape") {
	MaxPool2d layer(2, 2);
	Value input = Value::leaf(Tensor({1, 3, 8, 8}, 1.0f), false);
	Value output = layer.forward(input);
	CHECK(output.data().shape() == std::vector<size_t>{1, 3, 4, 4});
}

TEST_CASE("Flatten collapses every dimension after batch into one") {
	Flatten layer;
	Value input = Value::leaf(Tensor({2, 3, 4, 4}, 1.0f), true);
	Value output = layer.forward(input);
	CHECK(output.data().shape() == std::vector<size_t>{2, 48});
}

TEST_CASE("Flatten gradcheck") {
	Flatten layer;
	auto f = [&layer](Value x) { return sum(layer.forward(x)); };
	Value input = Value::leaf(Tensor({1, 2, 2, 2}, std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8}), true);
	CHECK(gradcheck(f, input));
}

TEST_CASE("ReLU, Sigmoid, Tanh modules match their autograd ops") {
	Value input = Value::leaf(Tensor({3}, std::vector<float>{-1.0f, 0.0f, 2.0f}), true);
	ReLU relu_layer;
	CHECK(relu_layer.forward(input).data().get({0}) == 0.0f);
	CHECK(relu_layer.forward(input).data().get({2}) == 2.0f);

	Sigmoid sigmoid_layer;
	CHECK(sigmoid_layer.forward(input).data().get({1}) == doctest::Approx(0.5f));
}

TEST_CASE("Softmax module normalizes along the given axis") {
	Softmax softmax_layer(1);
	Value input = Value::leaf(Tensor({1, 3}, std::vector<float>{1.0f, 1.0f, 1.0f}), true);
	Value output = softmax_layer.forward(input);
	CHECK(output.data().get({0, 0}) == doctest::Approx(1.0f / 3.0f));
}

TEST_CASE("Sequential chains layers and collects all their parameters") {
	Sequential model;
	model.add(std::make_unique<Linear>(3, 4));
	model.add(std::make_unique<ReLU>());
	model.add(std::make_unique<Linear>(4, 2));

	CHECK(model.parameters().size() == 4);

	Value input = Value::leaf(Tensor({5, 3}, 1.0f), false);
	Value output = model.forward(input);
	CHECK(output.data().shape() == std::vector<size_t>{5, 2});
}

TEST_CASE("end-to-end gradcheck through a small Linear-ReLU-Linear network") {
	// Fixed parameters: random init could put a pre-activation within the finite-difference step of the ReLU kink
	auto first = std::make_unique<Linear>(2, 3);
	first->parameters()[0].data() = Tensor({3, 2}, std::vector<float>{1, 1, -1, 2, 2, -1});
	first->parameters()[1].data() = Tensor({3}, std::vector<float>{0.5f, 0.5f, -0.5f});
	auto second = std::make_unique<Linear>(3, 1);
	second->parameters()[0].data() = Tensor({1, 3}, std::vector<float>{0.7f, -0.4f, 0.9f});
	second->parameters()[1].data() = Tensor({1}, std::vector<float>{0.1f});

	Value input = Value::leaf(Tensor({2, 2}, std::vector<float>{0.5f, -0.3f, 1.2f, 0.1f}), false);
	Tensor pre = first->forward(input).data();
	for (float v : pre.data()) CHECK(std::fabs(v) > 0.3f);  // far from the kink (step is 1e-3)

	Sequential model;
	model.add(std::move(first));
	model.add(std::make_unique<ReLU>());
	model.add(std::move(second));
	for (Value& p : model.parameters()) {
		auto f = [&model, &input](Value) { return sum(model.forward(input)); };
		CHECK(gradcheck(f, p, 1e-3f, 0.1f));
	}
}

static std::vector<std::string> names_of(const Module& m) {
	std::vector<std::string> names;
	for (const auto& entry : m.named_parameters()) names.push_back(entry.first);
	return names;
}

TEST_CASE("named_parameters of Sequential are index-prefixed") {
	Sequential m;
	m.add(std::make_unique<Linear>(2, 3));
	m.add(std::make_unique<ReLU>());
	m.add(std::make_unique<Linear>(3, 1));
	CHECK(names_of(m) == std::vector<std::string>{"0.weight", "0.bias", "2.weight", "2.bias"});
}

TEST_CASE("nested Sequential names compose") {
	auto inner = std::make_unique<Sequential>();
	inner->add(std::make_unique<Linear>(2, 2));
	Sequential outer;
	outer.add(std::move(inner));
	CHECK(names_of(outer) == std::vector<std::string>{"0.0.weight", "0.0.bias"});
}

TEST_CASE("named_parameters and parameters list the same handles in the same order") {
	Sequential m;
	m.add(std::make_unique<Conv2d>(1, 2, 3));
	m.add(std::make_unique<Flatten>());
	m.add(std::make_unique<Linear>(2, 2));
	auto named = m.named_parameters();
	auto plain = m.parameters();
	REQUIRE(named.size() == plain.size());
	for (size_t i = 0; i < plain.size(); i++) CHECK(&named[i].second.data() == &plain[i].data());
}

namespace {

// Overrides only forward, as a v1.0.0 subclass would
class ForwardOnly : public Module {
public:
	Value forward(const Value& input) override { return input; }
	// The test reads the flag through a Sequential, so expose a probe for a bare child
	bool flag() const { return training_; }
};

}  // namespace

TEST_CASE("Module defaults to training mode") {
	ForwardOnly m;
	CHECK(m.is_training());
}

TEST_CASE("Sequential eval and train reach every child") {
	auto seq = std::make_unique<Sequential>();
	ForwardOnly* kids[3];
	for (auto& k : kids) {
		auto m = std::make_unique<ForwardOnly>();
		k = m.get();
		seq->add(std::move(m));
	}
	seq->eval();
	CHECK_FALSE(seq->is_training());
	for (auto* k : kids) {
		CHECK_FALSE(k->flag());
	}
	seq->train();
	CHECK(seq->is_training());
	for (auto* k : kids) {
		CHECK(k->flag());
	}
}

namespace {

std::vector<float> dropout_mask(Dropout& d, size_t n) {
	return d.forward(Value::leaf(Tensor({n}, 1.0f))).data().data();
}

}  // namespace

TEST_CASE("Dropout eval and p = 0 return the input Value itself") {
	manual_seed(1);
	Value x = Value::leaf(Tensor({4}, 2.0f), true);
	Dropout d(0.5f);
	d.eval();
	Value same = d.forward(x);
	CHECK(&same.data() == &x.data());
	Dropout none(0.0f);
	Value same0 = none.forward(x);
	CHECK(&same0.data() == &x.data());
	CHECK_THROWS_AS(Dropout(-0.1f), std::invalid_argument);
	CHECK_THROWS_AS(Dropout(1.5f), std::invalid_argument);
}

TEST_CASE("Dropout with p = 1 gives finite zeros and zero gradient") {
	manual_seed(1);
	Dropout d(1.0f);
	Value x = Value::leaf(Tensor({6}, 3.0f), true);
	Value y = d.forward(x);
	for (float v : y.data().data()) CHECK(v == 0.0f);
	sum(y).backward();
	for (float g : x.grad().data()) CHECK(g == 0.0f);
}

TEST_CASE("Dropout in train mode keeps the mean and drops about p") {
	manual_seed(1);
	Dropout d(0.3f);
	std::vector<float> out = dropout_mask(d, 100000);
	double total = 0;
	size_t zeros = 0;
	for (float v : out) {
		total += v;
		if (v == 0.0f) zeros++;
	}
	CHECK(total / out.size() == doctest::Approx(1.0).epsilon(0.02));
	CHECK(static_cast<double>(zeros) / out.size() == doctest::Approx(0.3).epsilon(0.02));
}

TEST_CASE("Dropout masks are reproducible and unaffected by eval toggling") {
	manual_seed(5);
	Dropout a(0.5f);
	auto a1 = dropout_mask(a, 64);
	auto a2 = dropout_mask(a, 64);
	manual_seed(5);
	Dropout b(0.5f);
	CHECK(dropout_mask(b, 64) == a1);
	b.eval();
	dropout_mask(b, 64);
	b.train();
	CHECK(dropout_mask(b, 64) == a2);
	CHECK(a1 != a2);
}

TEST_CASE("Dropout gradcheck with a fixed seed") {
	Value x = Value::leaf(Tensor({7, 3}, std::vector<float>(21, 0.5f)), true);
	auto f = [](Value a) {
		manual_seed(3);
		Dropout d(0.4f);
		std::vector<float> w(21);
		for (size_t i = 0; i < w.size(); i++) w[i] = 1.0f + 0.1f * static_cast<float>(i);
		return sum(mul(d.forward(a), Value::leaf(Tensor({7, 3}, w))));
	};
	CHECK(gradcheck(f, x));
}

namespace {
// Per-channel mean and biased variance of y, channel axis 1
std::pair<std::vector<float>, std::vector<float>> channel_stats(const Tensor& y) {
	size_t N = y.shape()[0], C = y.shape()[1], inner = y.size() / (N * C);
	std::vector<float> m(C, 0.0f), v(C, 0.0f);
	for (size_t c = 0; c < C; c++) {
		for (size_t n = 0; n < N; n++)
			for (size_t i = 0; i < inner; i++) m[c] += y.data()[(n * C + c) * inner + i];
		m[c] /= static_cast<float>(N * inner);
		for (size_t n = 0; n < N; n++)
			for (size_t i = 0; i < inner; i++) {
				float d = y.data()[(n * C + c) * inner + i] - m[c];
				v[c] += d * d;
			}
		v[c] /= static_cast<float>(N * inner);
	}
	return {m, v};
}
}  // namespace

TEST_CASE("BatchNorm train output is normalised per channel") {
	BatchNorm1d b1(4);
	auto s1 = channel_stats(b1.forward(Value::leaf(Tensor::random({16, 4}, -3.0f, 5.0f))).data());
	BatchNorm2d b2(3);
	auto s2 = channel_stats(b2.forward(Value::leaf(Tensor::random({8, 3, 5, 5}, -3.0f, 5.0f))).data());
	for (size_t c = 0; c < 4; c++) {
		CHECK(s1.first[c] == doctest::Approx(0.0f).epsilon(1e-3).scale(1.0));
		CHECK(s1.second[c] == doctest::Approx(1.0f).epsilon(1e-3));
	}
	for (size_t c = 0; c < 3; c++) {
		CHECK(s2.first[c] == doctest::Approx(0.0f).epsilon(1e-3).scale(1.0));
		CHECK(s2.second[c] == doctest::Approx(1.0f).epsilon(1e-3));
	}
}

TEST_CASE("BatchNorm running statistics follow the unbiased formula and eval uses them") {
	BatchNorm1d bn(2, 1e-5f, 0.25f);
	// channel 0: 1,2,3,6 -> mean 3, biased var 3.5, unbiased 14/3; channel 1: constant 5
	Tensor x({4, 2}, std::vector<float>{1, 5, 2, 5, 3, 5, 6, 5});
	bn.forward(Value::leaf(x));
	auto bufs = bn.named_buffers();
	CHECK(bufs[0].first == "running_mean");
	CHECK(bufs[1].first == "running_var");
	CHECK(bufs[0].second->data()[0] == doctest::Approx(0.75f * 0.0f + 0.25f * 3.0f));
	CHECK(bufs[0].second->data()[1] == doctest::Approx(0.25f * 5.0f));
	CHECK(bufs[1].second->data()[0] == doctest::Approx(0.75f + 0.25f * 14.0f / 3.0f));
	CHECK(bufs[1].second->data()[1] == doctest::Approx(0.75f));
	bn.eval();
	Value y = bn.forward(Value::leaf(Tensor({1, 2}, std::vector<float>{1.0f, 2.0f})));
	float rm0 = 0.75f, rv0 = 0.75f + 0.25f * 14.0f / 3.0f, rm1 = 1.25f, rv1 = 0.75f;
	CHECK(y.data().data()[0] == doctest::Approx((1.0f - rm0) / std::sqrt(rv0 + 1e-5f)));
	CHECK(y.data().data()[1] == doctest::Approx((2.0f - rm1) / std::sqrt(rv1 + 1e-5f)));
	// eval does not touch the statistics
	CHECK(bufs[0].second->data()[0] == doctest::Approx(0.75f));
}

TEST_CASE("BatchNorm eval before any step is the identity up to eps") {
	BatchNorm2d bn(2);
	bn.eval();
	Tensor x = Tensor::random({2, 2, 3, 3}, -2.0f, 2.0f);
	Value y = bn.forward(Value::leaf(x));
	for (size_t i = 0; i < x.size(); i++) CHECK(y.data().data()[i] == doctest::Approx(x.data()[i]).epsilon(1e-4));
}

TEST_CASE("BatchNorm with one element per channel stays finite and keeps running_var") {
	BatchNorm2d bn(3);
	Value y = bn.forward(Value::leaf(Tensor({1, 3, 1, 1}, std::vector<float>{1.0f, -2.0f, 4.0f})));
	for (float v : y.data().data()) {
		CHECK(std::isfinite(v));
		CHECK(v == 0.0f);
	}
	CHECK(bn.named_buffers()[1].second->data() == std::vector<float>(3, 1.0f));
	CHECK(bn.named_buffers()[0].second->data()[0] == doctest::Approx(0.1f));
}

TEST_CASE("BatchNorm rejects wrong rank and channel count") {
	BatchNorm1d b1(3);
	BatchNorm2d b2(3);
	CHECK_THROWS_AS(b1.forward(Value::leaf(Tensor({2, 4}, 1.0f))), TensorShapeError);
	CHECK_THROWS_AS(b1.forward(Value::leaf(Tensor({2, 3, 1, 1}, 1.0f))), TensorShapeError);
	CHECK_THROWS_AS(b2.forward(Value::leaf(Tensor({2, 3}, 1.0f))), TensorShapeError);
	CHECK_THROWS_AS(b2.forward(Value::leaf(Tensor({2, 4, 2, 2}, 1.0f))), TensorShapeError);
}

TEST_CASE("BatchNorm gradchecks input, weight and bias in train mode") {
	for (bool two_d : {false, true}) {
		std::vector<size_t> shape = two_d ? std::vector<size_t>{4, 2, 3, 3} : std::vector<size_t>{7, 3};
		size_t C = shape[1];
		Tensor xt = Tensor::random(shape, -1.0f, 1.0f);
		Tensor wt = Tensor::random(shape, 0.5f, 1.5f);  // fixed random weighting so the loss is not shift-invariant
		std::unique_ptr<Module> bn;
		if (two_d) bn = std::make_unique<BatchNorm2d>(C); else bn = std::make_unique<BatchNorm1d>(C);
		auto params = bn->parameters();
		params[0].data() = Tensor::random({C}, 0.5f, 1.5f);
		params[1].data() = Tensor::random({C}, -0.5f, 0.5f);
		auto f_x = [&](Value x) { return sum(mul(bn->forward(x), Value::leaf(wt))); };
		CHECK(gradcheck(f_x, Value::leaf(xt, true)));
		// weight and bias: perturb the layer's own leaves
		Value x = Value::leaf(xt);
		for (size_t pi = 0; pi < 2; pi++) {
			auto f_p = [&](Value) { return sum(mul(bn->forward(x), Value::leaf(wt))); };
			CHECK(gradcheck(f_p, params[pi]));
		}
	}
}

TEST_CASE("BatchNorm 3-step train then eval flow matches a hand-computed reference") {
	BatchNorm1d bn(1, 0.0f, 0.5f);
	// single channel; batches {0,2}, {4,4}, {1,3}: mean 1,4,2; unbiased var 2,0,2
	float batches[3][2] = {{0, 2}, {4, 4}, {1, 3}};
	for (auto& b : batches) bn.forward(Value::leaf(Tensor({2, 1}, std::vector<float>{b[0], b[1]})));
	// running_mean: 0 -> .5 -> 2.25 -> 2.125; running_var: 1 -> 1.5 -> .75 -> 1.375
	CHECK(bn.named_buffers()[0].second->data()[0] == doctest::Approx(2.125f));
	CHECK(bn.named_buffers()[1].second->data()[0] == doctest::Approx(1.375f));
	bn.eval();
	Value y = bn.forward(Value::leaf(Tensor({1, 1}, 5.0f)));
	CHECK(y.data().data()[0] == doctest::Approx((5.0f - 2.125f) / std::sqrt(1.375f)));
}

TEST_CASE("AvgPool2d layer pools and has no parameters") {
	AvgPool2d pool(2, 2);
	CHECK(pool.parameters().empty());
	Value y = pool.forward(Value::leaf(Tensor({1, 1, 2, 2}, std::vector<float>{1, 2, 3, 6})));
	CHECK(y.data().shape() == std::vector<size_t>{1, 1, 1, 1});
	CHECK(y.data().data()[0] == doctest::Approx(3.0f));
}

TEST_CASE("GlobalAvgPool2d equals the manual spatial mean and passes gradcheck") {
	GlobalAvgPool2d gap;
	Tensor xt = Tensor::random({2, 3, 4, 5}, -1.0f, 1.0f);
	Value y = gap.forward(Value::leaf(xt));
	CHECK(y.data().shape() == std::vector<size_t>{2, 3});
	for (size_t n = 0; n < 2; n++)
		for (size_t c = 0; c < 3; c++) {
			float s = 0.0f;
			for (size_t i = 0; i < 20; i++) s += xt.data()[(n * 3 + c) * 20 + i];
			CHECK(y.data().get({n, c}) == doctest::Approx(s / 20.0f));
		}
	Tensor wt = Tensor::random({2, 3}, 0.5f, 1.5f);
	auto f = [&](Value x) { return sum(mul(gap.forward(x), Value::leaf(wt))); };
	CHECK(gradcheck(f, Value::leaf(xt, true)));
	CHECK_THROWS_AS(gap.forward(Value::leaf(Tensor({2, 3}, 1.0f))), TensorShapeError);
}

TEST_CASE("Conv2d with rectangular kernel, dilation and groups") {
	Conv2d depthwise(4, 4, std::make_pair<size_t, size_t>(1, 3), 1, 1, 2, 4);
	CHECK(depthwise.parameters()[0].data().shape() == std::vector<size_t>{4, 1, 1, 3});
	Value input = Value::leaf(Tensor({2, 4, 6, 6}, 1.0f), false);
	// H: 6 + 2 - 0 - 1 + 1 = 8; W: 6 + 2 - 2*2 - 1 + 1 = 4
	CHECK(depthwise.forward(input).data().shape() == std::vector<size_t>{2, 4, 8, 4});
	for (Value& p : depthwise.parameters()) {
		auto f = [&depthwise, &input](Value) { return sum(depthwise.forward(input)); };
		CHECK(gradcheck(f, p, 1e-3f, 0.1f));
	}
	CHECK_THROWS_AS(Conv2d(4, 4, std::make_pair<size_t, size_t>(3, 3), 1, 0, 1, 3), std::invalid_argument);
	CHECK_THROWS_AS(Conv2d(4, 6, std::make_pair<size_t, size_t>(3, 3), 1, 0, 1, 4), std::invalid_argument);
	CHECK_THROWS_AS(Conv2d(4, 4, std::make_pair<size_t, size_t>(3, 3), 1, 0, 0, 1), std::invalid_argument);
	CHECK_THROWS_AS(Conv2d(4, 4, std::make_pair<size_t, size_t>(3, 3), 1, 0, 1, 0), std::invalid_argument);
}
