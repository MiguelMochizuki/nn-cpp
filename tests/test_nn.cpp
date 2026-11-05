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
#include "nn/nn/activation.hpp"
#include "nn/nn/sequential.hpp"
#include "nn/autograd/ops.hpp"

#include <memory>
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
	Sequential model;
	model.add(std::make_unique<Linear>(2, 3));
	model.add(std::make_unique<ReLU>());
	model.add(std::make_unique<Linear>(3, 1));

	Value input = Value::leaf(Tensor({2, 2}, std::vector<float>{0.5f, -0.3f, 1.2f, 0.1f}), false);
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
