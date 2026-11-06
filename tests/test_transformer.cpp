/**
 * test_transformer.cpp
 * Author: Miguel Mochizuki Silva
 * Description: Unit tests for the transformer-path modules
 */

#include "doctest/doctest.h"
#include "gradcheck.hpp"
#include "nn/nn/embedding.hpp"
#include "nn/nn/layernorm.hpp"
#include "nn/nn/positional.hpp"
#include "nn/autograd/ops.hpp"

#include <limits>

using namespace nn;

TEST_CASE("embedding forward picks rows and shapes") {
	Embedding e(4, 3);
	Value out = e.forward(Value::leaf(Tensor({3}, std::vector<float>{2, 0, 2}), false));
	CHECK(out.data().shape() == std::vector<size_t>{3, 3});
	Value w = e.parameters()[0];
	for (size_t d = 0; d < 3; d++) {
		CHECK(out.data().data()[d] == w.data().data()[2 * 3 + d]);
		CHECK(out.data().data()[3 + d] == w.data().data()[d]);
		CHECK(out.data().data()[6 + d] == w.data().data()[2 * 3 + d]);
	}
	Value o2 = e.forward(Value::leaf(Tensor({2, 3}, 1.0f), false));
	CHECK(o2.data().shape() == std::vector<size_t>{2, 3, 3});
	auto np = e.named_parameters();
	CHECK(np.size() == 1);
	CHECK(np[0].first == "weight");
}

TEST_CASE("embedding gradcheck with repeated ids") {
	for (size_t D : {size_t(1), size_t(7)}) {
		Value w = Value::leaf(Tensor::random({3, D}, -1.0f, 1.0f), true);
		Tensor ids({3}, std::vector<float>{1, 1, 0});
		Tensor wt = Tensor::random({3, D}, 0.5f, 2.0f);
		CHECK(gradcheck([&](Value v) { return sum(mul(embedding(v, ids), Value::leaf(wt, false))); }, w));
	}
}

TEST_CASE("embedding rejects invalid ids") {
	Value w = Value::leaf(Tensor::random({3, 2}, -1.0f, 1.0f), false);
	for (float bad : {3.0f, -1.0f, std::numeric_limits<float>::quiet_NaN(), 2.5f})
		CHECK_THROWS_AS(embedding(w, Tensor({2}, std::vector<float>{0, bad})), TensorShapeError);
}

TEST_CASE("layernorm normalises rows and handles constants") {
	LayerNorm ln(4);
	Value y = ln.forward(Value::leaf(Tensor({1, 4}, std::vector<float>{1, 2, 3, 4}), false));
	float m = 0, v = 0;
	for (int i = 0; i < 4; i++) m += y.data().data()[i] / 4;
	for (int i = 0; i < 4; i++) v += (y.data().data()[i] - m) * (y.data().data()[i] - m) / 4;
	CHECK(std::abs(m) < 1e-4f);
	CHECK(std::abs(v - 1.0f) < 1e-4f);
	Value c = ln.forward(Value::leaf(Tensor({1, 4}, 3.0f), false));
	for (int i = 0; i < 4; i++) CHECK(std::abs(c.data().data()[i]) < 1e-4f);

	Tensor row({1, 4}, std::vector<float>{1, 2, 3, 4});
	Value a = ln.forward(Value::leaf(row, false));
	Value b = ln.forward(Value::leaf(Tensor({2, 3, 4}, std::vector<float>{1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4}), false));
	for (size_t i = 0; i < 24; i++) CHECK(b.data().data()[i] == doctest::Approx(a.data().data()[i % 4]));

	auto np = ln.named_parameters();
	REQUIRE(np.size() == 2);
	CHECK(np[0].first == "gamma");
	CHECK(np[1].first == "beta");
}

TEST_CASE("layernorm gradcheck input gamma beta") {
	for (std::vector<size_t> shape : {std::vector<size_t>{2, 7}, std::vector<size_t>{2, 3, 5}}) {
		size_t D = shape.back();
		Tensor x = Tensor::random(shape, -1.0f, 1.0f);
		Tensor w = Tensor::random(shape, 0.5f, 2.0f);
		Value xv = Value::leaf(x, true);
		LayerNorm ln(D);
		Value wv = Value::leaf(w, false);
		auto loss = [&](Value) { return sum(mul(ln.forward(xv), wv)); };
		CHECK(gradcheck(loss, xv));
		for (Value p : ln.parameters()) {
			// perturb gamma/beta through the module's own handle
			Value pp = p;
			CHECK(gradcheck([&](Value) { return sum(mul(ln.forward(xv), wv)); }, pp));
		}
	}
}

TEST_CASE("positional encoding sinusoidal and learned") {
	PositionalEncoding pe(5, 4);
	CHECK(pe.parameters().empty());
	Value y = pe.forward(Value::leaf(Tensor({2, 3, 4}, 0.0f), false));
	CHECK(y.data().shape() == std::vector<size_t>{2, 3, 4});
	const std::vector<float>& d = y.data().data();
	CHECK(d[0] == doctest::Approx(0.0f));
	CHECK(d[1] == doctest::Approx(1.0f));
	CHECK(d[2] == doctest::Approx(0.0f));
	CHECK(d[3] == doctest::Approx(1.0f));
	CHECK(d[4] == doctest::Approx(std::sin(1.0f)));
	CHECK(d[12] == d[0 + 0]);
	CHECK(d[12 + 4] == d[4]);
	PositionalEncoding odd(3, 3);
	Value yo = odd.forward(Value::leaf(Tensor({1, 2, 3}, 0.0f), false));
	CHECK(yo.data().data()[2] == doctest::Approx(0.0f));
	CHECK_THROWS_AS(pe.forward(Value::leaf(Tensor({1, 6, 4}, 0.0f), false)), TensorShapeError);
	CHECK_THROWS_AS(pe.forward(Value::leaf(Tensor({1, 2, 3}, 0.0f), false)), TensorShapeError);

	PositionalEncoding le(5, 3, true);
	REQUIRE(le.parameters().size() == 1);
	CHECK(le.named_parameters()[0].first == "pe");
	Value xv = Value::leaf(Tensor::random({2, 4, 3}, -1.0f, 1.0f), false);
	Value wv = Value::leaf(Tensor::random({2, 4, 3}, 0.5f, 2.0f), false);
	CHECK(gradcheck([&](Value) { return sum(mul(le.forward(xv), wv)); }, le.parameters()[0]));
}
