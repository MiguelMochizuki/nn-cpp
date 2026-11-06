/**
 * test_transformer.cpp
 * Author: Miguel Mochizuki Silva
 * Description: Unit tests for the transformer-path modules
 */

#include "doctest/doctest.h"
#include "gradcheck.hpp"
#include "nn/nn/attention.hpp"
#include "nn/nn/embedding.hpp"
#include "nn/nn/layernorm.hpp"
#include "nn/nn/positional.hpp"
#include "nn/autograd/ops.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

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

namespace {
Value rand_leaf(std::vector<size_t> s, bool rg = false) { return Value::leaf(Tensor::random(s, -1.0f, 1.0f), rg); }
void set_identity(MultiHeadAttention& m) {
	size_t D = 4;
	for (size_t i = 0; i < 4; i++) {
		Tensor w = Tensor::zeros({D, D});
		for (size_t d = 0; d < D; d++) w.data()[d * D + d] = 1.0f;
		m.parameters()[2 * i].data() = w;
		m.parameters()[2 * i + 1].data() = Tensor::zeros({D});
	}
}
}  // namespace

TEST_CASE("attention single head matches hand reference") {
	MultiHeadAttention a(2, 1);
	auto ps = a.parameters();
	for (size_t i = 0; i < 4; i++) {
		ps[2 * i].data() = Tensor({2, 2}, std::vector<float>{1, 0, 0, 1});
		ps[2 * i + 1].data() = Tensor::zeros({2});
	}
	std::vector<float> x = {1, 0, 0, 2};
	Value y = a.forward(Value::leaf(Tensor({1, 2, 2}, x), false));
	CHECK(y.data().shape() == std::vector<size_t>{1, 2, 2});
	float s = std::sqrt(2.0f);
	for (int i = 0; i < 2; i++) {
		float s0 = (x[2 * i] * x[0] + x[2 * i + 1] * x[1]) / s, s1 = (x[2 * i] * x[2] + x[2 * i + 1] * x[3]) / s;
		float e0 = std::exp(s0), e1 = std::exp(s1), z = e0 + e1;
		for (int c = 0; c < 2; c++) CHECK(y.data().data()[2 * i + c] == doctest::Approx((e0 * x[c] + e1 * x[2 + c]) / z).epsilon(1e-5));
	}
	auto np = a.named_parameters();
	CHECK(np.size() == 8);
	CHECK(np[0].first == "q.weight");
	CHECK(np[1].first == "q.bias");
	CHECK(np[6].first == "o.weight");
	CHECK(np[7].first == "o.bias");
}

TEST_CASE("attention causal ignores the future, non-causal does not") {
	Tensor x = Tensor::random({2, 4, 4}, -1.0f, 1.0f);
	// perturb only positions > 1 in each batch element
	Tensor x2 = x;
	for (size_t n = 0; n < 2; n++)
		for (size_t e = (n * 4 + 2) * 4; e < (n * 4 + 4) * 4; e++) x2.data()[e] += 0.7f;
	for (bool causal : {true, false}) {
		MultiHeadAttention a(4, 2, causal);
		Value y1 = a.forward(Value::leaf(x, false)), y2 = a.forward(Value::leaf(x2, false));
		bool same = true;
		for (size_t n = 0; n < 2; n++)
			for (size_t e = n * 16; e < n * 16 + 8; e++) same = same && (y1.data().data()[e] == y2.data().data()[e]);
		CHECK(same == causal);
	}
}

TEST_CASE("attention key padding has no influence, and fully masked rows are finite") {
	MultiHeadAttention a(4, 2);
	Tensor x = Tensor::random({2, 4, 4}, -1.0f, 1.0f);
	Tensor mask({2, 4}, std::vector<float>{1, 1, 0, 0, 1, 1, 1, 0});
	Tensor x2 = x;
	for (size_t e = 8; e < 16; e++) x2.data()[e] += 0.7f;  // batch 0, keys 2 and 3
	for (size_t e = 28; e < 32; e++) x2.data()[e] += 0.7f;  // batch 1, key 3
	Value xv = Value::leaf(x, false), xv2 = Value::leaf(x2, false);
	Value y1 = a.attend(xv, xv, mask), y2 = a.attend(xv, xv2, mask);
	bool same = true;
	for (size_t e = 0; e < 32; e++) same = same && (y1.data().data()[e] == y2.data().data()[e]);
	CHECK(same);
	Value u1 = a.attend(xv, xv), u2 = a.attend(xv, xv2);
	bool differs = false;
	for (size_t e = 0; e < 32; e++) differs = differs || (u1.data().data()[e] != u2.data().data()[e]);
	CHECK(differs);

	Tensor none({2, 4}, std::vector<float>{0, 0, 0, 0, 1, 1, 1, 1});
	Value y = a.attend(xv, xv, none);
	for (float f : y.data().data()) CHECK(std::isfinite(f));

	// identity projections: a fully masked row is exactly the mean of V
	MultiHeadAttention id(4, 2);
	set_identity(id);
	Value yi = id.attend(xv, xv, none);
	for (size_t t = 0; t < 4; t++)
		for (size_t c = 0; c < 4; c++) {
			float m = 0;
			for (size_t j = 0; j < 4; j++) m += x.data()[j * 4 + c] / 4;
			CHECK(yi.data().data()[t * 4 + c] == doctest::Approx(m).epsilon(1e-5));
		}
	CHECK_THROWS_AS(a.attend(xv, xv, Tensor({2, 5})), TensorShapeError);
	CHECK_THROWS_AS(a.attend(xv, xv, Tensor({2})), TensorShapeError);
}

TEST_CASE("attention heads=2 matches per-head reference and differs from heads=1") {
	MultiHeadAttention a(4, 2), one(4, 1);
	set_identity(a);
	set_identity(one);
	std::vector<float> x = {1, 0.5f, -1, 2, 0, 1, 1, -0.5f, 2, 1, 0.5f, 0};
	Value in = Value::leaf(Tensor({1, 3, 4}, x), false);
	Value y = a.forward(in), y1 = one.forward(in);
	for (size_t h = 0; h < 2; h++)
		for (size_t i = 0; i < 3; i++) {
			float sc[3], z = 0;
			for (size_t j = 0; j < 3; j++) {
				sc[j] = (x[i * 4 + h * 2] * x[j * 4 + h * 2] + x[i * 4 + h * 2 + 1] * x[j * 4 + h * 2 + 1]) / std::sqrt(2.0f);
				sc[j] = std::exp(sc[j]);
				z += sc[j];
			}
			for (size_t c = 0; c < 2; c++) {
				float ref = 0;
				for (size_t j = 0; j < 3; j++) ref += sc[j] / z * x[j * 4 + h * 2 + c];
				CHECK(y.data().data()[i * 4 + h * 2 + c] == doctest::Approx(ref).epsilon(1e-5));
			}
		}
	bool differs = false;
	for (size_t e = 0; e < 12; e++) differs = differs || std::abs(y.data().data()[e] - y1.data().data()[e]) > 1e-4f;
	CHECK(differs);
}

TEST_CASE("attention causal plus padding never reads the future") {
	MultiHeadAttention a(4, 2, true);
	Tensor x = Tensor::random({2, 4, 4}, -1.0f, 1.0f);
	Tensor mask({2, 4}, std::vector<float>{0, 0, 1, 1, 1, 1, 1, 1});
	// batch 0 rows 0 and 1 see only padded keys; perturb positions >= 2 (future for them)
	Tensor x2 = x;
	for (size_t e = 8; e < 16; e++) x2.data()[e] += 0.7f;
	Value y1 = a.attend(Value::leaf(x, false), Value::leaf(x, false), mask);
	Value y2 = a.attend(Value::leaf(x2, false), Value::leaf(x2, false), mask);
	// rows 0..1 depend on their own query too, which is unperturbed
	bool same = true;
	for (size_t e = 0; e < 8; e++) same = same && (y1.data().data()[e] == y2.data().data()[e]);
	CHECK(same);
	bool differs = false;
	for (size_t e = 8; e < 16; e++) differs = differs || (y1.data().data()[e] != y2.data().data()[e]);
	CHECK(differs);
}

TEST_CASE("attention T=1 and shape errors") {
	MultiHeadAttention a(4, 2, true);
	Value y = a.forward(rand_leaf({2, 1, 4}));
	CHECK(y.data().shape() == std::vector<size_t>{2, 1, 4});
	CHECK_THROWS_AS(MultiHeadAttention(5, 2), std::invalid_argument);
	CHECK_THROWS_AS(a.attend(rand_leaf({1, 2, 4}), rand_leaf({1, 3, 4})), TensorShapeError);
	MultiHeadAttention b(4, 2);
	CHECK(b.attend(rand_leaf({1, 2, 4}), rand_leaf({1, 3, 4})).data().shape() == std::vector<size_t>{1, 2, 4});
}

TEST_CASE("attention gradcheck on input and q.weight") {
	MultiHeadAttention a(4, 2, true);
	Tensor w = Tensor::random({2, 3, 4}, 0.5f, 2.0f);
	Value x = rand_leaf({2, 3, 4}, true);
	CHECK(gradcheck([&](Value v) { return sum(mul(a.forward(v), Value::leaf(w, false))); }, x));
	Tensor pad({2, 3}, std::vector<float>{1, 0, 1, 1, 1, 0});
	CHECK(gradcheck([&](Value v) { return sum(mul(a.attend(v, v, pad), Value::leaf(w, false))); }, x));
	Value qw = a.parameters()[0];
	Value xf = rand_leaf({2, 3, 4});
	CHECK(gradcheck([&](Value) {  // gradcheck perturbs qw in place, so a.forward sees it
		return sum(mul(a.forward(xf), Value::leaf(w, false)));
	}, qw));
}

TEST_CASE("attention train() reaches dropout") {
	MultiHeadAttention a(4, 2, false, 0.5f);
	Value x = rand_leaf({1, 4, 4});
	a.eval();
	Value e1 = a.forward(x), e2 = a.forward(x);
	CHECK(e1.data().data() == e2.data().data());
	a.train();
	bool differs = false;
	for (int i = 0; i < 5 && !differs; i++) differs = a.forward(x).data().data() != e1.data().data();
	CHECK(differs);
}
