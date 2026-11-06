/**
 * test_transformer.cpp
 * Author: Miguel Mochizuki Silva
 * Description: Unit tests for the transformer-path modules
 */

#include "doctest/doctest.h"
#include "gradcheck.hpp"
#include "nn/nn/embedding.hpp"
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
