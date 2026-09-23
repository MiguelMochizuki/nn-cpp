/**
 * test_autograd.cpp
 * Author: Miguel Mochizuki Silva
 * Description: Unit tests for the autograd Value/NodeImpl core and ops
 */

#include "doctest/doctest.h"
#include "nn/autograd/value.hpp"
#include "nn/autograd/ops.hpp"
#include "gradcheck.hpp"

#include <cmath>

using namespace nn;

TEST_CASE("leaf construction exposes data and requires_grad") {
	Value v = Value::leaf(Tensor({2}, std::vector<float>{1.0f, 2.0f}), true);
	CHECK(v.data().get({0}) == 1.0f);
	CHECK(v.requires_grad());
}

TEST_CASE("leaf grad starts at zero") {
	Value v = Value::leaf(Tensor({2}, 5.0f), true);
	CHECK(v.grad().get({0}) == 0.0f);
	CHECK(v.grad().get({1}) == 0.0f);
}

TEST_CASE("zero_grad resets grad to zero") {
	Value v = Value::leaf(Tensor({1}, 1.0f), true);
	v.grad().at({0}) = 3.0f;
	v.zero_grad();
	CHECK(v.grad().get({0}) == 0.0f);
}

TEST_CASE("backward on a non-scalar Value throws") {
	Value v = Value::leaf(Tensor({2}, 1.0f), true);
	CHECK_THROWS_AS(v.backward(), AutogradShapeError);
}

TEST_CASE("backward on a leaf scalar seeds its own grad to one") {
	Value v = Value::leaf(Tensor({1}, 5.0f), true);
	v.backward();
	CHECK(v.grad().get({0}) == 1.0f);
}

TEST_CASE("node backward runs its backward_fn with the seeded grad") {
	Value parent = Value::leaf(Tensor({1}, 2.0f), true);
	Value child = Value::node(Tensor({1}, 4.0f), {parent}, [parent](const Tensor& grad_out) mutable {
		parent.grad() = add(parent.grad(), scale(grad_out, 3.0f));
	});
	child.backward();
	CHECK(parent.grad().get({0}) == doctest::Approx(3.0f));
}

TEST_CASE("live_node_count returns to zero once all Values go out of scope") {
	size_t before = Value::live_node_count();
	{
		Value a = Value::leaf(Tensor({1}, 1.0f), true);
		Value b = Value::node(Tensor({1}, 2.0f), {a}, [](const Tensor&) {});
		CHECK(Value::live_node_count() == before + 2);
	}
	CHECK(Value::live_node_count() == before);
}

TEST_CASE("sum reduces to a scalar and backward gives grad 1 everywhere") {
	Value x = Value::leaf(Tensor({3}, std::vector<float>{1, 2, 3}), true);
	Value s = sum(x);
	CHECK(s.data().get({0}) == 6.0f);
	s.backward();
	CHECK(x.grad().get({0}) == 1.0f);
	CHECK(x.grad().get({2}) == 1.0f);
}

TEST_CASE("add gradcheck") {
	Value b = Value::leaf(Tensor({3}, std::vector<float>{4, 5, 6}), false);
	auto f = [b](Value a) mutable { return sum(add(a, b)); };
	Value a = Value::leaf(Tensor({3}, std::vector<float>{1, 2, 3}), true);
	CHECK(gradcheck(f, a));
}

TEST_CASE("mul gradcheck") {
	Value b = Value::leaf(Tensor({3}, std::vector<float>{4, 5, 6}), false);
	auto f = [b](Value a) mutable { return sum(mul(a, b)); };
	Value a = Value::leaf(Tensor({3}, std::vector<float>{1, 2, 3}), true);
	CHECK(gradcheck(f, a));
}

TEST_CASE("sub gradcheck") {
	Value b = Value::leaf(Tensor({3}, std::vector<float>{4, 5, 6}), false);
	auto f = [b](Value a) mutable { return sum(sub(a, b)); };
	Value a = Value::leaf(Tensor({3}, std::vector<float>{1, 2, 3}), true);
	CHECK(gradcheck(f, a));
}

TEST_CASE("scale gradcheck") {
	auto f = [](Value a) { return sum(scale(a, 2.5f)); };
	Value a = Value::leaf(Tensor({3}, std::vector<float>{1, 2, 3}), true);
	CHECK(gradcheck(f, a));
}

TEST_CASE("broadcasted add sums incoming grad over broadcast axes into the smaller parent") {
	Value bias = Value::leaf(Tensor({3}, 0.0f), true);
	Value x = Value::leaf(Tensor({2, 3}, std::vector<float>{1, 1, 1, 1, 1, 1}), false);
	Value y = sum(add(x, bias));
	y.backward();
	CHECK(bias.grad().get({0}) == doctest::Approx(2.0f));
	CHECK(bias.grad().get({1}) == doctest::Approx(2.0f));
}

TEST_CASE("diamond: one leaf feeding two ops accumulates both gradients") {
	Value w = Value::leaf(Tensor({1}, 3.0f), true);
	Value branch_a = scale(w, 2.0f);
	Value branch_b = scale(w, 5.0f);
	Value total = sum(add(branch_a, branch_b));
	total.backward();
	CHECK(w.grad().get({0}) == doctest::Approx(7.0f));
}

TEST_CASE("calling backward twice without zero_grad accumulates") {
	Value w = Value::leaf(Tensor({1}, 3.0f), true);
	Value out1 = scale(w, 2.0f);
	out1.backward();
	Value out2 = scale(w, 2.0f);
	out2.backward();
	CHECK(w.grad().get({0}) == doctest::Approx(4.0f));
}

TEST_CASE("graph nodes are freed once all Values referencing them go out of scope") {
	size_t before = Value::live_node_count();
	{
		Value w = Value::leaf(Tensor({1}, 1.0f), true);
		Value branch_a = scale(w, 2.0f);
		Value branch_b = scale(w, 3.0f);
		Value total = sum(add(branch_a, branch_b));
		total.backward();
	}
	CHECK(Value::live_node_count() == before);
}

TEST_CASE("matmul gradcheck") {
	Value b = Value::leaf(Tensor({3, 2}, std::vector<float>{1, 2, 3, 4, 5, 6}), false);
	auto f = [b](Value a) mutable { return sum(matmul(a, b)); };
	Value a = Value::leaf(Tensor({2, 3}, std::vector<float>{0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f}), true);
	CHECK(gradcheck(f, a));
}

TEST_CASE("matmul forward matches Tensor matmul") {
	Value a = Value::leaf(Tensor({2, 2}, std::vector<float>{1, 2, 3, 4}), false);
	Value b = Value::leaf(Tensor({2, 2}, std::vector<float>{5, 6, 7, 8}), false);
	Value c = matmul(a, b);
	CHECK(c.data().get({0, 0}) == doctest::Approx(19.0f));
	CHECK(c.data().get({1, 1}) == doctest::Approx(50.0f));
}

TEST_CASE("relu zeroes negative inputs and passes positive ones through") {
	Value x = Value::leaf(Tensor({3}, std::vector<float>{-1, 0, 2}), true);
	Value y = relu(x);
	CHECK(y.data().get({0}) == 0.0f);
	CHECK(y.data().get({1}) == 0.0f);
	CHECK(y.data().get({2}) == 2.0f);
}

TEST_CASE("relu gradcheck away from the kink") {
	auto f = [](Value x) { return sum(relu(x)); };
	Value x = Value::leaf(Tensor({3}, std::vector<float>{-2.0f, 1.0f, 3.0f}), true);
	CHECK(gradcheck(f, x));
}

TEST_CASE("sigmoid gradcheck") {
	auto f = [](Value x) { return sum(sigmoid(x)); };
	Value x = Value::leaf(Tensor({3}, std::vector<float>{-1.0f, 0.0f, 1.0f}), true);
	CHECK(gradcheck(f, x));
}

TEST_CASE("tanh gradcheck") {
	auto f = [](Value x) { return sum(tanh(x)); };
	Value x = Value::leaf(Tensor({3}, std::vector<float>{-1.0f, 0.0f, 1.0f}), true);
	CHECK(gradcheck(f, x));
}

TEST_CASE("softmax rows sum to one") {
	Value x = Value::leaf(Tensor({2, 3}, std::vector<float>{1, 2, 3, 1, 1, 1}), true);
	Value y = softmax(x, 1);
	float row0 = y.data().get({0, 0}) + y.data().get({0, 1}) + y.data().get({0, 2});
	CHECK(row0 == doctest::Approx(1.0f));
}

TEST_CASE("softmax gradcheck") {
	auto f = [](Value x) { return sum(softmax(x, 1)); };
	Value x = Value::leaf(Tensor({2, 3}, std::vector<float>{1, 2, 3, 0.5f, -1.0f, 2.0f}), true);
	CHECK(gradcheck(f, x));
}

TEST_CASE("conv2d forward shape and value") {
	Value x = Value::leaf(Tensor({1, 1, 3, 3}, std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9}), false);
	Value w = Value::leaf(Tensor({1, 1, 2, 2}, std::vector<float>{1, 0, 0, 1}), false);
	Value b = Value::leaf(Tensor({1}, 0.0f), false);
	Value y = conv2d(x, w, b, 1, 0);
	CHECK(y.data().shape() == std::vector<size_t>{1, 1, 2, 2});
	CHECK(y.data().get({0, 0, 0, 0}) == doctest::Approx(1.0f + 5.0f));
	CHECK(y.data().get({0, 0, 1, 1}) == doctest::Approx(5.0f + 9.0f));
}

TEST_CASE("conv2d gradcheck wrt input, with padding") {
	Value w = Value::leaf(Tensor({1, 1, 2, 2}, std::vector<float>{0.5f, -0.5f, 1.0f, 0.2f}), false);
	Value b = Value::leaf(Tensor({1}, 0.1f), false);
	auto f = [w, b](Value x) mutable { return sum(conv2d(x, w, b, 1, 1)); };
	Value x = Value::leaf(Tensor({1, 1, 3, 3}, std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9}), true);
	CHECK(gradcheck(f, x, 1e-3f, 0.1f));
}

TEST_CASE("conv2d gradcheck wrt weight") {
	Value x = Value::leaf(Tensor({1, 1, 3, 3}, std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9}), false);
	Value b = Value::leaf(Tensor({1}, 0.1f), false);
	auto f = [x, b](Value w) mutable { return sum(conv2d(x, w, b, 1, 1)); };
	Value w = Value::leaf(Tensor({1, 1, 2, 2}, std::vector<float>{0.5f, -0.5f, 1.0f, 0.2f}), true);
	CHECK(gradcheck(f, w, 1e-3f, 0.1f));
}

TEST_CASE("conv2d gradcheck wrt bias") {
	Value x = Value::leaf(Tensor({1, 1, 3, 3}, std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9}), false);
	Value w = Value::leaf(Tensor({1, 1, 2, 2}, std::vector<float>{0.5f, -0.5f, 1.0f, 0.2f}), false);
	auto f = [x, w](Value b) mutable { return sum(conv2d(x, w, b, 1, 1)); };
	Value b = Value::leaf(Tensor({1}, 0.1f), true);
	CHECK(gradcheck(f, b, 1e-3f, 0.1f));
}

TEST_CASE("maxpool2d picks the max of each window") {
	Value x = Value::leaf(Tensor({1, 1, 4, 4}, std::vector<float>{
	                           1, 2, 5, 6,
	                           3, 4, 7, 8,
	                           9, 10, 13, 14,
	                           11, 12, 15, 16}),
	                       true);
	Value y = maxpool2d(x, 2, 2);
	CHECK(y.data().shape() == std::vector<size_t>{1, 1, 2, 2});
	CHECK(y.data().get({0, 0, 0, 0}) == 4.0f);
	CHECK(y.data().get({0, 0, 0, 1}) == 8.0f);
	CHECK(y.data().get({0, 0, 1, 0}) == 12.0f);
	CHECK(y.data().get({0, 0, 1, 1}) == 16.0f);
}

TEST_CASE("maxpool2d routes gradient only to the max element of each window") {
	Value x = Value::leaf(Tensor({1, 1, 2, 2}, std::vector<float>{1, 5, 3, 2}), true);
	Value y = maxpool2d(x, 2, 2);
	y.backward();
	CHECK(x.grad().get({0, 0, 0, 0}) == 0.0f);
	CHECK(x.grad().get({0, 0, 0, 1}) == 1.0f);
	CHECK(x.grad().get({0, 0, 1, 0}) == 0.0f);
	CHECK(x.grad().get({0, 0, 1, 1}) == 0.0f);
}

TEST_CASE("maxpool2d gradcheck") {
	auto f = [](Value x) { return sum(maxpool2d(x, 2, 2)); };
	Value x = Value::leaf(Tensor({1, 1, 4, 4}, std::vector<float>{
	                           1, 2, 5, 6,
	                           3, 4, 7, 8,
	                           9, 10, 13, 14,
	                           11, 12, 15, 16}),
	                       true);
	CHECK(gradcheck(f, x));
}

TEST_CASE("maxpool2d floors a non-integer output size, ignoring the trailing row and column") {
	Value x = Value::leaf(Tensor({1, 1, 5, 5}, std::vector<float>{
	                           1, 2, 3, 4, 100,
	                           5, 6, 7, 8, 100,
	                           9, 10, 11, 12, 100,
	                           13, 14, 15, 16, 100,
	                           100, 100, 100, 100, 100}),
	                       false);
	Value y = maxpool2d(x, 2, 2);
	CHECK(y.data().shape() == std::vector<size_t>{1, 1, 2, 2});
	CHECK(y.data().get({0, 0, 0, 0}) == 6.0f);
	CHECK(y.data().get({0, 0, 1, 1}) == 16.0f);
}

TEST_CASE("cross_entropy_loss matches hand-computed value for a 2-class batch") {
	Value logits = Value::leaf(Tensor({2, 2}, std::vector<float>{1.0f, 1.0f, 2.0f, 0.0f}), true);
	Tensor targets({2}, std::vector<float>{0.0f, 0.0f});
	Value loss = cross_entropy_loss(logits, targets);
	float expected = (0.693147f + 0.126928f) / 2.0f;
	CHECK(loss.data().get({0}) == doctest::Approx(expected).epsilon(0.01));
}

TEST_CASE("cross_entropy_loss gradcheck") {
	Tensor targets({2}, std::vector<float>{1.0f, 0.0f});
	auto f = [targets](Value logits) mutable { return cross_entropy_loss(logits, targets); };
	Value logits = Value::leaf(Tensor({2, 3}, std::vector<float>{0.2f, 1.5f, -0.3f, 1.0f, -1.0f, 0.5f}), true);
	CHECK(gradcheck(f, logits));
}

TEST_CASE("mse_loss matches hand-computed value") {
	Value pred = Value::leaf(Tensor({2}, std::vector<float>{1.0f, 2.0f}), true);
	Value target = Value::leaf(Tensor({2}, std::vector<float>{0.0f, 0.0f}), false);
	Value loss = mse_loss(pred, target);
	CHECK(loss.data().get({0}) == doctest::Approx((1.0f + 4.0f) / 2.0f));
}

TEST_CASE("mse_loss gradcheck") {
	Value target = Value::leaf(Tensor({3}, std::vector<float>{0.5f, -1.0f, 2.0f}), false);
	auto f = [target](Value pred) mutable { return mse_loss(pred, target); };
	Value pred = Value::leaf(Tensor({3}, std::vector<float>{1.0f, 0.0f, 1.5f}), true);
	CHECK(gradcheck(f, pred));
}
