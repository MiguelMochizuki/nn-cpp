/**
 * test_optim.cpp
 * Author: Miguel Mochizuki Silva
 * Description: Unit tests for the Optimizer/SGD update rule
 */

#include "doctest/doctest.h"
#include "nn/optim/sgd.hpp"

using namespace nn;

TEST_CASE("SGD without momentum applies param -= lr * grad") {
	Value p = Value::leaf(Tensor({2}, std::vector<float>{1.0f, 2.0f}), true);
	p.grad().at({0}) = 0.5f;
	p.grad().at({1}) = -1.0f;
	SGD optimizer({p}, 0.1f);
	optimizer.step();
	CHECK(p.data().get({0}) == doctest::Approx(1.0f - 0.1f * 0.5f));
	CHECK(p.data().get({1}) == doctest::Approx(2.0f - 0.1f * -1.0f));
}

TEST_CASE("SGD zero_grad resets every tracked parameter's grad") {
	Value p = Value::leaf(Tensor({1}, 1.0f), true);
	p.grad().at({0}) = 5.0f;
	SGD optimizer({p}, 0.1f);
	optimizer.zero_grad();
	CHECK(p.grad().get({0}) == 0.0f);
}

TEST_CASE("SGD with momentum accumulates velocity across steps") {
	Value p = Value::leaf(Tensor({1}, 0.0f), true);
	SGD optimizer({p}, 1.0f, 0.9f);

	p.grad().at({0}) = 1.0f;
	optimizer.step();
	CHECK(p.data().get({0}) == doctest::Approx(-1.0f));

	p.grad().at({0}) = 1.0f;
	optimizer.step();
	CHECK(p.data().get({0}) == doctest::Approx(-1.0f - 1.9f));
}
