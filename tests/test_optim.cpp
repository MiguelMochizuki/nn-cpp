/**
 * test_optim.cpp
 * Author: Miguel Mochizuki Silva
 * Description: Unit tests for the Optimizer/SGD update rule
 */

#include "doctest/doctest.h"
#include "nn/optim/adam.hpp"
#include "nn/optim/sgd.hpp"

#include <stdexcept>

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

TEST_CASE("SGD state round-trips lr, momentum and velocity") {
	Value p = Value::leaf(Tensor({1}, 0.0f), true);
	Value q = Value::leaf(Tensor({1}, 0.0f), true);
	SGD a({p}, 0.1f, 0.9f);
	SGD b({q}, 0.5f, 0.0f);
	for (int i = 0; i < 2; i++) {
		p.grad().at({0}) = 1.0f;
		a.step();
	}
	q.data() = p.data();
	b.load_state(a.state());
	p.grad().at({0}) = 1.0f;
	q.grad().at({0}) = 1.0f;
	a.step();
	b.step();
	CHECK(q.data().get({0}) == p.data().get({0}));
}

TEST_CASE("SGD load_state rejects wrong shapes and leaves state unchanged") {
	Value p = Value::leaf(Tensor({1}, 0.0f), true);
	SGD optimizer({p}, 0.1f);
	OptimizerState bad = optimizer.state();
	bad[0].second = Tensor({1}, 99.0f);  // lr, valid
	bad.back().second = Tensor({2}, 0.0f);  // velocity.0, wrong shape
	CHECK_THROWS_AS(optimizer.load_state(bad), std::runtime_error);
	p.grad().at({0}) = 1.0f;
	optimizer.step();
	CHECK(p.data().get({0}) == doctest::Approx(-0.1f));
}

TEST_CASE("SGD load_state rejects missing and unexpected names") {
	Value p = Value::leaf(Tensor({1}, 0.0f), true);
	SGD optimizer({p}, 0.1f);

	OptimizerState missing = optimizer.state();
	missing.pop_back();
	CHECK_THROWS_WITH_AS(optimizer.load_state(missing), doctest::Contains("velocity.0"), std::runtime_error);

	OptimizerState extra = optimizer.state();
	extra.emplace_back("bogus", Tensor({1}, 0.0f));
	CHECK_THROWS_WITH_AS(optimizer.load_state(extra), doctest::Contains("bogus"), std::runtime_error);
}

TEST_CASE("SGD load_state rejects duplicate names") {
	Value p = Value::leaf(Tensor({1}, 0.0f), true);
	SGD optimizer({p}, 0.1f);
	OptimizerState dup = optimizer.state();
	dup.push_back(dup[0]);
	CHECK_THROWS_AS(optimizer.load_state(dup), std::runtime_error);
}

TEST_CASE("Adam with a constant gradient moves each step by about lr") {
	Value p = Value::leaf(Tensor({1}, 0.0f), true);
	Adam optimizer({p}, 0.1f);
	for (int i = 0; i < 3; i++) {
		p.grad().at({0}) = 1.0f;
		optimizer.step();
	}
	CHECK(p.data().get({0}) == doctest::Approx(-0.3f).epsilon(1e-4));
}

TEST_CASE("Adam minimizes a quadratic") {
	Value p = Value::leaf(Tensor({1}, 0.0f), true);
	Adam optimizer({p}, 0.1f);
	for (int i = 0; i < 300; i++) {
		p.grad().at({0}) = 2.0f * (p.data().get({0}) - 3.0f);
		optimizer.step();
	}
	CHECK(p.data().get({0}) == doctest::Approx(3.0f).epsilon(0.02));
}

TEST_CASE("Adam state round-trips moments and step count") {
	Value p = Value::leaf(Tensor({2}, 0.0f), true);
	Value q = Value::leaf(Tensor({2}, 0.0f), true);
	Adam a({p}, 0.1f), b({q}, 0.5f);
	for (int i = 0; i < 3; i++) {
		p.grad().at({0}) = 1.0f;
		p.grad().at({1}) = -0.5f;
		a.step();
	}
	q.data() = p.data();
	b.load_state(a.state());
	p.grad().at({0}) = 0.25f;
	p.grad().at({1}) = 2.0f;
	q.grad() = p.grad();
	a.step();
	b.step();
	CHECK(q.data().data() == p.data().data());
}

TEST_CASE("Adam load_state rejects SGD state and wrong shapes") {
	Value p = Value::leaf(Tensor({2}, 0.0f), true);
	Adam adam({p}, 0.1f);
	SGD sgd({p}, 0.1f, 0.9f);
	CHECK_THROWS_AS(adam.load_state(sgd.state()), std::runtime_error);
	OptimizerState bad = adam.state();
	bad.back().second = Tensor({3}, 0.0f);
	CHECK_THROWS_AS(adam.load_state(bad), std::runtime_error);
}
