/**
 * test_optim.cpp
 * Author: Miguel Mochizuki Silva
 * Description: Unit tests for the Optimizer/SGD update rule
 */

#include "doctest/doctest.h"
#include "nn/optim/adam.hpp"
#include "nn/optim/adamw.hpp"
#include "nn/optim/clip.hpp"
#include "nn/optim/scheduler.hpp"
#include "nn/optim/sgd.hpp"

#include <cmath>
#include <memory>
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

TEST_CASE("SGD Nesterov momentum steps with g + momentum * velocity") {
	Value p = Value::leaf(Tensor({1}, 1.0f), true);
	SGD optimizer({p}, 0.1f, 0.9f, true);
	p.grad().at({0}) = 0.5f;
	optimizer.step();
	CHECK(p.data().get({0}) == doctest::Approx(0.905f));
	p.grad().at({0}) = 0.5f;
	optimizer.step();
	CHECK(p.data().get({0}) == doctest::Approx(0.905f - 0.1f * (0.5f + 0.9f * (0.9f * 0.5f + 0.5f))));
}

TEST_CASE("SGD weight decay shrinks a zero-gradient parameter by lr * wd * param") {
	Value p = Value::leaf(Tensor({1}, 2.0f), true);
	SGD optimizer({p}, 0.5f, 0.0f, false, 0.1f);
	optimizer.step();
	CHECK(p.data().get({0}) == doctest::Approx(2.0f - 0.5f * 0.1f * 2.0f));
}

TEST_CASE("SGD folds weight decay into the gradient before the velocity update") {
	// lr 0.1, momentum 0.9, wd 0.2, raw gradient 0.5, p0 = 1:
	// step 1: g = 0.5 + 0.2 * 1 = 0.7, v = 0.7, p = 1 - 0.07 = 0.93
	// step 2: g = 0.5 + 0.2 * 0.93 = 0.686, v = 0.9 * 0.7 + 0.686 = 1.316, p = 0.93 - 0.1316 = 0.7984
	// (decaying p after the velocity update instead would give a different p after step 1)
	Value p = Value::leaf(Tensor({1}, 1.0f), true);
	SGD optimizer({p}, 0.1f, 0.9f, false, 0.2f);
	p.grad().at({0}) = 0.5f;
	optimizer.step();
	CHECK(p.data().get({0}) == doctest::Approx(0.93f));
	p.grad().at({0}) = 0.5f;
	optimizer.step();
	CHECK(p.data().get({0}) == doctest::Approx(0.7984f));
}

TEST_CASE("SGD Nesterov folds weight decay into the gradient before the velocity update") {
	// lr 0.1, momentum 0.9, wd 0.2, raw gradient 0.5, p0 = 1:
	// step 1: g = 0.7, v = 0.7, update = g + 0.9 * v = 1.33, p = 1 - 0.133 = 0.867
	// step 2: g = 0.5 + 0.2 * 0.867 = 0.6734, v = 0.63 + 0.6734 = 1.3034,
	//         update = 0.6734 + 0.9 * 1.3034 = 1.84646, p = 0.867 - 0.184646 = 0.682354
	Value p = Value::leaf(Tensor({1}, 1.0f), true);
	SGD optimizer({p}, 0.1f, 0.9f, true, 0.2f);
	p.grad().at({0}) = 0.5f;
	optimizer.step();
	CHECK(p.data().get({0}) == doctest::Approx(0.867f));
	p.grad().at({0}) = 0.5f;
	optimizer.step();
	CHECK(p.data().get({0}) == doctest::Approx(0.682354f));
}

TEST_CASE("SGD rejects Nesterov without momentum and negative weight decay") {
	Value p = Value::leaf(Tensor({1}, 0.0f), true);
	CHECK_THROWS_AS(SGD({p}, 0.1f, 0.0f, true), std::invalid_argument);
	CHECK_THROWS_AS(SGD({p}, 0.1f, 0.9f, false, -0.1f), std::invalid_argument);
}

TEST_CASE("Nesterov SGD state has the same keys as plain momentum SGD and round-trips") {
	Value p = Value::leaf(Tensor({1}, 0.0f), true);
	Value q = Value::leaf(Tensor({1}, 0.0f), true);
	SGD plain({p}, 0.1f, 0.9f);
	SGD a({p}, 0.1f, 0.9f, true, 0.01f);
	SGD b({q}, 0.1f, 0.9f, true, 0.01f);
	const OptimizerState expected = plain.state();
	const OptimizerState got = a.state();
	REQUIRE(got.size() == expected.size());
	for (size_t i = 0; i < got.size(); i++) CHECK(got[i].first == expected[i].first);
	p.grad().at({0}) = 1.0f;
	a.step();
	q.data() = p.data();
	b.load_state(a.state());
	p.grad().at({0}) = 1.0f;
	q.grad().at({0}) = 1.0f;
	a.step();
	b.step();
	CHECK(q.data().get({0}) == p.data().get({0}));
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

namespace {

// Runs n steps on a fixed gradient and returns the parameter values
std::vector<float> run_steps(Optimizer& opt, Value& p, int n) {
	for (int i = 0; i < n; i++) {
		p.grad() = Tensor({2}, std::vector<float>{0.5f, -1.0f});
		opt.step();
	}
	return p.data().data();
}

}  // namespace

TEST_CASE("set_lr changes the next step size for SGD and Adam") {
	Value p = Value::leaf(Tensor({2}, std::vector<float>{1.0f, 1.0f}), true);
	SGD sgd({p}, 0.1f);
	CHECK(sgd.lr() == 0.1f);
	sgd.set_lr(0.5f);
	CHECK(run_steps(sgd, p, 1)[0] == doctest::Approx(1.0f - 0.5f * 0.5f));

	Value q = Value::leaf(Tensor({2}, std::vector<float>{1.0f, 1.0f}), true);
	Adam adam({q}, 0.1f);
	adam.set_lr(0.3f);
	CHECK(adam.lr() == 0.3f);
	// the first Adam step moves by about lr
	CHECK(run_steps(adam, q, 1)[0] == doctest::Approx(1.0f - 0.3f).epsilon(1e-4));
}

TEST_CASE("Optimizer base without a learning rate throws logic_error") {
	struct Plain : Optimizer {
		using Optimizer::Optimizer;
		void step() override {}
	} plain(std::vector<Value>{});
	CHECK_THROWS_AS(plain.lr(), std::logic_error);
	CHECK_THROWS_AS(plain.set_lr(1.0f), std::logic_error);
}

TEST_CASE("AdamW with weight_decay 0 equals Adam") {
	Value a = Value::leaf(Tensor({2}, std::vector<float>{1.0f, -2.0f}), true);
	Value b = Value::leaf(Tensor({2}, std::vector<float>{1.0f, -2.0f}), true);
	Adam adam({a}, 0.01f);
	AdamW adamw({b}, 0.01f, 0.9f, 0.999f, 1e-8f, 0.0f);
	auto ra = run_steps(adam, a, 10);
	auto rb = run_steps(adamw, b, 10);
	for (size_t i = 0; i < ra.size(); i++) CHECK(rb[i] == doctest::Approx(ra[i]).epsilon(1e-6));
}

TEST_CASE("AdamW decay shrinks a zero-gradient parameter by 1 - lr*wd per step") {
	Value p = Value::leaf(Tensor({2}, std::vector<float>{2.0f, -4.0f}), true);
	AdamW opt({p}, 0.1f, 0.9f, 0.999f, 1e-8f, 0.5f);
	for (int i = 0; i < 3; i++) opt.step();  // grad stays zero
	const float f = 1.0f - 0.1f * 0.5f;
	CHECK(p.data().get({0}) == doctest::Approx(2.0f * f * f * f).epsilon(1e-6));
	CHECK(p.data().get({1}) == doctest::Approx(-4.0f * f * f * f).epsilon(1e-6));
	CHECK(opt.lr() == 0.1f);
}

TEST_CASE("StepLR decays by gamma every step_size steps") {
	Value p = Value::leaf(Tensor({1}, 0.0f), true);
	SGD opt({p}, 0.1f);
	StepLR sched(opt, 5, 0.5f);
	CHECK(opt.lr() == 0.1f);
	std::vector<float> lrs;
	for (int i = 1; i <= 11; i++) {
		sched.step();
		lrs.push_back(opt.lr());
	}
	CHECK(lrs[3] == doctest::Approx(0.1f));    // step 4
	CHECK(lrs[4] == doctest::Approx(0.05f));   // step 5
	CHECK(lrs[9] == doctest::Approx(0.025f));  // step 10
	CHECK(sched.step_count() == 11);
	CHECK(sched.base_lr() == 0.1f);
	CHECK_THROWS_AS(StepLR(opt, 0), std::invalid_argument);
}

TEST_CASE("CosineAnnealingLR reaches eta_min at t_max and stays") {
	Value p = Value::leaf(Tensor({1}, 0.0f), true);
	SGD opt({p}, 1.0f);
	CosineAnnealingLR sched(opt, 10, 0.1f);
	sched.set_step(5);
	CHECK(opt.lr() == doctest::Approx(0.55f));
	sched.set_step(10);
	CHECK(opt.lr() == doctest::Approx(0.1f));
	sched.step();
	sched.step();
	CHECK(opt.lr() == doctest::Approx(0.1f));
	CHECK_THROWS_AS(CosineAnnealingLR(opt, 0), std::invalid_argument);
}

TEST_CASE("LinearWarmup ramps to base then follows the inner schedule") {
	Value p = Value::leaf(Tensor({1}, 0.0f), true);
	SGD opt({p}, 0.4f);
	LinearWarmup sched(opt, std::make_unique<StepLR>(opt, 2, 0.5f), 4);
	CHECK(opt.lr() == 0.0f);
	sched.step();
	CHECK(opt.lr() == doctest::Approx(0.1f));
	sched.set_step(4);
	CHECK(opt.lr() == doctest::Approx(0.4f));
	sched.set_step(6);
	CHECK(opt.lr() == doctest::Approx(0.2f));
}

TEST_CASE("set_step resumes identically to stepping n times") {
	Value p = Value::leaf(Tensor({1}, 0.0f), true);
	SGD a({p}, 0.3f), b({p}, 0.3f);
	CosineAnnealingLR sa(a, 20), sb(b, 20);
	for (int i = 0; i < 7; i++) sa.step();
	sb.set_step(7);
	CHECK(a.lr() == b.lr());
	sa.step();
	sb.step();
	CHECK(a.lr() == b.lr());
}

TEST_CASE("clip_grad_norm returns the pre-clip norm and rescales") {
	Value p = Value::leaf(Tensor({2}, 0.0f), true);
	p.grad() = Tensor({2}, std::vector<float>{3.0f, 4.0f});
	CHECK(clip_grad_norm({p}, 10.0f) == doctest::Approx(5.0f));
	CHECK(p.grad().get({0}) == 3.0f);
	CHECK(clip_grad_norm({p}, 1.0f) == doctest::Approx(5.0f));
	float n = std::sqrt(p.grad().get({0}) * p.grad().get({0}) + p.grad().get({1}) * p.grad().get({1}));
	CHECK(n == doctest::Approx(1.0f).epsilon(1e-5));
	Value none = Value::leaf(Tensor({2}, 0.0f), true);
	CHECK(clip_grad_norm({none}, 1.0f) == 0.0f);
}

TEST_CASE("clip_grad_value clamps each element") {
	Value p = Value::leaf(Tensor({3}, 0.0f), true);
	p.grad() = Tensor({3}, std::vector<float>{-5.0f, 0.5f, 7.0f});
	clip_grad_value({p}, 1.0f);
	CHECK(p.grad().data() == std::vector<float>{-1.0f, 0.5f, 1.0f});
}
