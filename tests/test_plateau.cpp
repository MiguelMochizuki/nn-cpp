/**
 * test_plateau.cpp
 * Author: Miguel Mochizuki Silva
 * Description: Unit tests for the ReduceLROnPlateau scheduler
 */

#include "doctest/doctest.h"
#include "nn/optim/plateau.hpp"
#include "nn/optim/sgd.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

using namespace nn;

static Value dummy_param() { return Value::leaf(Tensor({1}, std::vector<float>{0.0f}), true); }

TEST_CASE("plateau reduces the lr after patience epochs without improvement (Max)") {
	SGD opt({dummy_param()}, 1.0f);
	ReduceLROnPlateau sched(opt, PlateauMode::Max, 0.5f, 2);
	sched.step(0.5f);
	sched.step(0.6f);
	sched.step(0.6f);
	sched.step(0.6f);
	CHECK(opt.lr() == 1.0f);
	CHECK(sched.num_bad_epochs() == 2);
	sched.step(0.6f);
	CHECK(opt.lr() == 0.5f);
	CHECK(sched.num_bad_epochs() == 0);
	CHECK(sched.best() == 0.6f);
}

TEST_CASE("plateau best starts at the worst value for the mode") {
	SGD opt({dummy_param()}, 1.0f);
	CHECK(ReduceLROnPlateau(opt, PlateauMode::Min).best() == std::numeric_limits<float>::infinity());
	CHECK(ReduceLROnPlateau(opt, PlateauMode::Max).best() == -std::numeric_limits<float>::infinity());
}

TEST_CASE("plateau improvement resets the bad-epoch counter") {
	SGD opt({dummy_param()}, 1.0f);
	ReduceLROnPlateau sched(opt, PlateauMode::Min, 0.5f, 2);
	sched.step(1.0f);
	sched.step(1.0f);
	CHECK(sched.num_bad_epochs() == 1);
	sched.step(0.5f);
	CHECK(sched.num_bad_epochs() == 0);
	sched.step(0.5f);
	CHECK(opt.lr() == 1.0f);
}

TEST_CASE("plateau threshold is relative: tiny gains do not count as improvement") {
	SGD opt({dummy_param()}, 1.0f);
	ReduceLROnPlateau sched(opt, PlateauMode::Min, 0.5f, 5, 0.1f);
	sched.step(10.0f);
	sched.step(9.5f); // better than 10 but not below 10 * 0.9
	CHECK(sched.num_bad_epochs() == 1);
	CHECK(sched.best() == 10.0f);
	sched.step(8.9f);
	CHECK(sched.num_bad_epochs() == 0);
	CHECK(sched.best() == doctest::Approx(8.9f));

	SGD opt2({dummy_param()}, 1.0f);
	ReduceLROnPlateau mx(opt2, PlateauMode::Max, 0.5f, 5, 0.1f);
	mx.step(10.0f);
	mx.step(10.5f); // not above 10 * 1.1
	CHECK(mx.num_bad_epochs() == 1);
	mx.step(11.5f);
	CHECK(mx.num_bad_epochs() == 0);
}

TEST_CASE("plateau min_lr clamps the reduction and later ones stay there") {
	SGD opt({dummy_param()}, 1.0f);
	ReduceLROnPlateau sched(opt, PlateauMode::Max, 0.5f, 0, 0.0f, 0, 0.4f);
	sched.step(1.0f);
	sched.step(1.0f);
	CHECK(opt.lr() == 0.5f);
	sched.step(1.0f);
	CHECK(opt.lr() == 0.4f);
	sched.step(1.0f);
	CHECK(opt.lr() == 0.4f);
}

TEST_CASE("plateau never raises the lr when it is already below min_lr") {
	SGD opt({dummy_param()}, 0.1f);
	ReduceLROnPlateau sched(opt, PlateauMode::Min, 0.5f, 0, 0.0f, 0, 0.4f);
	sched.step(1.0f);
	sched.step(1.0f);
	CHECK(opt.lr() == 0.1f);
}

TEST_CASE("plateau cooldown suppresses bad-epoch counting after a reduction") {
	SGD opt({dummy_param()}, 1.0f);
	ReduceLROnPlateau sched(opt, PlateauMode::Min, 0.5f, 1, 0.0f, 2);
	sched.step(1.0f);
	sched.step(1.0f); // bad 1
	CHECK(opt.lr() == 1.0f);
	sched.step(1.0f); // bad 2 > patience, reduces
	CHECK(opt.lr() == 0.5f);
	sched.step(1.0f); // cooldown
	CHECK(sched.num_bad_epochs() == 0);
	sched.step(1.0f); // cooldown
	CHECK(sched.num_bad_epochs() == 0);
	CHECK(opt.lr() == 0.5f);
	sched.step(1.0f); // counted again: bad 1
	CHECK(sched.num_bad_epochs() == 1);
	CHECK(opt.lr() == 0.5f);
	sched.step(1.0f); // bad 2
	CHECK(opt.lr() == 0.25f);
}

TEST_CASE("plateau treats a NaN metric as a bad epoch and never as best") {
	SGD opt({dummy_param()}, 1.0f);
	ReduceLROnPlateau sched(opt, PlateauMode::Min, 0.5f, 2);
	sched.step(1.0f);
	sched.step(std::nanf(""));
	CHECK(sched.num_bad_epochs() == 1);
	CHECK(sched.best() == 1.0f);
	sched.step(std::nanf(""));
	CHECK(opt.lr() == 1.0f);
	sched.step(std::nanf(""));
	CHECK(opt.lr() == 0.5f);
	CHECK(sched.best() == 1.0f);

	SGD opt2({dummy_param()}, 1.0f);
	ReduceLROnPlateau first(opt2, PlateauMode::Max);
	first.step(std::nanf(""));
	CHECK(first.best() == -std::numeric_limits<float>::infinity());
	CHECK(first.num_bad_epochs() == 1);
}

TEST_CASE("plateau skips a reduction smaller than 1e-8") {
	SGD opt({dummy_param()}, 1e-8f);
	ReduceLROnPlateau sched(opt, PlateauMode::Min, 0.5f, 0);
	sched.step(1.0f);
	sched.step(1.0f);
	CHECK(opt.lr() == 1e-8f);
}

TEST_CASE("plateau rejects invalid arguments") {
	SGD opt({dummy_param()}, 1.0f);
	CHECK_THROWS_AS(ReduceLROnPlateau(opt, PlateauMode::Min, 0.0f), std::invalid_argument);
	CHECK_THROWS_AS(ReduceLROnPlateau(opt, PlateauMode::Min, 1.0f), std::invalid_argument);
	CHECK_THROWS_AS(ReduceLROnPlateau(opt, PlateauMode::Min, -0.5f), std::invalid_argument);
	CHECK_THROWS_AS(ReduceLROnPlateau(opt, PlateauMode::Min, std::nanf("")), std::invalid_argument);
	CHECK_THROWS_AS(ReduceLROnPlateau(opt, PlateauMode::Min, 0.5f, 1, -1e-4f), std::invalid_argument);
	CHECK_THROWS_AS(ReduceLROnPlateau(opt, PlateauMode::Min, 0.5f, 1, 1e-4f, 0, -0.1f), std::invalid_argument);
}

TEST_CASE("plateau with patience 0 does not reduce during cooldown") {
	SGD opt({dummy_param()}, 1.0f);
	ReduceLROnPlateau sched(opt, PlateauMode::Min, 0.5f, 0, 0.0f, 1);
	sched.step(1.0f);
	sched.step(1.0f); // reduces
	sched.step(1.0f); // cooldown
	CHECK(opt.lr() == 0.5f);
	sched.step(1.0f);
	CHECK(opt.lr() == 0.25f);
}
