/**
 * test_random.cpp
 * Author: Miguel Mochizuki Silva
 * Description: Tests for manual_seed and derived generators
 */

#include "doctest/doctest.h"
#include "nn/data/dataloader.hpp"
#include "nn/random.hpp"

#include <vector>

using namespace nn;

namespace {

std::vector<uint32_t> draw(std::mt19937 g) {
	std::vector<uint32_t> v;
	for (int i = 0; i < 8; i++) {
		v.push_back(g());
	}
	return v;
}

class CountDataset : public Dataset {
public:
	size_t size() const override { return 32; }
	std::pair<Tensor, Tensor> get(size_t i) const override {
		return {Tensor({1}, static_cast<float>(i)), Tensor({1}, static_cast<float>(i))};
	}
};

std::vector<float> first_batch(DataLoader& dl) { return dl.get_batch(0).first.data().data(); }

}  // namespace

TEST_CASE("derive_generator is reproducible per seed") {
	manual_seed(7);
	CHECK(is_seeded());
	auto a = draw(derive_generator("a"));
	manual_seed(7);
	CHECK(draw(derive_generator("a")) == a);
}

TEST_CASE("derive_generator streams differ by call and by name") {
	manual_seed(7);
	auto a1 = draw(derive_generator("a"));
	auto a2 = draw(derive_generator("a"));
	auto b = draw(derive_generator("b"));
	CHECK(a1 != a2);
	CHECK(a1 != b);
}

TEST_CASE("other streams do not shift a stream") {
	manual_seed(3);
	draw(derive_generator("a"));
	auto plain = draw(derive_generator("a"));
	manual_seed(3);
	draw(derive_generator("a"));
	derive_generator("b");
	CHECK(draw(derive_generator("a")) == plain);
}

TEST_CASE("Tensor::random is reproducible and independent of DataLoader") {
	manual_seed(1);
	Tensor t1 = Tensor::random({16}, 0.0f, 1.0f);
	manual_seed(1);
	CountDataset ds;
	DataLoader dl(ds, 4, true);
	Tensor t2 = Tensor::random({16}, 0.0f, 1.0f);
	CHECK(t1.data() == t2.data());
	// consecutive calls continue one stream
	CHECK(Tensor::random({16}, 0.0f, 1.0f).data() != t2.data());
}

TEST_CASE("DataLoaders created in the same order shuffle identically") {
	CountDataset ds;
	manual_seed(5);
	DataLoader a1(ds, 8, true);
	DataLoader a2(ds, 8, true);
	manual_seed(5);
	DataLoader b1(ds, 8, true);
	DataLoader b2(ds, 8, true);
	CHECK(first_batch(a1) == first_batch(b1));
	CHECK(first_batch(a2) == first_batch(b2));
	CHECK(first_batch(a1) != first_batch(a2));
}
