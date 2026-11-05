/**
 * test_dataloader.cpp
 * Author: Miguel Mochizuki Silva
 * Description: Unit tests for DataLoader, exercised against a fake in-memory Dataset
 */

#include "doctest/doctest.h"
#include "nn/data/dataloader.hpp"
#include "nn/data/subset.hpp"
#include "nn/data/tensor_dataset.hpp"
#include "nn/data/transform.hpp"
#include "nn/random.hpp"

#include <algorithm>
#include <set>
#include <stdexcept>

using namespace nn;

namespace {

/*
 * Fake dataset for tests: sample i's input and target both encode i as a float,
 * so any mismatch after shuffling or batching is directly observable.
 *
 * Attributes:
 * size_t count_: number of samples
 */
class IndexEncodingDataset : public Dataset {
public:
	explicit IndexEncodingDataset(size_t count) : count_(count) {}

	size_t size() const override { return count_; }

	std::pair<Tensor, Tensor> get(size_t index) const override {
		Tensor x({1}, static_cast<float>(index));
		Tensor y({1}, static_cast<float>(index));
		return {x, y};
	}

private:
	size_t count_;
};

}  // namespace

TEST_CASE("num_batches accounts for a partial final batch") {
	IndexEncodingDataset dataset(5);
	DataLoader loader(dataset, 2, false);
	CHECK(loader.num_batches() == 3);
}

TEST_CASE("DataLoader throws on a zero batch_size") {
	IndexEncodingDataset dataset(5);
	CHECK_THROWS_AS(DataLoader(dataset, 0, false), TensorShapeError);
}

TEST_CASE("get_batch throws for a batch_index past the end") {
	IndexEncodingDataset dataset(5);
	DataLoader loader(dataset, 2, false);
	CHECK_THROWS_AS(loader.get_batch(3), TensorIndexError);
}

TEST_CASE("the final partial batch has the correct, smaller size") {
	IndexEncodingDataset dataset(5);
	DataLoader loader(dataset, 2, false);
	auto [x, y] = loader.get_batch(2);
	CHECK(x.data().shape() == std::vector<size_t>{1, 1});
	CHECK(y.shape() == std::vector<size_t>{1, 1});
}

TEST_CASE("unshuffled batches preserve dataset order") {
	IndexEncodingDataset dataset(4);
	DataLoader loader(dataset, 2, false);
	auto [x0, y0] = loader.get_batch(0);
	CHECK(x0.data().get({0, 0}) == 0.0f);
	CHECK(x0.data().get({1, 0}) == 1.0f);
}

TEST_CASE("every batch's image-encoded index matches its target-encoded index after shuffling") {
	IndexEncodingDataset dataset(20);
	DataLoader loader(dataset, 4, true);
	loader.reset_epoch();
	for (size_t b = 0; b < loader.num_batches(); b++) {
		auto [x, y] = loader.get_batch(b);
		for (size_t i = 0; i < x.data().shape()[0]; i++) {
			CHECK(x.data().get({i, 0}) == y.get({i, 0}));
		}
	}
}

TEST_CASE("TensorDataset returns row slices with the remaining shape") {
	TensorDataset ds(Tensor({3, 2, 2}, std::vector<float>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}),
					 Tensor({3}, std::vector<float>{5, 6, 7}));
	CHECK(ds.size() == 3);
	auto [x, y] = ds.get(1);
	CHECK(x.shape() == std::vector<size_t>{2, 2});
	CHECK(x.data() == std::vector<float>{4, 5, 6, 7});
	CHECK(y.shape().empty());
	CHECK(y.data()[0] == 6.0f);
	CHECK_THROWS_AS(ds.get(3), TensorIndexError);
	CHECK_THROWS_AS(TensorDataset(Tensor({3, 1}, 0.0f), Tensor({2}, 0.0f)), TensorShapeError);
	DataLoader loader(ds, 2, false);
	CHECK(loader.get_batch(0).second.shape() == std::vector<size_t>{2});
}

TEST_CASE("Subset selects and validates indices") {
	IndexEncodingDataset base(5);
	Subset sub(base, {4, 0});
	CHECK(sub.size() == 2);
	CHECK(sub.get(0).first.data()[0] == 4.0f);
	CHECK_THROWS_AS(sub.get(2), TensorIndexError);
	CHECK_THROWS_AS(Subset(base, {5}), TensorIndexError);
}

TEST_CASE("random_split is reproducible, disjoint and covering") {
	IndexEncodingDataset base(10);
	auto ids = [](const Subset& s) {
		std::vector<float> v;
		for (size_t i = 0; i < s.size(); i++) v.push_back(s.get(i).first.data()[0]);
		return v;
	};
	auto a = random_split(base, {0.7, 0.3}, 1);
	auto b = random_split(base, {0.7, 0.3}, 1);
	auto c = random_split(base, {0.7, 0.3}, 2);
	CHECK(a[0].size() + a[1].size() == 10);
	CHECK(ids(a[0]) == ids(b[0]));
	CHECK(ids(a[0]) != ids(c[0]));
	std::set<float> seen;
	for (auto& s : a) for (float v : ids(s)) seen.insert(v);
	CHECK(seen.size() == 10);
	auto r = random_split(base, {0.33, 0.33, 0.34}, 0);  // floors 3,3,3, remainder to the last
	CHECK(r[2].size() == 4);
}

TEST_CASE("random_split rejects bad fractions and empty splits") {
	IndexEncodingDataset base(10);
	CHECK_THROWS_AS(random_split(base, {0.5, 0.3}, 0), std::invalid_argument);
	CHECK_THROWS_AS(random_split(base, {1.5, -0.5}, 0), std::invalid_argument);
	CHECK_THROWS_AS(random_split(base, {}, 0), std::invalid_argument);
	CHECK_THROWS_AS(random_split(IndexEncodingDataset(3), {0.9, 0.05, 0.05}, 0), std::invalid_argument);
}

TEST_CASE("drop_last yields only full batches") {
	IndexEncodingDataset base(5);
	DataLoader keep(base, 2, false, false);
	DataLoader drop(base, 2, false, true);
	CHECK(keep.num_batches() == 3);
	CHECK(drop.num_batches() == 2);
	CHECK(drop.get_batch(1).first.data().shape()[0] == 2);
	CHECK_THROWS_AS(drop.get_batch(2), TensorIndexError);
}

namespace {

Tensor ramp(std::vector<size_t> shape) {
	size_t n = 1;
	for (size_t d : shape) n *= d;
	std::vector<float> v(n);
	for (size_t i = 0; i < n; i++) v[i] = static_cast<float>(i + 1);
	return Tensor(shape, v);
}

}  // namespace

TEST_CASE("Normalize subtracts the mean and divides by std") {
	Normalize norm(2.0f, 4.0f);
	CHECK(norm(Tensor({3}, std::vector<float>{2, 6, -2})).data() == std::vector<float>{0, 1, -1});
	CHECK_THROWS_AS(Normalize(0.0f, 0.0f), std::invalid_argument);
	CHECK_THROWS_AS(Normalize(0.0f, -1.0f), std::invalid_argument);
}

TEST_CASE("RandomHorizontalFlip mirrors columns at p=1, is identity at p=0 and replays under a seed") {
	Tensor x = ramp({1, 2, 3});
	CHECK(RandomHorizontalFlip(1.0f)(x).data() == std::vector<float>{3, 2, 1, 6, 5, 4});
	CHECK(RandomHorizontalFlip(0.0f)(x).data() == x.data());
	auto decisions = [&](uint64_t seed) {
		manual_seed(seed);
		RandomHorizontalFlip flip(0.5f);
		std::vector<bool> d;
		for (int i = 0; i < 32; i++) d.push_back(flip(x).data() != x.data());
		return d;
	};
	CHECK(decisions(3) == decisions(3));
	CHECK(decisions(3) != decisions(4));
	auto d = decisions(3);
	CHECK(std::count(d.begin(), d.end(), true) > 0);
	CHECK(std::count(d.begin(), d.end(), false) > 0);
	CHECK_THROWS_AS(RandomHorizontalFlip(1.5f), std::invalid_argument);
}

TEST_CASE("RandomShift translates with zero fill and keeps the shape") {
	Tensor x = ramp({2, 4, 5});
	CHECK(RandomShift(0)(x).data() == x.data());
	manual_seed(9);
	RandomShift shift(2);
	bool moved = false;
	for (int i = 0; i < 20; i++) {
		Tensor y = shift(x);
		CHECK(y.shape() == x.shape());
		// recover the shift from the first channel's content and rebuild the expected tensor
		bool matched = false;
		for (long dy = -2; dy <= 2 && !matched; dy++) {
			for (long dx = -2; dx <= 2 && !matched; dx++) {
				Tensor e(x.shape(), 0.0f);
				for (long c = 0; c < 2; c++)
					for (long r = 0; r < 4; r++)
						for (long q = 0; q < 5; q++) {
							long sr = r - dy, sq = q - dx;
							if (sr >= 0 && sr < 4 && sq >= 0 && sq < 5) e.data()[(c * 4 + r) * 5 + q] = x.data()[(c * 4 + sr) * 5 + sq];
						}
				if (e.data() == y.data()) {
					matched = true;
					if (dy != 0 || dx != 0) moved = true;
				}
			}
		}
		CHECK(matched);
	}
	CHECK(moved);
	CHECK_THROWS_AS(shift(Tensor({4, 5}, 0.0f)), TensorShapeError);
}

TEST_CASE("Compose applies transforms in order") {
	std::vector<std::unique_ptr<Transform>> steps;
	steps.push_back(std::make_unique<Normalize>(1.0f, 1.0f));
	steps.push_back(std::make_unique<Normalize>(0.0f, 2.0f));
	Compose c(std::move(steps));
	CHECK(c(Tensor({1}, 5.0f)).data()[0] == doctest::Approx(2.0f));
	std::vector<std::unique_ptr<Transform>> order;
	order.push_back(std::make_unique<RandomHorizontalFlip>(1.0f));
	order.push_back(std::make_unique<Normalize>(1.0f, 1.0f));
	CHECK(Compose(std::move(order))(Tensor({2}, std::vector<float>{1, 3})).data() == std::vector<float>{2, 0});
}

TEST_CASE("compute_mean_std matches the population statistics") {
	TensorDataset ds(Tensor({2, 2}, std::vector<float>{1, 2, 3, 6}), Tensor({2}, 0.0f));
	auto [mean, sd] = compute_mean_std(ds);
	CHECK(mean == doctest::Approx(3.0f));
	CHECK(sd == doctest::Approx(std::sqrt(3.5f)));  // squared deviations 4,1,0,9 over 4 elements -> 3.5
	CHECK_THROWS_AS(compute_mean_std(IndexEncodingDataset(0)), std::invalid_argument);
}

TEST_CASE("TransformedDataset transforms inputs only") {
	TensorDataset ds(Tensor({2, 1}, std::vector<float>{4, 8}), Tensor({2}, std::vector<float>{7, 9}));
	TransformedDataset t(ds, std::make_shared<Normalize>(4.0f, 2.0f));
	auto [x, y] = t.get(1);
	CHECK(t.size() == 2);
	CHECK(x.data()[0] == doctest::Approx(2.0f));
	CHECK(y.data()[0] == 9.0f);
	CHECK_THROWS_AS(TransformedDataset(ds, nullptr), std::invalid_argument);
}

TEST_CASE("a transform's stream is unaffected by creating another transform") {
	Tensor x = ramp({1, 2, 3});
	auto run = [&](bool extra) {
		manual_seed(21);
		RandomHorizontalFlip flip(0.5f);
		std::unique_ptr<RandomShift> other;
		if (extra) other = std::make_unique<RandomShift>(2);
		std::vector<bool> d;
		for (int i = 0; i < 32; i++) {
			if (other) (*other)(Tensor({1, 4, 4}, 0.0f));
			d.push_back(flip(x).data() != x.data());
		}
		return d;
	};
	CHECK(run(false) == run(true));
}
