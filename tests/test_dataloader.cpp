/**
 * test_dataloader.cpp
 * Author: Miguel Mochizuki Silva
 * Description: Unit tests for DataLoader, exercised against a fake in-memory Dataset
 */

#include "doctest/doctest.h"
#include "nn/data/dataloader.hpp"

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
