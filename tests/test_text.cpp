/**
 * test_text.cpp
 * Author: Miguel Mochizuki Silva
 * Description: Unit tests for CharVocab and SequenceDataset
 */

#include "doctest/doctest.h"
#include "nn/data/dataloader.hpp"
#include "nn/data/text.hpp"

#include <stdexcept>

using namespace nn;

TEST_CASE("CharVocab encodes sorted distinct bytes and decode inverts it") {
	CharVocab v("hello");
	CHECK(v.size() == 4);
	std::vector<float> ids = v.encode("hello");
	CHECK(ids == std::vector<float>{1, 0, 2, 2, 3});
	CHECK(v.decode(ids) == "hello");
}

TEST_CASE("CharVocab throws on unknown characters and invalid ids") {
	CharVocab v("hello");
	CHECK_THROWS_AS(v.encode("hex"), std::invalid_argument);
	CHECK_THROWS_AS(v.decode({4}), std::invalid_argument);
	CHECK_THROWS_AS(v.decode({-1}), std::invalid_argument);
	CHECK_THROWS_AS(v.decode({0.5f}), std::invalid_argument);
}

TEST_CASE("SequenceDataset yields shifted sliding windows") {
	SequenceDataset ds({10, 11, 12, 13, 14, 15}, 3);
	CHECK(ds.size() == 3);
	auto [x, y] = ds.get(1);
	CHECK(x.shape() == std::vector<size_t>{3});
	CHECK(x.data() == std::vector<float>{11, 12, 13});
	CHECK(y.data() == std::vector<float>{12, 13, 14});
}

TEST_CASE("SequenceDataset rejects bad arguments and indices") {
	CHECK_THROWS_AS(SequenceDataset({1, 2, 3}, 3), std::invalid_argument);
	CHECK_THROWS_AS(SequenceDataset({1, 2, 3}, 0), std::invalid_argument);
	SequenceDataset ds({1, 2, 3, 4}, 2);
	CHECK_THROWS_AS(ds.get(2), TensorIndexError);
}

TEST_CASE("SequenceDataset batches to {B,T} through DataLoader") {
	SequenceDataset ds({0, 1, 2, 3, 4, 5}, 3);
	DataLoader loader(ds, 2, false);
	auto [x, y] = loader.get_batch(0);
	CHECK(x.data().shape() == std::vector<size_t>{2, 3});
	CHECK(y.shape() == std::vector<size_t>{2, 3});
}
