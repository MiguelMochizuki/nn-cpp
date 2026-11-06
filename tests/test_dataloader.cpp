/**
 * test_dataloader.cpp
 * Author: Miguel Mochizuki Silva
 * Description: Unit tests for DataLoader, exercised against a fake in-memory Dataset
 */

#include "doctest/doctest.h"
#include "nn/data/dataloader.hpp"
#include "nn/data/subset.hpp"
#include "nn/data/fer2013.hpp"
#include "nn/data/tensor_dataset.hpp"
#include "nn/data/transform.hpp"
#include "nn/random.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
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

TEST_CASE("random_split permutation is pinned for a fixed seed and covers every index once") {
	IndexEncodingDataset base(10);
	auto parts = random_split(base, {0.5, 0.5}, 42);
	std::vector<int> order;
	for (auto& s : parts) for (size_t i = 0; i < s.size(); i++) order.push_back(static_cast<int>(s.get(i).first.data()[0]));
	CHECK(order == std::vector<int>{1, 3, 9, 7, 6, 0, 8, 4, 5, 2});
	std::vector<int> sorted = order;
	std::sort(sorted.begin(), sorted.end());
	for (int i = 0; i < 10; i++) CHECK(sorted[i] == i);
}

TEST_CASE("random_split folds the high seed bits into the engine seed") {
	IndexEncodingDataset base(10);
	auto first = [&](uint64_t seed) {
		std::vector<float> v;
		auto parts = random_split(base, {0.5, 0.5}, seed);
		for (size_t i = 0; i < parts[0].size(); i++) v.push_back(parts[0].get(i).first.data()[0]);
		return v;
	};
	CHECK(first(5) != first(5 + (uint64_t{1} << 32)));
	// high and low halves cancel: 5 ^ 0 == 0 ^ 5, so these seeds fold to the same engine seed
	CHECK(first(5) == first(uint64_t{5} << 32));
}

TEST_CASE("a seeded DataLoader epoch has a pinned order and visits every index once") {
	IndexEncodingDataset base(10);
	manual_seed(42);
	DataLoader loader(base, 10, true);
	auto order = [&]() {
		Value x = loader.get_batch(0).first;
		std::vector<int> v;
		for (size_t i = 0; i < 10; i++) v.push_back(static_cast<int>(x.data().get({i, 0})));
		return v;
	};
	CHECK(order() == std::vector<int>{1, 2, 3, 7, 6, 8, 0, 5, 9, 4});
	std::vector<int> sorted = order();
	std::sort(sorted.begin(), sorted.end());
	for (int i = 0; i < 10; i++) CHECK(sorted[i] == i);
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

TEST_CASE("RandomHorizontalFlip returns a zero-width sample unchanged") {
	Tensor empty({2, 0});
	Tensor out = RandomHorizontalFlip(1.0f)(empty);
	CHECK(out.shape() == empty.shape());
	CHECK(out.size() == 0);
}

TEST_CASE("Compose rejects a null step") {
	std::vector<std::unique_ptr<Transform>> steps;
	steps.push_back(std::make_unique<Normalize>(0.0f, 1.0f));
	steps.push_back(nullptr);
	CHECK_THROWS_AS(Compose(std::move(steps)), std::invalid_argument);
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

namespace {

// Writes text to a fresh file in the system temp directory and removes it on destruction
struct TempFile {
	std::filesystem::path path;
	explicit TempFile(const std::string& name, const std::string& text) {
		path = std::filesystem::temp_directory_path() / ("nn_cpp_test_" + name);
		std::ofstream(path, std::ios::binary) << text;
	}
	~TempFile() { std::error_code ec; std::filesystem::remove(path, ec); }
};

// size x size image whose pixels all equal v
std::string pixels_of(int v, int size = 2) {
	std::string s;
	for (int i = 0; i < size * size; i++) s += (i ? " " : "") + std::to_string(v);
	return s;
}

std::string fer_text(const std::string& eol, bool quote_pixels) {
	auto q = [&](const std::string& p) { return quote_pixels ? "\"" + p + "\"" : p; };
	return "emotion,pixels,Usage" + eol +
		   "0," + q(pixels_of(0)) + ",Training" + eol +
		   "3," + q(pixels_of(255)) + ",PublicTest" + eol +
		   "3," + q(pixels_of(51)) + ",Training" + eol +
		   "6," + q(pixels_of(255)) + ",PrivateTest" + eol +
		   "1," + q(pixels_of(102)) + ",Other" + eol + eol;
}

}  // namespace

TEST_CASE("Fer2013Dataset parses LF, CRLF and quoted variants with a trailing blank line") {
	for (auto [eol, quoted] : {std::pair<std::string, bool>{"\n", false}, {"\r\n", false}, {"\n", true}, {"\r\n", true}}) {
		TempFile f("fer_ok.csv", fer_text(eol, quoted));
		Fer2013Dataset all(f.path.string(), "", 2);
		CHECK(all.size() == 5);
		auto [x, y] = all.get(2);
		CHECK(x.shape() == std::vector<size_t>{1, 2, 2});
		CHECK(x.data()[0] == doctest::Approx(51.0f / 255.0f));
		CHECK(y.shape().empty());
		CHECK(y.data()[0] == 3.0f);
		CHECK(all.get(1).first.data()[3] == doctest::Approx(1.0f));
		CHECK(all.class_counts() == std::vector<size_t>{1, 1, 0, 2, 0, 0, 1});

		Fer2013Dataset train(f.path.string(), "Training", 2);
		CHECK(train.size() == 2);
		CHECK(train.class_counts() == std::vector<size_t>{1, 0, 0, 1, 0, 0, 0});
		CHECK(Fer2013Dataset(f.path.string(), "PublicTest", 2).size() == 1);
		CHECK(Fer2013Dataset(f.path.string(), "Nothing", 2).size() == 0);
		CHECK_THROWS_AS(all.get(5), TensorIndexError);
	}
}

TEST_CASE("Fer2013Dataset locates columns by header name") {
	TempFile f("fer_cols.csv", "Usage,pixels,emotion\nTraining,1 2 3 4,5\n");
	Fer2013Dataset ds(f.path.string(), "Training", 2);
	CHECK(ds.get(0).second.data()[0] == 5.0f);
}

TEST_CASE("Fer2013Dataset errors carry the line number and reason") {
	auto message = [](const std::string& body) {
		TempFile f("fer_bad.csv", "emotion,pixels,Usage\n" + body);
		try {
			Fer2013Dataset(f.path.string(), "", 2);
		} catch (const std::runtime_error& e) {
			return std::string(e.what());
		}
		return std::string("no error");
	};
	auto has = [](const std::string& m, const std::string& part) { return m.find(part) != std::string::npos; };
	std::string ok = "0," + pixels_of(1) + ",Training\n";
	std::string m = message(ok + ok + "0,1 2 3,Training\n");
	CHECK(has(m, "line 4"));
	CHECK(has(m, "expected 4 pixels, found 3"));
	m = message(ok + "0,1 2 3 256,Training\n");
	CHECK(has(m, "line 3"));
	CHECK(has(m, "[0, 255]"));
	CHECK(has(message("7," + pixels_of(1) + ",Training\n"), "emotion"));
	CHECK(has(message("x," + pixels_of(1) + ",Training\n"), "line 2"));
	CHECK(has(message("0\n"), "fields"));
	CHECK(has(message("0,\"1 2 3 4,Training\n"), "unterminated"));

	CHECK_THROWS_AS(Fer2013Dataset((std::filesystem::temp_directory_path() / "nn_cpp_no_such.csv").string()), std::runtime_error);
	TempFile no_cols("fer_nocols.csv", "a,b\n1,2\n");
	CHECK_THROWS_AS(Fer2013Dataset(no_cols.path.string()), std::runtime_error);
	TempFile empty("fer_empty.csv", "");
	CHECK_THROWS_AS(Fer2013Dataset(empty.path.string()), std::runtime_error);
}

TEST_CASE("a weighted loader draws samples in proportion to their weights") {
	IndexEncodingDataset dataset(4);
	std::vector<double> w = {1.0, 0.0, 3.0, 6.0};  // sample 1 must never appear
	DataLoader loader(dataset, 50, w, 10000);
	CHECK(loader.num_batches() == 200);
	std::vector<size_t> hits(4, 0);
	for (size_t b = 0; b < loader.num_batches(); b++) {
		auto [x, y] = loader.get_batch(b);
		for (float v : y.data()) hits[static_cast<size_t>(v)]++;
	}
	CHECK(hits[1] == 0);
	CHECK(hits[0] + hits[2] + hits[3] == 10000);
	CHECK(hits[0] == doctest::Approx(1000.0).epsilon(0.15));
	CHECK(hits[2] == doctest::Approx(3000.0).epsilon(0.08));
	CHECK(hits[3] == doctest::Approx(6000.0).epsilon(0.05));
}

TEST_CASE("a weighted loader defaults to one epoch of dataset.size() samples and repeats under a seed") {
	IndexEncodingDataset dataset(7);
	std::vector<double> w(7, 1.0);
	auto epoch = [&](unsigned long long seed) {
		manual_seed(seed);
		DataLoader loader(dataset, 3, w);
		CHECK(loader.num_batches() == 3);  // ceil(7 / 3)
		std::vector<float> seen;
		for (size_t b = 0; b < loader.num_batches(); b++) {
			auto [x, y] = loader.get_batch(b);
			for (float v : y.data()) seen.push_back(v);
		}
		CHECK(seen.size() == 7);
		return seen;
	};
	CHECK(epoch(5) == epoch(5));
	CHECK(epoch(5) != epoch(6));
	DataLoader dropped(dataset, 3, w, 0, true);
	CHECK(dropped.num_batches() == 2);
}

TEST_CASE("a weighted loader redraws on reset_epoch and rejects bad weights") {
	IndexEncodingDataset dataset(50);
	std::vector<double> w(50, 1.0);
	DataLoader loader(dataset, 50, w);
	auto first = loader.get_batch(0).second.data();
	loader.reset_epoch();
	CHECK(loader.get_batch(0).second.data() != first);
	CHECK_THROWS_AS(DataLoader(dataset, 4, std::vector<double>(49, 1.0)), std::invalid_argument);
	CHECK_THROWS_AS(DataLoader(dataset, 4, std::vector<double>(50, 0.0)), std::invalid_argument);
	std::vector<double> neg(50, 1.0);
	neg[3] = -1.0;
	CHECK_THROWS_AS(DataLoader(dataset, 4, neg), std::invalid_argument);
	neg[3] = std::numeric_limits<double>::infinity();
	CHECK_THROWS_AS(DataLoader(dataset, 4, neg), std::invalid_argument);
	CHECK_THROWS_AS(DataLoader(dataset, 0, w), TensorShapeError);
}

TEST_CASE("balanced_sample_weights equalise the classes") {
	std::vector<size_t> labels = {0, 0, 0, 0, 1, 2, 2};
	std::vector<double> w = balanced_sample_weights(labels, 4);  // class 3 is absent
	CHECK(w[0] == doctest::Approx(0.25));
	CHECK(w[4] == doctest::Approx(1.0));
	CHECK(w[5] == doctest::Approx(0.5));
	double per_class[3] = {0, 0, 0};
	for (size_t i = 0; i < labels.size(); i++) per_class[labels[i]] += w[i];
	CHECK(per_class[0] == doctest::Approx(1.0));
	CHECK(per_class[1] == doctest::Approx(1.0));
	CHECK(per_class[2] == doctest::Approx(1.0));
	CHECK_THROWS_AS(balanced_sample_weights({0, 4}, 4), std::invalid_argument);
	CHECK(balanced_sample_weights({}, 3).empty());
}

namespace {

Tensor index_image(std::vector<size_t> shape) {
	Tensor t(shape, 0.0f);
	for (size_t i = 0; i < t.size(); i++) t.data()[i] = static_cast<float>(i);
	return t;
}

bool all_finite(const Tensor& t) {
	for (float v : t.data())
		if (!std::isfinite(v)) return false;
	return true;
}

}  // namespace

TEST_CASE("RandomCrop returns a contiguous window and validates the size") {
	Tensor same = index_image({1, 7, 7});
	CHECK(RandomCrop(7)(same).data() == same.data());
	Tensor big = index_image({3, 33, 33});
	manual_seed(4);
	RandomCrop crop(7);
	bool moved = false;
	for (int i = 0; i < 20; i++) {
		Tensor y = crop(big);
		REQUIRE(y.shape() == std::vector<size_t>({3, 7, 7}));
		// value = (c*33 + r)*33 + q, so the first element fixes the offset
		size_t first = static_cast<size_t>(y.data()[0]);
		size_t oy = (first / 33) % 33, ox = first % 33;
		CHECK(oy + 7 <= 33);
		CHECK(ox + 7 <= 33);
		if (oy != 0 || ox != 0) moved = true;
		for (size_t c = 0; c < 3; c++)
			for (size_t r = 0; r < 7; r++)
				for (size_t q = 0; q < 7; q++) CHECK(y.data()[(c * 7 + r) * 7 + q] == big.data()[(c * 33 + oy + r) * 33 + ox + q]);
	}
	CHECK(moved);
	CHECK_THROWS_AS(RandomCrop(8)(same), std::invalid_argument);
	CHECK_THROWS_AS(RandomCrop(0), std::invalid_argument);
	CHECK_THROWS_AS(RandomCrop(3)(Tensor({5, 5}, 0.0f)), TensorShapeError);
	manual_seed(5);
	Tensor a = RandomCrop(7)(big);
	manual_seed(5);
	Tensor b = RandomCrop(7)(big);
	CHECK(a.data() == b.data());
}

TEST_CASE("RandomAffine is the identity for zero ranges or p = 0") {
	for (auto shape : {std::vector<size_t>{1, 7, 7}, {3, 33, 33}, {1, 1, 1}}) {
		Tensor x = index_image(shape);
		CHECK(RandomAffine(0, 0, 0, 1.0f)(x).data() == x.data());
		CHECK(RandomAffine(30, 0.5f, 0.5f, 0.0f)(x).data() == x.data());
	}
}

TEST_CASE("RandomAffine translates, rotates and scales with bilinear zero-fill") {
	Tensor dot({1, 9, 9}, 0.0f);
	dot.data()[4 * 9 + 4] = 1.0f;
	manual_seed(2);
	RandomAffine shift(0, 0.5f, 0, 1.0f);
	int changed = 0, up = 0, down = 0, left = 0, right = 0;
	for (int i = 0; i < 200; i++) {
		Tensor y = shift(dot);
		int nonzero = 0;
		float mass = 0.0f, my = 0.0f, mx = 0.0f;
		for (size_t k = 0; k < y.size(); k++) {
			float v = y.data()[k];
			if (v != 0.0f) nonzero++;
			mass += v;
			my += v * static_cast<float>(k / 9);
			mx += v * static_cast<float>(k % 9);
			CHECK(v >= 0.0f);
		}
		CHECK(nonzero <= 4);
		CHECK(mass <= 1.0f + 1e-5f);
		if (y.data() != dot.data()) changed++;
		// a fully inside dot keeps unit mass, and its centre of mass is the translation
		if (mass > 0.999f) {
			if (my / mass < 3.999f) up++;
			if (my / mass > 4.001f) down++;
			if (mx / mass < 3.999f) left++;
			if (mx / mass > 4.001f) right++;
		}
	}
	CHECK(changed > 150);
	CHECK(up > 20);
	CHECK(down > 20);
	CHECK(left > 20);
	CHECK(right > 20);
	Tensor ones({1, 33, 33}, 1.0f);
	RandomAffine rot(10, 0, 0, 1.0f);
	manual_seed(3);
	for (int i = 0; i < 10; i++) {
		Tensor y = rot(ones);
		CHECK(y.data()[16 * 33 + 16] == doctest::Approx(1.0f));
		for (float v : y.data()) {
			CHECK(v >= 0.0f);
			CHECK(v <= 1.0f + 1e-5f);
		}
	}
	// extreme settings (large shift, shrink) keep values finite, tiny image included
	RandomAffine wild(180, 1.0f, 0.9f, 1.0f);
	for (int i = 0; i < 20; i++) {
		CHECK(all_finite(wild(ones)));
		CHECK(all_finite(wild(Tensor({1, 1, 1}, 1.0f))));
	}
	manual_seed(6);
	Tensor a = RandomAffine(15, 0.1f, 0.1f)(ones);
	manual_seed(6);
	Tensor b = RandomAffine(15, 0.1f, 0.1f)(ones);
	CHECK(a.data() == b.data());
	CHECK_THROWS_AS(RandomAffine(-1, 0, 0), std::invalid_argument);
	CHECK_THROWS_AS(RandomAffine(std::numeric_limits<float>::infinity(), 0, 0), std::invalid_argument);
	CHECK_THROWS_AS(RandomAffine(std::nanf(""), 0, 0), std::invalid_argument);
	CHECK_THROWS_AS(RandomAffine(0, 1.5f, 0), std::invalid_argument);
	CHECK_THROWS_AS(RandomAffine(0, 0, 1.0f), std::invalid_argument);
	CHECK_THROWS_AS(RandomAffine(0, 0, 0, 2.0f), std::invalid_argument);
	CHECK_THROWS_AS(shift(Tensor({4, 5}, 0.0f)), TensorShapeError);
}

TEST_CASE("ten_crop returns the four corners and centre then their mirrors") {
	Tensor x = index_image({1, 5, 5});
	auto crops = ten_crop(x, 3);
	REQUIRE(crops.size() == 10);
	auto window = [&](size_t oy, size_t ox, bool mirror) {
		std::vector<float> v;
		for (size_t r = 0; r < 3; r++)
			for (size_t q = 0; q < 3; q++) v.push_back(x.data()[(oy + r) * 5 + ox + (mirror ? 2 - q : q)]);
		return v;
	};
	size_t offs[5][2] = {{0, 0}, {0, 2}, {2, 0}, {2, 2}, {1, 1}};
	for (size_t i = 0; i < 5; i++) {
		CHECK(crops[i].shape() == std::vector<size_t>({1, 3, 3}));
		CHECK(crops[i].data() == window(offs[i][0], offs[i][1], false));
		CHECK(crops[i + 5].data() == window(offs[i][0], offs[i][1], true));
	}
	CHECK(crops[0].data() == std::vector<float>({0, 1, 2, 5, 6, 7, 10, 11, 12}));
	CHECK(crops[4].data() == std::vector<float>({6, 7, 8, 11, 12, 13, 16, 17, 18}));
	CHECK_THROWS_AS(ten_crop(x, 6), std::invalid_argument);
	CHECK_THROWS_AS(ten_crop(x, 0), std::invalid_argument);
	CHECK_THROWS_AS(ten_crop(Tensor({5, 5}, 0.0f), 3), TensorShapeError);
}

TEST_CASE("ten_crop_batch stacks ten_crop per sample") {
	Tensor batch = index_image({2, 1, 5, 5});
	auto views = ten_crop_batch(batch, 3);
	REQUIRE(views.size() == 10);
	for (size_t n = 0; n < 2; n++) {
		Tensor sample({1, 5, 5}, std::vector<float>(batch.data().begin() + n * 25, batch.data().begin() + (n + 1) * 25));
		auto expect = ten_crop(sample, 3);
		for (size_t v = 0; v < 10; v++) {
			CHECK(views[v].shape() == std::vector<size_t>({2, 1, 3, 3}));
			CHECK(std::vector<float>(views[v].data().begin() + n * 9, views[v].data().begin() + (n + 1) * 9) == expect[v].data());
		}
	}
	CHECK_THROWS_AS(ten_crop_batch(batch, 6), std::invalid_argument);
	CHECK_THROWS_AS(ten_crop_batch(batch, 0), std::invalid_argument);
}

TEST_CASE("RandomErasing p = 0 leaves the sample unchanged") {
	Tensor x = index_image({1, 33, 33});
	RandomErasing erase(0.0f);
	for (int i = 0; i < 20; i++) CHECK(erase(x).data() == x.data());
}

TEST_CASE("RandomErasing erases one axis-aligned rectangle of bounded area in every channel") {
	Tensor ones({1, 33, 33}, 1.0f);
	manual_seed(8);
	RandomErasing erase(1.0f, 0.25f, 0.25f, 0.3f, 3.3f, -1.0f);
	int erased_draws = 0;
	for (int i = 0; i < 200; i++) {
		Tensor y = erase(ones);
		size_t count = 0, y0 = 33, y1 = 0, x0 = 33, x1 = 0;
		for (size_t r = 0; r < 33; r++)
			for (size_t q = 0; q < 33; q++)
				if (y.data()[r * 33 + q] == -1.0f) {
					count++;
					y0 = std::min(y0, r);
					y1 = std::max(y1, r);
					x0 = std::min(x0, q);
					x1 = std::max(x1, q);
				}
		CHECK(static_cast<double>(count) <= 0.25 * 33 * 33 + 40);
		if (count == 0) continue;
		erased_draws++;
		CHECK(count == (y1 - y0 + 1) * (x1 - x0 + 1));  // the bounding box is fully erased
	}
	CHECK(erased_draws > 100);

	Tensor rgb({3, 7, 7}, 1.0f);
	RandomErasing all(1.0f, 0.3f, 0.3f, 1.0f, 1.0f, 5.0f);
	Tensor z = all(rgb);
	size_t fives = 0;
	for (size_t i = 0; i < 49; i++) {
		CHECK(z.data()[49 + i] == z.data()[i]);
		CHECK(z.data()[98 + i] == z.data()[i]);
		if (z.data()[i] == 5.0f) fives++;
	}
	CHECK(fives == 16);  // area 14.7 -> 4 x 4
}

TEST_CASE("RandomErasing at ratio 1 erases about the requested area and moves the rectangle") {
	Tensor ones({1, 33, 33}, 1.0f);
	manual_seed(11);
	RandomErasing erase(1.0f, 0.25f, 0.25f, 1.0f, 1.0f, -1.0f);
	bool moved_y = false, moved_x = false;
	for (int i = 0; i < 100; i++) {
		Tensor y = erase(ones);
		size_t count = 0, y0 = 33, x0 = 33;
		for (size_t r = 0; r < 33; r++)
			for (size_t q = 0; q < 33; q++)
				if (y.data()[r * 33 + q] == -1.0f) {
					count++;
					y0 = std::min(y0, r);
					x0 = std::min(x0, q);
				}
		CHECK(static_cast<double>(count) <= 0.25 * 1089 + 25);
		CHECK(static_cast<double>(count) >= 0.25 * 1089 - 25);
		if (y0 > 0) moved_y = true;
		if (x0 > 0) moved_x = true;
	}
	CHECK(moved_y);
	CHECK(moved_x);
}

TEST_CASE("RandomErasing draws the same numbers whether or not an attempt fits") {
	// 1x1 samples almost never fit a rectangle (h or w rounds to 0); 33x33 samples fit.
	// If the draw count depended on that, the second call on the same sample would differ.
	Tensor tiny({1, 1, 1}, 1.0f), big = index_image({1, 33, 33});
	manual_seed(12);
	RandomErasing a(1.0f, 0.02f, 0.02f);
	manual_seed(12);
	RandomErasing b(1.0f, 0.02f, 0.02f);
	a(tiny);
	b(big);
	CHECK(a(big).data() == b(big).data());
}

TEST_CASE("RandomErasing handles tiny images, is repeatable and validates arguments") {
	manual_seed(9);
	RandomErasing erase(1.0f, 0.02f, 1.0f, 0.3f, 3.3f, -1.0f);
	for (int i = 0; i < 100; i++) {
		CHECK(erase(Tensor({1, 1, 1}, 1.0f)).size() == 1);
		CHECK(erase(Tensor({1, 7, 7}, 1.0f)).size() == 49);
	}
	Tensor big = index_image({3, 33, 33});
	manual_seed(10);
	Tensor a = RandomErasing(0.7f)(big);
	manual_seed(10);
	Tensor b = RandomErasing(0.7f)(big);
	CHECK(a.data() == b.data());
	CHECK_THROWS_AS(RandomErasing(-0.1f), std::invalid_argument);
	CHECK_THROWS_AS(RandomErasing(1.1f), std::invalid_argument);
	CHECK_THROWS_AS(RandomErasing(0.5f, 0.0f), std::invalid_argument);
	CHECK_THROWS_AS(RandomErasing(0.5f, 0.4f, 0.3f), std::invalid_argument);
	CHECK_THROWS_AS(RandomErasing(0.5f, 0.1f, 1.5f), std::invalid_argument);
	CHECK_THROWS_AS(RandomErasing(0.5f, 0.1f, 0.3f, 0.0f), std::invalid_argument);
	CHECK_THROWS_AS(RandomErasing(0.5f, 0.1f, 0.3f, 2.0f, 1.0f), std::invalid_argument);
	CHECK_THROWS_AS(RandomErasing(std::nanf("")), std::invalid_argument);
	CHECK_THROWS_AS(RandomErasing(0.5f, std::nanf("")), std::invalid_argument);
	CHECK_THROWS_AS(RandomErasing(0.5f, 0.1f, 0.3f, 0.3f, std::nanf("")), std::invalid_argument);
	CHECK_THROWS_AS(RandomErasing(0.5f, 0.1f, 0.3f, 3.0f, 2.0f), std::invalid_argument);
	CHECK_NOTHROW(RandomErasing(0.5f, 0.2f, 0.2f));
	CHECK_THROWS_AS(erase(Tensor({4, 5}, 0.0f)), TensorShapeError);
}
