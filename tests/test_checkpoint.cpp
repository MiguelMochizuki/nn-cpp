/**
 * test_checkpoint.cpp
 * Author: Miguel Mochizuki Silva
 * Description: Unit tests for save_checkpoint / load_checkpoint
 */

#include "doctest/doctest.h"
#include "nn/checkpoint.hpp"
#include "nn/nn/activation.hpp"
#include "nn/nn/batchnorm.hpp"
#include "nn/nn/linear.hpp"
#include "nn/nn/loss.hpp"
#include "nn/nn/sequential.hpp"
#include "nn/optim/adam.hpp"
#include "nn/optim/adamw.hpp"
#include "nn/optim/sgd.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace nn;

namespace {

// Unique per process: ctest runs each test case as its own process, in parallel with -j
const std::string PATH = "ckpt_test_" + std::to_string(std::random_device{}()) + ".bin";

Sequential make_mlp(size_t hidden = 3) {
	Sequential m;
	m.add(std::make_unique<Linear>(2, hidden));
	m.add(std::make_unique<ReLU>());
	m.add(std::make_unique<Linear>(hidden, 1));
	return m;
}

std::vector<Storage> snapshot(const Module& m) {
	std::vector<Storage> out;
	for (const Value& p : m.parameters()) out.push_back(p.data().data());
	return out;
}

std::vector<uint8_t> read_bytes(const std::string& path) {
	std::ifstream in(path, std::ios::binary);
	return std::vector<uint8_t>(std::istreambuf_iterator<char>(in), {});
}

void write_bytes(const std::string& path, const std::vector<uint8_t>& bytes) {
	std::ofstream out(path, std::ios::binary | std::ios::trunc);
	out.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
}

void put32(std::vector<uint8_t>& b, uint32_t v) {
	for (int i = 0; i < 4; i++) b.push_back((v >> (8 * i)) & 0xFF);
}

void put64(std::vector<uint8_t>& b, uint64_t v) {
	for (int i = 0; i < 8; i++) b.push_back((v >> (8 * i)) & 0xFF);
}

void put_str(std::vector<uint8_t>& b, const std::string& s) { b.insert(b.end(), s.begin(), s.end()); }

std::vector<uint8_t> header(uint32_t count) {
	std::vector<uint8_t> b;
	put_str(b, "NNCK");
	put32(b, 1);
	put32(b, count);
	return b;
}

// Loading `bytes` into a fresh MLP must throw and leave its parameters untouched.
void check_rejected(const std::vector<uint8_t>& bytes) {
	write_bytes(PATH, bytes);
	Sequential m = make_mlp();
	auto before = snapshot(m);
	CHECK_THROWS_AS(load_checkpoint(PATH, m), std::runtime_error);
	CHECK(snapshot(m) == before);
	std::remove(PATH.c_str());
}

void train(Module& model, Optimizer& optimizer, int steps) {
	Value x = Value::leaf(Tensor({4, 2}, std::vector<float>{0, 0, 0, 1, 1, 0, 1, 1}), false);
	Value y = Value::leaf(Tensor({4, 1}, std::vector<float>{0, 1, 1, 0}), false);
	for (int i = 0; i < steps; i++) {
		Value loss = mse_loss(model.forward(x), y);
		optimizer.zero_grad();
		loss.backward();
		optimizer.step();
	}
}

}  // namespace

TEST_CASE("model round trip restores weights and step") {
	Sequential a = make_mlp(), b = make_mlp();
	save_checkpoint(PATH, a, nullptr, 42);
	CHECK(load_checkpoint(PATH, b) == 42);
	CHECK(snapshot(a) == snapshot(b));
	std::remove(PATH.c_str());
}

TEST_CASE("checkpoint bytes match the documented format") {
	Sequential m;
	m.add(std::make_unique<Linear>(1, 1));
	auto params = m.named_parameters();
	params[0].second.data().data() = {1.0f};
	params[1].second.data().data() = {2.0f};
	save_checkpoint(PATH, m);

	std::vector<uint8_t> want = header(3);
	put32(want, 14); put_str(want, "model.0.weight"); want.push_back(0);
	put32(want, 2); put64(want, 1); put64(want, 1); put32(want, 0x3F800000);
	put32(want, 12); put_str(want, "model.0.bias"); want.push_back(0);
	put32(want, 1); put64(want, 1); put32(want, 0x40000000);
	put32(want, 4); put_str(want, "step"); want.push_back(1); put64(want, 0);
	CHECK(read_bytes(PATH) == want);
	std::remove(PATH.c_str());
}

TEST_CASE("zero-parameter model round-trips") {
	Sequential a, b;
	a.add(std::make_unique<ReLU>());
	b.add(std::make_unique<ReLU>());
	save_checkpoint(PATH, a, nullptr, 7);
	CHECK(load_checkpoint(PATH, b) == 7);
	std::remove(PATH.c_str());
}

TEST_CASE("NaN, inf and -0.0 are restored bit-exact") {
	Sequential a, b;
	a.add(std::make_unique<Linear>(2, 2));
	b.add(std::make_unique<Linear>(2, 2));
	const float nan = std::numeric_limits<float>::quiet_NaN(), inf = std::numeric_limits<float>::infinity();
	a.parameters()[0].data().data() = {nan, inf, -inf, -0.0f};
	save_checkpoint(PATH, a);
	load_checkpoint(PATH, b);
	const auto& got = b.parameters()[0].data().data();
	const auto& want = a.parameters()[0].data().data();
	for (size_t i = 0; i < 4; i++) CHECK(std::memcmp(&got[i], &want[i], sizeof(float)) == 0);
	std::remove(PATH.c_str());
}

TEST_CASE("a file with extra or missing layers is rejected and the model is unchanged") {
	Sequential small;
	small.add(std::make_unique<Linear>(2, 3));
	Sequential big = make_mlp();

	save_checkpoint(PATH, big);
	auto before = snapshot(small);
	CHECK_THROWS_WITH_AS(load_checkpoint(PATH, small), doctest::Contains("model.2.weight"), std::runtime_error);
	CHECK(snapshot(small) == before);

	save_checkpoint(PATH, small);
	before = snapshot(big);
	CHECK_THROWS_WITH_AS(load_checkpoint(PATH, big), doctest::Contains("model.2.weight"), std::runtime_error);
	CHECK(snapshot(big) == before);
	std::remove(PATH.c_str());
}

TEST_CASE("a wrong shape is rejected with the entry name") {
	Sequential a = make_mlp(3), b = make_mlp(4);
	save_checkpoint(PATH, a);
	auto before = snapshot(b);
	CHECK_THROWS_WITH_AS(load_checkpoint(PATH, b), doctest::Contains("model.0.weight"), std::runtime_error);
	CHECK(snapshot(b) == before);
	std::remove(PATH.c_str());
}

TEST_CASE("every truncation of a valid file is rejected") {
	Sequential a = make_mlp();
	save_checkpoint(PATH, a);
	std::vector<uint8_t> bytes = read_bytes(PATH);
	std::remove(PATH.c_str());
	for (size_t n = 0; n < bytes.size(); n++) check_rejected(std::vector<uint8_t>(bytes.begin(), bytes.begin() + n));
}

TEST_CASE("bad magic and bad version are rejected") {
	Sequential a = make_mlp();
	save_checkpoint(PATH, a);
	std::vector<uint8_t> bytes = read_bytes(PATH);
	std::remove(PATH.c_str());
	std::vector<uint8_t> magic = bytes;
	magic[0] = 'X';
	check_rejected(magic);
	std::vector<uint8_t> version = bytes;
	version[4] = 2;
	check_rejected(version);
}

TEST_CASE("duplicate entry names are rejected") {
	std::vector<uint8_t> b = header(2);
	for (int i = 0; i < 2; i++) {
		put32(b, 4); put_str(b, "step"); b.push_back(1); put64(b, 0);
	}
	check_rejected(b);
}

TEST_CASE("headers that claim absurd sizes are rejected without allocating them") {
	std::vector<uint8_t> name = header(1);
	put32(name, 0x7FFFFFFF);
	check_rejected(name);

	std::vector<uint8_t> ndim = header(1);
	put32(ndim, 14); put_str(ndim, "model.0.weight"); ndim.push_back(0); put32(ndim, 255);
	check_rejected(ndim);

	std::vector<uint8_t> elems = header(1);
	put32(elems, 14); put_str(elems, "model.0.weight"); elems.push_back(0);
	put32(elems, 2); put64(elems, uint64_t(1) << 40); put64(elems, uint64_t(1) << 40);
	check_rejected(elems);

	std::vector<uint8_t> count = header(0xFFFFFFFF);
	check_rejected(count);
}

TEST_CASE("a failed save throws and keeps the previous checkpoint") {
	Sequential a = make_mlp(), b = make_mlp();
	save_checkpoint(PATH, a, nullptr, 5);
	std::vector<uint8_t> original = read_bytes(PATH);

	const std::string tmp = PATH + ".tmp";
	REQUIRE(std::filesystem::create_directory(tmp));  // a directory squats on the temp name
	CHECK_THROWS_AS(save_checkpoint(PATH, b, nullptr, 9), std::runtime_error);
	CHECK(read_bytes(PATH) == original);
	CHECK(load_checkpoint(PATH, b) == 5);
	std::filesystem::remove_all(tmp);
	std::remove(PATH.c_str());
}

TEST_CASE("resume equivalence: a loaded model and optimizer continue exactly like the original") {
	Sequential a = make_mlp();
	SGD opt_a(a.parameters(), 0.1f, 0.9f);
	train(a, opt_a, 5);
	save_checkpoint(PATH, a, &opt_a, 5);
	train(a, opt_a, 5);

	Sequential b = make_mlp();
	SGD opt_b(b.parameters(), 0.7f, 0.0f);
	CHECK(load_checkpoint(PATH, b, &opt_b) == 5);
	train(b, opt_b, 5);
	CHECK(snapshot(a) == snapshot(b));
	std::remove(PATH.c_str());
}

TEST_CASE("inference-only load ignores optimizer entries") {
	Sequential a = make_mlp(), b = make_mlp();
	SGD opt(a.parameters(), 0.1f, 0.9f);
	train(a, opt, 3);
	save_checkpoint(PATH, a, &opt);
	load_checkpoint(PATH, b);
	CHECK(snapshot(a) == snapshot(b));
	std::remove(PATH.c_str());
}

TEST_CASE("loading with an optimizer from a model-only file throws and changes nothing") {
	Sequential a = make_mlp(), b = make_mlp();
	save_checkpoint(PATH, a);
	SGD opt(b.parameters(), 0.1f, 0.9f);
	auto before = snapshot(b);
	CHECK_THROWS_WITH_AS(load_checkpoint(PATH, b, &opt), doctest::Contains("optim"), std::runtime_error);
	CHECK(snapshot(b) == before);
	std::remove(PATH.c_str());
}

TEST_CASE("momentum 0 in the file overrides momentum 0.9 on the loading optimizer") {
	Sequential a = make_mlp();
	SGD opt_a(a.parameters(), 0.1f, 0.0f);
	save_checkpoint(PATH, a, &opt_a);

	Sequential c = make_mlp();
	SGD opt_c(c.parameters(), 0.5f, 0.9f);
	load_checkpoint(PATH, c, &opt_c);
	for (const Value& q : c.parameters()) q.grad() = Tensor(q.data().shape(), 1.0f);
	auto before = snapshot(c);
	opt_c.step();
	auto after = snapshot(c);
	for (size_t i = 0; i < before.size(); i++)
		for (size_t j = 0; j < before[i].size(); j++) CHECK(after[i][j] == doctest::Approx(before[i][j] - 0.1f));
	std::remove(PATH.c_str());
}

TEST_CASE("resume equivalence holds for Adam") {
	Sequential a = make_mlp();
	Adam opt_a(a.parameters(), 0.01f);
	train(a, opt_a, 5);
	save_checkpoint(PATH, a, &opt_a, 5);
	train(a, opt_a, 5);

	Sequential b = make_mlp();
	Adam opt_b(b.parameters(), 0.5f);
	load_checkpoint(PATH, b, &opt_b);
	train(b, opt_b, 5);
	CHECK(snapshot(a) == snapshot(b));
	std::remove(PATH.c_str());
}

TEST_CASE("checkpoint round trip restores the learning rate for SGD, Adam and AdamW") {
	Sequential a = make_mlp();
	SGD s(a.parameters(), 0.1f, 0.9f);
	s.set_lr(0.0125f);
	save_checkpoint(PATH, a, &s);
	SGD s2(a.parameters(), 0.5f, 0.9f);
	load_checkpoint(PATH, a, &s2);
	CHECK(s2.lr() == 0.0125f);

	AdamW w(a.parameters(), 0.01f);
	w.set_lr(0.003f);
	save_checkpoint(PATH, a, &w);
	AdamW w2(a.parameters(), 0.5f);
	load_checkpoint(PATH, a, &w2);
	CHECK(w2.lr() == 0.003f);
	std::remove(PATH.c_str());
}

TEST_CASE("an optimizer snapshot without lr loads and keeps the current lr") {
	Sequential a = make_mlp();
	Adam opt(a.parameters(), 0.01f);
	OptimizerState st = opt.state();
	st.erase(std::remove_if(st.begin(), st.end(), [](const auto& e) { return e.first == "lr"; }), st.end());
	Adam other(a.parameters(), 0.25f);
	other.load_state(st);
	CHECK(other.lr() == 0.25f);
	SGD sgd(a.parameters(), 0.1f, 0.5f);
	OptimizerState ss = sgd.state();
	ss.erase(std::remove_if(ss.begin(), ss.end(), [](const auto& e) { return e.first == "lr"; }), ss.end());
	SGD sgd2(a.parameters(), 0.7f, 0.0f);
	sgd2.load_state(ss);
	CHECK(sgd2.lr() == 0.7f);
	ss.emplace_back("bogus", Tensor({1}, 1.0f));
	CHECK_THROWS_AS(sgd2.load_state(ss), std::runtime_error);
}

namespace {

// Test-only module with one parameter and one non-trainable buffer
class WithBuffer : public Module {
public:
	explicit WithBuffer(const std::string& buffer_name = "stat")
	    : w_(Value::leaf(Tensor({2}, std::vector<float>{1.0f, 2.0f}), true)),
	      stat_({3}, std::vector<float>{7.0f, 8.0f, 9.0f}),
	      buffer_name_(buffer_name) {}
	Value forward(const Value& input) override { return input; }
	std::vector<Value> parameters() const override { return {w_}; }
	std::vector<std::pair<std::string, Value>> named_parameters() const override { return {{"w", w_}}; }
	std::vector<std::pair<std::string, Tensor*>> named_buffers() const override { return {{buffer_name_, &stat_}}; }
	Tensor& stat() { return stat_; }

private:
	Value w_;
	mutable Tensor stat_;
	std::string buffer_name_;
};

}  // namespace

TEST_CASE("buffers round trip next to parameters") {
	WithBuffer a, b;
	a.stat().data() = {1.5f, 2.5f, 3.5f};
	a.parameters()[0].data().data() = {-1.0f, -2.0f};
	save_checkpoint(PATH, a, nullptr, 3);
	CHECK(load_checkpoint(PATH, b) == 3);
	CHECK(b.stat().data() == std::vector<float>{1.5f, 2.5f, 3.5f});
	CHECK(b.parameters()[0].data().data() == std::vector<float>{-1.0f, -2.0f});
	std::remove(PATH.c_str());
}

TEST_CASE("buffers are saved after the parameters and prefixed in Sequential") {
	Sequential seq;
	seq.add(std::make_unique<WithBuffer>());
	save_checkpoint(PATH, seq);
	std::vector<uint8_t> bytes = read_bytes(PATH);
	std::string text(bytes.begin(), bytes.end());
	size_t w = text.find("model.0.w");
	size_t s = text.find("model.0.stat");
	CHECK(w != std::string::npos);
	CHECK(s != std::string::npos);
	CHECK(w < s);
	Sequential other;
	other.add(std::make_unique<WithBuffer>());
	CHECK_NOTHROW(load_checkpoint(PATH, other));
	std::remove(PATH.c_str());
}

TEST_CASE("a buffer-less file does not load into a model with buffers and changes nothing") {
	Sequential plain;
	plain.add(std::make_unique<Linear>(2, 1));
	save_checkpoint(PATH, plain);
	WithBuffer m;
	Storage w_before = m.parameters()[0].data().data();
	try {
		load_checkpoint(PATH, m);
		FAIL("expected std::runtime_error");
	} catch (const std::runtime_error& e) {
		CHECK(std::string(e.what()).find("model.stat") != std::string::npos);
	}
	CHECK(m.stat().data() == std::vector<float>{7.0f, 8.0f, 9.0f});
	CHECK(m.parameters()[0].data().data() == w_before);
	std::remove(PATH.c_str());
}

TEST_CASE("a file with buffers does not load into a buffer-less model") {
	WithBuffer a;
	save_checkpoint(PATH, a);
	Sequential plain;
	plain.add(std::make_unique<Linear>(2, 1));
	CHECK_THROWS_AS(load_checkpoint(PATH, plain), std::runtime_error);
	std::remove(PATH.c_str());
}

TEST_CASE("a buffer name colliding with a parameter name throws") {
	WithBuffer clash("w");
	CHECK_THROWS_AS(save_checkpoint(PATH, clash), std::runtime_error);
	// A real file, so the load reaches the collision check instead of failing on a missing file
	WithBuffer fine;
	save_checkpoint(PATH, fine);
	CHECK_THROWS_AS(load_checkpoint(PATH, clash), std::runtime_error);
	std::remove(PATH.c_str());
}

TEST_CASE("a v1.0.0 checkpoint loads and a parameter-only save is byte-identical to it") {
	const std::string golden = std::string(NN_TEST_DATA_DIR) + "/linear_v1.ckpt";
	Sequential m;
	m.add(std::make_unique<Linear>(3, 2));
	CHECK(load_checkpoint(golden, m) == 7);
	auto p = m.named_parameters();
	CHECK(p[0].second.data().data() == std::vector<float>{0.5f, -1.25f, 2.0f, 3.0f, 0.75f, -0.5f});
	CHECK(p[1].second.data().data() == std::vector<float>{0.125f, -2.0f});
	save_checkpoint(PATH, m, nullptr, 7);
	CHECK(read_bytes(PATH) == read_bytes(golden));
	std::remove(PATH.c_str());
}

TEST_CASE("BatchNorm2d running statistics survive a checkpoint round trip inside Sequential") {
	auto make = [] {
		Sequential m;
		m.add(std::make_unique<BatchNorm2d>(3));
		return m;
	};
	Sequential a = make(), b = make();
	for (int step = 0; step < 3; step++) {
		Tensor x = Tensor::random({4, 3, 2, 2}, -1.0f + static_cast<float>(step), 2.0f);
		a.forward(Value::leaf(x));
	}
	auto bufs = a.named_buffers();
	CHECK(bufs[0].second->data() != std::vector<float>(3, 0.0f));
	save_checkpoint(PATH, a);
	load_checkpoint(PATH, b);
	a.eval();
	b.eval();
	Tensor probe = Tensor::random({2, 3, 2, 2}, -1.0f, 1.0f);
	CHECK(a.forward(Value::leaf(probe)).data().data() == b.forward(Value::leaf(probe)).data().data());
	CHECK(b.named_buffers()[1].first == "0.running_var");
	CHECK(b.named_buffers()[1].second->data() == a.named_buffers()[1].second->data());
	std::remove(PATH.c_str());
}
