/**
 * test_checkpoint.cpp
 * Author: Miguel Mochizuki Silva
 * Description: Unit tests for save_checkpoint / load_checkpoint
 */

#include "doctest/doctest.h"
#include "nn/checkpoint.hpp"
#include "nn/nn/activation.hpp"
#include "nn/nn/linear.hpp"
#include "nn/nn/loss.hpp"
#include "nn/nn/sequential.hpp"
#include "nn/optim/sgd.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace nn;

namespace {

const std::string PATH = "ckpt_test.bin";

Sequential make_mlp(size_t hidden = 3) {
	Sequential m;
	m.add(std::make_unique<Linear>(2, hidden));
	m.add(std::make_unique<ReLU>());
	m.add(std::make_unique<Linear>(hidden, 1));
	return m;
}

std::vector<std::vector<float>> snapshot(const Module& m) {
	std::vector<std::vector<float>> out;
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
