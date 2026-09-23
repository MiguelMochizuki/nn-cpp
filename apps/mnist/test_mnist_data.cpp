/**
 * test_mnist_data.cpp
 * Author: Miguel Mochizuki Silva
 * Description: Unit tests for the IDX parser and MnistDataset, against small fixture files
 */

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"
#include "mnist_dataset.hpp"

#include <fstream>
#include <cstdio>
#include <cstdint>

using namespace nn;

namespace {

void write_u32_be(std::ofstream& file, uint32_t v) {
	unsigned char bytes[4] = {
		static_cast<unsigned char>((v >> 24) & 0xFF),
		static_cast<unsigned char>((v >> 16) & 0xFF),
		static_cast<unsigned char>((v >> 8) & 0xFF),
		static_cast<unsigned char>(v & 0xFF),
	};
	file.write(reinterpret_cast<char*>(bytes), 4);
}

void write_images_fixture(const std::string& path) {
	std::ofstream file(path, std::ios::binary);
	unsigned char magic[4] = {0x00, 0x00, 0x08, 0x03};
	file.write(reinterpret_cast<char*>(magic), 4);

	uint32_t image_count = 2;
	uint32_t rows = 2;
	uint32_t cols = 2;
	write_u32_be(file, image_count);
	write_u32_be(file, rows);
	write_u32_be(file, cols);

	unsigned char pixels[8] = {1, 2, 3, 4, 5, 6, 7, 8};
	file.write(reinterpret_cast<char*>(pixels), 8);
}

void write_labels_fixture(const std::string& path) {
	std::ofstream file(path, std::ios::binary);
	unsigned char magic[4] = {0x00, 0x00, 0x08, 0x01};
	file.write(reinterpret_cast<char*>(magic), 4);

	uint32_t label_count = 2;
	write_u32_be(file, label_count);

	unsigned char labels[2] = {0, 1};
	file.write(reinterpret_cast<char*>(labels), 2);
}

}  // namespace

TEST_CASE("load_idx_ubyte parses shape and big-endian header correctly") {
	write_images_fixture("fixture_images.idx");
	Tensor t = load_idx_ubyte("fixture_images.idx");
	CHECK(t.shape() == std::vector<size_t>{2, 2, 2});
	CHECK(t.get({0, 0, 0}) == 1.0f);
	CHECK(t.get({1, 1, 1}) == 8.0f);
	std::remove("fixture_images.idx");
}

TEST_CASE("load_idx_ubyte parses a 1D label file") {
	write_labels_fixture("fixture_labels.idx");
	Tensor t = load_idx_ubyte("fixture_labels.idx");
	CHECK(t.shape() == std::vector<size_t>{2});
	CHECK(t.get({0}) == 0.0f);
	CHECK(t.get({1}) == 1.0f);
	std::remove("fixture_labels.idx");
}

TEST_CASE("MnistDataset pairs images with labels and adds a channel dimension") {
	write_images_fixture("fixture_images.idx");
	write_labels_fixture("fixture_labels.idx");
	MnistDataset dataset = MnistDataset::load("fixture_images.idx", "fixture_labels.idx");

	CHECK(dataset.size() == 2);
	auto [image0, label0] = dataset.get(0);
	CHECK(image0.shape() == std::vector<size_t>{1, 2, 2});
	CHECK(image0.get({0, 0, 0}) == 1.0f);
	CHECK(label0.shape() == std::vector<size_t>{});
	CHECK(label0.get({}) == 0.0f);

	auto [image1, label1] = dataset.get(1);
	CHECK(image1.get({0, 1, 1}) == 8.0f);
	CHECK(label1.get({}) == 1.0f);

	std::remove("fixture_images.idx");
	std::remove("fixture_labels.idx");
}

TEST_CASE("MnistDataset::get throws for an out-of-range index") {
	write_images_fixture("fixture_images.idx");
	write_labels_fixture("fixture_labels.idx");
	MnistDataset dataset = MnistDataset::load("fixture_images.idx", "fixture_labels.idx");
	CHECK_THROWS_AS(dataset.get(2), TensorIndexError);
	std::remove("fixture_images.idx");
	std::remove("fixture_labels.idx");
}

TEST_CASE("MnistDataset::load throws when image and label counts differ") {
	write_images_fixture("fixture_images.idx");

	std::ofstream file("fixture_labels_mismatch.idx", std::ios::binary);
	unsigned char magic[4] = {0x00, 0x00, 0x08, 0x01};
	file.write(reinterpret_cast<char*>(magic), 4);
	write_u32_be(file, 1);
	unsigned char labels[1] = {0};
	file.write(reinterpret_cast<char*>(labels), 1);
	file.close();

	CHECK_THROWS_AS(MnistDataset::load("fixture_images.idx", "fixture_labels_mismatch.idx"), IdxParseError);

	std::remove("fixture_images.idx");
	std::remove("fixture_labels_mismatch.idx");
}

TEST_CASE("load_idx_ubyte throws on bad magic bytes") {
	std::ofstream file("bad_magic.idx", std::ios::binary);
	unsigned char magic[4] = {0x01, 0x02, 0x08, 0x03};
	file.write(reinterpret_cast<char*>(magic), 4);
	file.close();

	CHECK_THROWS_AS(load_idx_ubyte("bad_magic.idx"), IdxParseError);
	std::remove("bad_magic.idx");
}

TEST_CASE("load_idx_ubyte throws instead of silently zero-filling a truncated payload") {
	std::ofstream file("truncated.idx", std::ios::binary);
	unsigned char magic[4] = {0x00, 0x00, 0x08, 0x03};
	file.write(reinterpret_cast<char*>(magic), 4);
	write_u32_be(file, 2);
	write_u32_be(file, 2);
	write_u32_be(file, 2);
	unsigned char pixels[3] = {1, 2, 3};
	file.write(reinterpret_cast<char*>(pixels), 3);
	file.close();

	CHECK_THROWS_AS(load_idx_ubyte("truncated.idx"), IdxParseError);
	std::remove("truncated.idx");
}

TEST_CASE("load_idx_ubyte throws on a file truncated inside the dimension header") {
	std::ofstream file("truncated_header.idx", std::ios::binary);
	unsigned char magic[4] = {0x00, 0x00, 0x08, 0x03};
	file.write(reinterpret_cast<char*>(magic), 4);
	write_u32_be(file, 2);
	file.close();

	CHECK_THROWS_AS(load_idx_ubyte("truncated_header.idx"), IdxParseError);
	std::remove("truncated_header.idx");
}
