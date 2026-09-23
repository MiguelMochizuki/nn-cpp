/**
 * idx_dataset.cpp
 * Author: Miguel Mochizuki Silva
 * Description: Parser for the IDX ubyte file format used by MNIST
 */

#include "idx_dataset.hpp"

#include <fstream>
#include <cstdint>

namespace nn {

namespace {

uint32_t read_u32_be(std::ifstream& file) {
	unsigned char bytes[4];
	file.read(reinterpret_cast<char*>(bytes), 4);
	return (static_cast<uint32_t>(bytes[0]) << 24) | (static_cast<uint32_t>(bytes[1]) << 16) |
	       (static_cast<uint32_t>(bytes[2]) << 8) | static_cast<uint32_t>(bytes[3]);
}

}  // namespace

Tensor load_idx_ubyte(const std::string& path) {
	std::ifstream file(path, std::ios::binary);
	if (!file) {
		throw IdxParseError();
	}

	uint32_t magic = read_u32_be(file);
	uint8_t ndim = static_cast<uint8_t>(magic & 0xFF);

	std::vector<size_t> shape(ndim);
	size_t total = 1;
	for (uint8_t i = 0; i < ndim; i++) {
		shape[i] = read_u32_be(file);
		total *= shape[i];
	}

	std::vector<unsigned char> raw(total);
	file.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(total));

	std::vector<float> data(total);
	for (size_t i = 0; i < total; i++) {
		data[i] = static_cast<float>(raw[i]);
	}

	return Tensor(shape, data);
}

}  // namespace nn
