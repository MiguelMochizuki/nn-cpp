/**
 * idx_dataset.hpp
 * Author: Miguel Mochizuki Silva
 * Description: Parser for the IDX ubyte file format used by MNIST
 */

#pragma once

#include "nn/tensor.hpp"

#include <string>
#include <exception>

namespace nn {

/*
 * Thrown when an IDX file can't be opened or read.
 *
 * Attributes: none
 */
class IdxParseError : public std::exception {
public:
	const char* what() const noexcept override {
		return "Could not read IDX file.";
	}
};

/* Public, parse one IDX ubyte file into a Tensor
 *
 * Parameters:
 * string path: filesystem path to an IDX file (image or label set)
 *
 * Returns Tensor: values cast from raw bytes (0..255), shape read from the file's own
 * big-endian header; throws IdxParseError if the file can't be opened
 */
Tensor load_idx_ubyte(const std::string& path);

}  // namespace nn
