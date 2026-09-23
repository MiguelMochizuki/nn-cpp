/**
 * pool2d.hpp
 * Author: Miguel Mochizuki Silva
 * Description: 2D max pooling layer
 */

#pragma once

#include "nn/nn/module.hpp"

namespace nn {

/*
 * 2D max pooling layer, implemented via the autograd maxpool2d op. No parameters.
 *
 * Attributes:
 * size_t kernel_size_: pooling window side length
 * size_t stride_: pooling stride
 */
class MaxPool2d : public Module {
public:
	/* Public, construct a pooling layer
	 *
	 * Parameters:
	 * size_t kernel_size: pooling window side length
	 * size_t stride: pooling stride
	 *
	 * Returns MaxPool2d: none (constructor)
	 */
	MaxPool2d(size_t kernel_size, size_t stride);

	Value forward(const Value& input) override;

private:
	size_t kernel_size_;
	size_t stride_;
};

}  // namespace nn
