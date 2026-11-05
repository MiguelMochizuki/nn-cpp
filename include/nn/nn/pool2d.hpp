/**
 * pool2d.hpp
 * Author: Miguel Mochizuki Silva
 * Description: 2D max, average and global average pooling layers
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

	/* Public, max-pool each channel over kernel_size windows
	 *
	 * Parameters:
	 * Value input: input tensor wrapped in the autograd graph
	 *
	 * Returns Value: shape {N, C, H_out, W_out}
	 */
	Value forward(const Value& input) override;

private:
	size_t kernel_size_;
	size_t stride_;
};

/*
 * 2D average pooling layer, implemented via the autograd avgpool2d op. No parameters.
 *
 * Attributes:
 * size_t kernel_size_: pooling window side length
 * size_t stride_: pooling stride
 */
class AvgPool2d : public Module {
public:
	/* Public, construct a pooling layer
	 *
	 * Parameters:
	 * size_t kernel_size: pooling window side length
	 * size_t stride: pooling stride
	 *
	 * Returns AvgPool2d: none (constructor)
	 */
	AvgPool2d(size_t kernel_size, size_t stride);

	/* Public, average each channel over kernel_size windows
	 *
	 * Parameters:
	 * Value input: input tensor wrapped in the autograd graph
	 *
	 * Returns Value: shape {N, C, H_out, W_out}
	 */
	Value forward(const Value& input) override;

private:
	size_t kernel_size_;
	size_t stride_;
};

/*
 * Global average pooling: the mean over each channel's whole spatial extent. No parameters.
 */
class GlobalAvgPool2d : public Module {
public:
	/* Public, average each channel over H and W
	 *
	 * Parameters:
	 * Value input: shape {N, C, H, W}; other ranks throw TensorShapeError
	 *
	 * Returns Value: shape {N, C}
	 */
	Value forward(const Value& input) override;
};

}  // namespace nn
