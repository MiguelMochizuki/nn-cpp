/**
 * pool2d.cpp
 * Author: Miguel Mochizuki Silva
 * Description: 2D max, average and global average pooling layers
 */

#include "nn/nn/pool2d.hpp"
#include "nn/autograd/ops.hpp"

namespace nn {

MaxPool2d::MaxPool2d(size_t kernel_size, size_t stride) : kernel_size_(kernel_size), stride_(stride) {}

Value MaxPool2d::forward(const Value& input) { return maxpool2d(input, kernel_size_, stride_); }

AvgPool2d::AvgPool2d(size_t kernel_size, size_t stride) : kernel_size_(kernel_size), stride_(stride) {}

Value AvgPool2d::forward(const Value& input) { return avgpool2d(input, kernel_size_, stride_); }

Value GlobalAvgPool2d::forward(const Value& input) {
	if (input.data().ndim() != 4) throw TensorShapeError();
	return mean(mean(input, 3), 2);
}

}  // namespace nn
