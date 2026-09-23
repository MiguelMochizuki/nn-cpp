/**
 * pool2d.cpp
 * Author: Miguel Mochizuki Silva
 * Description: 2D max pooling layer
 */

#include "nn/nn/pool2d.hpp"
#include "nn/autograd/ops.hpp"

namespace nn {

MaxPool2d::MaxPool2d(size_t kernel_size, size_t stride) : kernel_size_(kernel_size), stride_(stride) {}

Value MaxPool2d::forward(const Value& input) { return maxpool2d(input, kernel_size_, stride_); }

}  // namespace nn
