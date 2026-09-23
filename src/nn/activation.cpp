/**
 * activation.cpp
 * Author: Miguel Mochizuki Silva
 * Description: Elementwise activation layers
 */

#include "nn/nn/activation.hpp"
#include "nn/autograd/ops.hpp"

namespace nn {

Value ReLU::forward(const Value& input) { return relu(input); }
Value Sigmoid::forward(const Value& input) { return sigmoid(input); }
Value Tanh::forward(const Value& input) { return tanh(input); }

Softmax::Softmax(int axis) : axis_(axis) {}
Value Softmax::forward(const Value& input) { return softmax(input, axis_); }

}  // namespace nn
