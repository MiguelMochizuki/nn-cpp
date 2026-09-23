/**
 * ops.hpp
 * Author: Miguel Mochizuki Silva
 * Description: Differentiable free functions over autograd Values
 */

#pragma once

#include "nn/autograd/value.hpp"

namespace nn {

/* Public, differentiable elementwise addition with broadcasting
 *
 * Parameters:
 * Value a: left operand
 * Value b: right operand
 *
 * Returns Value: a.data() + b.data(); backward accumulates grad into a and b, reduced
 * over any broadcast dimensions
 */
Value add(const Value& a, const Value& b);

/* Public, differentiable elementwise subtraction with broadcasting
 *
 * Parameters:
 * Value a: left operand
 * Value b: right operand
 *
 * Returns Value: a.data() - b.data(); backward accumulates grad into a and -grad into b,
 * each reduced over any broadcast dimensions
 */
Value sub(const Value& a, const Value& b);

/* Public, differentiable elementwise multiplication with broadcasting
 *
 * Parameters:
 * Value a: left operand
 * Value b: right operand
 *
 * Returns Value: a.data() * b.data(); backward accumulates grad_out*b into a and
 * grad_out*a into b, each reduced over any broadcast dimensions
 */
Value mul(const Value& a, const Value& b);

/* Public, differentiable scalar multiplication
 *
 * Parameters:
 * Value a: operand
 * float scalar: multiplier
 *
 * Returns Value: a.data() scaled by scalar; backward accumulates grad_out*scalar into a
 */
Value scale(const Value& a, float scalar);

/* Public, differentiable full reduction to a scalar
 *
 * Parameters:
 * Value x: operand, any shape
 *
 * Returns Value: shape {1}, the sum of every element of x; backward accumulates the
 * single incoming gradient into every element of x's grad
 */
Value sum(const Value& x);

/* Public, differentiable dimension permutation
 *
 * Parameters:
 * Value x: operand
 * vector<size_t> perm: perm[i] is the source dimension that becomes output dimension i
 *
 * Returns Value: x.data() permuted by perm; backward permutes grad_out by the inverse
 * permutation
 */
Value transpose(const Value& x, std::vector<size_t> perm);

/* Public, differentiable reshape
 *
 * Parameters:
 * Value x: operand
 * vector<size_t> new_shape: new dimension sizes; product must equal x.data().size()
 *
 * Returns Value: x.data() reshaped to new_shape; backward reshapes grad_out back to
 * x's original shape
 */
Value reshape(const Value& x, std::vector<size_t> new_shape);

/* Public, differentiable batched matrix multiplication
 *
 * Parameters:
 * Value a: left operand, shape {..., M, K}
 * Value b: right operand, shape {..., K, N}
 *
 * Returns Value: a.data() matmul b.data(); backward computes grad_a = grad_out matmul
 * transpose(b), grad_b = transpose(a) matmul grad_out, each reduced over any broadcast
 * batch dimensions
 */
Value matmul(const Value& a, const Value& b);

/* Public, differentiable rectified linear unit
 *
 * Parameters:
 * Value x: operand
 *
 * Returns Value: max(x, 0) elementwise; backward passes grad_out through where x > 0,
 * zero elsewhere
 */
Value relu(const Value& x);

/* Public, differentiable logistic sigmoid
 *
 * Parameters:
 * Value x: operand
 *
 * Returns Value: 1 / (1 + exp(-x)) elementwise; backward multiplies grad_out by
 * output * (1 - output)
 */
Value sigmoid(const Value& x);

/* Public, differentiable hyperbolic tangent
 *
 * Parameters:
 * Value x: operand
 *
 * Returns Value: tanh(x) elementwise; backward multiplies grad_out by (1 - output^2)
 */
Value tanh(const Value& x);

/* Public, differentiable softmax along one axis
 *
 * Parameters:
 * Value x: operand
 * int axis: dimension to normalize over
 *
 * Returns Value: numerically-stable softmax of x along axis; backward applies the
 * standard softmax Jacobian-vector product, output * (grad_out - sum(grad_out * output, axis))
 */
Value softmax(const Value& x, int axis);

/* Public, differentiable 2D convolution
 *
 * Parameters:
 * Value x: input, shape {N, Cin, H, W}
 * Value weight: kernel, shape {Cout, Cin, kH, kW}
 * Value bias: per-output-channel bias, shape {Cout}
 * int stride: convolution stride, applied to both spatial dimensions
 * int padding: zero-padding applied to both spatial dimensions before convolving
 *
 * Returns Value: shape {N, Cout, H_out, W_out}, where H_out = (H + 2*padding - kH) /
 * stride + 1 and W_out analogously; implemented via im2col + matmul, backward via the
 * transposed matmuls and col2im
 */
Value conv2d(const Value& x, const Value& weight, const Value& bias, int stride, int padding);

/* Public, differentiable 2D max pooling
 *
 * Parameters:
 * Value x: input, shape {N, C, H, W}
 * size_t kernel_size: pooling window side length, applied to both spatial dimensions
 * size_t stride: pooling stride, applied to both spatial dimensions
 *
 * Returns Value: shape {N, C, H_out, W_out}, where H_out = (H - kernel_size) / stride + 1
 * and W_out analogously; backward routes each output element's gradient to the single
 * input element that was its window's maximum, accumulating where windows overlap
 */
Value maxpool2d(const Value& x, size_t kernel_size, size_t stride);

/* Public, differentiable fused log-softmax + negative-log-likelihood loss
 *
 * Parameters:
 * Value logits: raw model output, shape {N, num_classes}
 * Tensor targets: class index per sample stored as a float (e.g. 3.0f for class 3),
 * shape {N}; not a Value, since labels are never differentiated
 *
 * Returns Value: scalar shape {1}, the mean over the batch of -log_softmax(logits)[n,
 * targets[n]]; backward is the fused (softmax(logits) - one_hot(targets)) / N
 */
Value cross_entropy_loss(const Value& logits, const Tensor& targets);

/* Public, differentiable mean squared error
 *
 * Parameters:
 * Value predictions: model output, any shape
 * Value targets: ground truth, same shape as predictions
 *
 * Returns Value: scalar shape {1}, the mean of (predictions - targets)^2 over every element
 */
Value mse_loss(const Value& predictions, const Value& targets);

}  // namespace nn
