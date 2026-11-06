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

/* Public, differentiable sum along one axis
 *
 * Parameters:
 * Value x: operand, any shape
 * int axis: dimension to reduce; negative counts from the end; out of range throws
 * bool keepdim: keep the reduced axis with size 1 instead of dropping it
 *
 * Returns Value: the sum over axis. Dropping the only axis of a 1-D tensor gives shape {1}.
 * Backward broadcasts grad_out over the reduced axis. Results do not depend on thread count
 */
Value sum(const Value& x, int axis, bool keepdim = false);

/* Public, differentiable mean along one axis
 *
 * Parameters:
 * Value x: operand, any shape
 * int axis: dimension to reduce; negative counts from the end; out of range throws
 * bool keepdim: keep the reduced axis with size 1 instead of dropping it
 *
 * Returns Value: the sum over axis divided by the axis length. Dropping the only axis of a
 * 1-D tensor gives shape {1}. Backward broadcasts grad_out / length over the reduced axis
 */
Value mean(const Value& x, int axis, bool keepdim = false);

/* Public, differentiable reciprocal square root
 *
 * Parameters:
 * Value x: operand; undefined (inf or NaN) for x <= 0
 *
 * Returns Value: 1 / sqrt(x) elementwise; backward multiplies grad_out by -0.5 * x^-1.5
 */
Value rsqrt(const Value& x);

/* Public, differentiable elementwise exponential
 *
 * Parameters:
 * Value x: operand
 *
 * Returns Value: e^x elementwise (overflows to inf per IEEE); backward multiplies grad_out by e^x
 */
Value exp(const Value& x);

/* Public, differentiable elementwise natural logarithm
 *
 * Parameters:
 * Value x: operand; follows IEEE for x <= 0 (-inf at 0, NaN below), nothing is thrown
 *
 * Returns Value: ln(x) elementwise; backward multiplies grad_out by 1 / x
 */
Value log(const Value& x);

/* Public, differentiable elementwise square root
 *
 * Parameters:
 * Value x: operand; follows IEEE for x < 0 (NaN), nothing is thrown
 *
 * Returns Value: sqrt(x) elementwise; backward multiplies grad_out by 0.5 / sqrt(x)
 */
Value sqrt(const Value& x);

/* Public, differentiable elementwise division with broadcasting
 *
 * Parameters:
 * Value a: numerator
 * Value b: denominator; zero follows IEEE (inf or NaN)
 *
 * Returns Value: a.data() / b.data(); backward accumulates grad_out / b into a and
 * -grad_out * a / b^2 into b, each reduced over any broadcast dimensions
 */
Value div(const Value& a, const Value& b);

/* Public, differentiable Gaussian error linear unit (tanh approximation)
 *
 * Parameters:
 * Value x: operand
 *
 * Returns Value: 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3))) elementwise; backward
 * multiplies grad_out by the analytic derivative of that expression
 */
Value gelu(const Value& x);

/* Public, differentiable masked fill
 *
 * Parameters:
 * Value x: operand
 * Tensor mask: numpy-broadcastable to x.shape() exactly (it may not enlarge x); nonzero means replace
 * float value: replacement for masked positions
 *
 * Returns Value: x with masked positions set to value; backward passes grad_out where the mask is
 * zero and 0 where it is set. Throws TensorShapeError if mask does not broadcast to x.shape()
 */
Value masked_fill(const Value& x, const Tensor& mask, float value);

/* Public, differentiable half-open range along one axis
 *
 * Parameters:
 * Value x: operand
 * int axis: dimension to cut, negative counts from the end as in sum
 * size_t start: first index kept
 * size_t end: one past the last index kept
 *
 * Returns Value: the sub-tensor; backward scatters grad_out into the covered region. Throws
 * TensorIndexError for a bad axis and TensorShapeError if start >= end or end > size
 */
Value slice(const Value& x, int axis, size_t start, size_t end);

/* Public, differentiable concatenation along one axis
 *
 * Parameters:
 * vector<Value> xs: inputs, equal shapes except on axis
 * int axis: dimension to join, negative counts from the end
 *
 * Returns Value: the joined tensor; backward slices grad_out back to each input. Throws
 * TensorShapeError for an empty list or mismatched shapes
 */
Value concat(const std::vector<Value>& xs, int axis);

/* Public, differentiable equal split along one axis
 *
 * Parameters:
 * Value x: operand
 * int axis: dimension to split, negative counts from the end
 * size_t parts: number of slices; must divide the axis size
 *
 * Returns vector<Value>: parts equal slices in order. Throws TensorShapeError if parts is 0 or
 * does not divide the axis size
 */
std::vector<Value> split(const Value& x, int axis, size_t parts);

/* Public, differentiable row lookup into an embedding table
 *
 * Parameters:
 * Value weight: table of shape {V, D}
 * Tensor ids: any shape, float-valued integers in [0, V)
 *
 * Returns Value: shape ids.shape() + {D}; backward scatter-adds each output row's gradient into the
 * weight row of its id, serially, so repeated ids accumulate deterministically. Throws
 * TensorShapeError if weight is not 2-D or an id is non-integral, negative, NaN or >= V
 */
Value embedding(const Value& weight, const Tensor& ids);

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

/* Public, differentiable matrix multiplication against a transposed right operand
 *
 * Parameters:
 * Value a: left operand, shape {..., M, K}
 * Value b: right operand, shape {..., N, K}
 *
 * Returns Value: a.data() matmul transpose(b.data()), shape {..., M, N}, without
 * materializing the transpose; backward computes grad_a = grad_out matmul b,
 * grad_b = transpose(grad_out) matmul a, each reduced over any broadcast batch dimensions
 */
Value matmul_bt(const Value& a, const Value& b);

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

/* Public, differentiable 2D convolution with dilation and groups
 *
 * Parameters:
 * Value x: input, shape {N, Cin, H, W}
 * Value weight: kernel, shape {Cout, Cin/groups, kH, kW}; kH and kW may differ
 * Value bias: per-output-channel bias, shape {Cout}
 * int stride: convolution stride, applied to both spatial dimensions
 * int padding: zero-padding applied to both spatial dimensions before convolving
 * int dilation: spacing between kernel taps, at least 1
 * int groups: number of channel groups; must divide Cin and Cout (groups == Cin is depthwise)
 *
 * Returns Value: shape {N, Cout, H_out, W_out}, where H_out = (H + 2*padding - dilation*(kH-1) - 1) /
 * stride + 1 and W_out analogously. Throws TensorShapeError on a bad groups/dilation/stride value,
 * a channel mismatch or an effective kernel larger than the padded input. With dilation == 1 and
 * groups == 1 it calls the 5-argument overload, so results are bitwise identical to it; otherwise
 * it runs a per-group im2col + sgemm path, batch-parallel with results independent of thread count
 */
Value conv2d(const Value& x, const Value& weight, const Value& bias, int stride, int padding, int dilation,
	int groups);

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

/* Public, differentiable 2D average pooling
 *
 * Parameters:
 * Value x: input, shape {N, C, H, W}
 * size_t kernel_size: pooling window side length, applied to both spatial dimensions; larger
 * than H or W (or zero) throws TensorShapeError
 * size_t stride: pooling stride, applied to both spatial dimensions; zero throws
 *
 * Returns Value: shape {N, C, H_out, W_out}, same size rule as maxpool2d; each element is the
 * mean of its window. Backward adds grad_out / kernel_size^2 to every window element,
 * accumulating where windows overlap. Results do not depend on thread count
 */
Value avgpool2d(const Value& x, size_t kernel_size, size_t stride);

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

/* Public, differentiable class-weighted, label-smoothed cross entropy
 *
 * Parameters:
 * Value logits: raw model output, shape {N, num_classes}
 * Tensor targets: class index per sample stored as a float, shape {N}
 * Tensor class_weights: per-class weight, shape {num_classes}; an empty tensor is not allowed
 * (use the 2-argument overload for the unweighted loss)
 * float label_smoothing: s in [0, 1), the share of the target mass spread uniformly over all classes
 *
 * Returns Value: scalar shape {1}, sum_n w[y_n] * l_n / sum_n w[y_n] with
 * l_n = (1-s) * (-log p[y_n]) + s * mean_c(-log p[c]); backward is fused,
 * w[y_n] / sum(w[y]) * (softmax - (1-s) * one_hot - s / num_classes). Throws TensorShapeError on a
 * shape mismatch, std::invalid_argument on a target outside [0, num_classes), a label_smoothing
 * outside [0, 1) or a zero total weight
 */
Value cross_entropy_loss(const Value& logits, const Tensor& targets, const Tensor& class_weights,
	float label_smoothing = 0.0f);

/* Public, differentiable sequence cross entropy with an ignored target id
 *
 * Parameters:
 * Value logits: raw model output, shape {N, T, V}
 * Tensor targets: class id per position stored as a float, shape {N, T}
 * int ignore_index: positions whose target equals this id are excluded from the loss and the
 * gradient (default -100)
 * float label_smoothing: s in [0, 1), the share of the target mass spread uniformly over all V classes
 *
 * Returns Value: scalar shape {1}, the mean over the non-ignored positions of
 * (1-s) * (-log p[y]) + s * mean_c(-log p[c]); backward is fused,
 * (softmax - (1-s) * one_hot - s / V) / count, exactly 0 on ignored positions. Throws
 * TensorShapeError on a shape mismatch, std::invalid_argument on a target outside [0, V) that is
 * not ignore_index, a label_smoothing outside [0, 1) or when every position is ignored
 */
Value sequence_cross_entropy(const Value& logits, const Tensor& targets, int ignore_index = -100,
	float label_smoothing = 0.0f);

/* Public, differentiable mean squared error
 *
 * Parameters:
 * Value predictions: model output, any shape
 * Value targets: ground truth, same shape as predictions
 *
 * Returns Value: scalar shape {1}, the mean of (predictions - targets)^2 over every element
 */
Value mse_loss(const Value& predictions, const Value& targets);

/* Public, differentiable elementwise addition operator, alias for add(a, b) */
Value operator+(const Value& a, const Value& b);

/* Public, differentiable elementwise subtraction operator, alias for sub(a, b) */
Value operator-(const Value& a, const Value& b);

/* Public, differentiable elementwise multiplication operator, alias for mul(a, b) */
Value operator*(const Value& a, const Value& b);

/* Public, differentiable scalar multiplication operator, alias for scale(a, scalar) */
Value operator*(const Value& a, float scalar);
Value operator*(float scalar, const Value& a);

/* Public, differentiable negation operator, alias for scale(a, -1.0f) */
Value operator-(const Value& a);

}  // namespace nn
