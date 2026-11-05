/**
 * ops.cpp
 * Author: Miguel Mochizuki Silva
 * Description: Differentiable free functions over autograd Values
 */

#include "nn/autograd/ops.hpp"
#include "nn/kernels/parallel.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace nn {

namespace {

// Elements per thread below which an elementwise loop stays serial
constexpr size_t kElementwiseChunk = size_t(1) << 13;

/**
 * Sums a gradient over the axes that broadcasting expanded, so it matches the operand's shape
 *
 * Parameters:
 * const Tensor& grad: Gradient with the broadcast (output) shape
 * const std::vector<size_t>& target_shape: Shape of the operand that received the broadcast
 *
 * Returns Tensor: Gradient reduced to target_shape
 */
Tensor sum_to_shape(const Tensor& grad, const std::vector<size_t>& target_shape) {
	Tensor result = grad;
	while (result.ndim() > target_shape.size()) {
		result = result.sum(0);
	}
	for (size_t i = 0; i < target_shape.size(); i++) {
		if (target_shape[i] == 1 && result.shape()[i] != 1) {
			Tensor reduced = result.sum(static_cast<int>(i));
			std::vector<size_t> reshaped = reduced.shape();
			reshaped.insert(reshaped.begin() + i, 1);
			result = std::move(reduced).reshape(reshaped);
		}
	}
	return result;
}

/**
 * Adds a gradient into an accumulator, reducing broadcast axes first only when the shapes differ
 *
 * Parameters:
 * Tensor& dst: Accumulator, shaped like the operand that received the gradient
 * const Tensor& grad: Gradient with the broadcast (output) shape
 * bool negate: Subtract instead of add
 *
 * Returns void
 */
void accumulate_grad(Tensor& dst, const Tensor& grad, bool negate = false) {
	if (grad.shape() == dst.shape()) {
		if (negate) dst -= grad;
		else dst += grad;
		return;
	}
	Tensor reduced = sum_to_shape(grad, dst.shape());
	if (negate) dst -= reduced;
	else dst += reduced;
}

struct Conv2dOutputSize {
	size_t h_out;
	size_t w_out;
};

/**
 * Computes the spatial output size of a 2D convolution or pooling window
 *
 * Parameters:
 * size_t h, size_t w: Input height and width
 * size_t kh, size_t kw: Kernel height and width
 * size_t stride: Step between windows
 * size_t padding: Zero padding added on each side
 *
 * Returns Conv2dOutputSize: Output height and width
 */
Conv2dOutputSize conv2d_output_size(size_t h, size_t w, size_t kh, size_t kw, size_t stride, size_t padding) {
	size_t h_out = (h + 2 * padding - kh) / stride + 1;
	size_t w_out = (w + 2 * padding - kw) / stride + 1;
	return {h_out, w_out};
}

struct WindowRange {
	size_t lo;
	size_t hi;
};

/**
 * Finds the output columns whose window tap falls inside the input along one axis
 *
 * Parameters:
 * size_t w_out: Output size along the axis
 * size_t tap: Kernel offset along the axis
 * size_t stride: Step between windows
 * size_t padding: Zero padding added on each side
 * size_t W: Input size along the axis
 *
 * Returns WindowRange: Half-open range [lo, hi) of output indices o with 0 <= o*stride + tap - padding < W
 */
WindowRange valid_window(size_t w_out, size_t tap, size_t stride, size_t padding, size_t W) {
	size_t lo = tap >= padding ? 0 : (padding - tap + stride - 1) / stride;
	size_t hi = (W + padding > tap) ? (W + padding - tap - 1) / stride + 1 : 0;
	hi = std::min(hi, w_out);
	return {std::min(lo, hi), hi};
}

/**
 * Unrolls every convolution window of x into a column, so the convolution becomes a matmul
 *
 * Parameters:
 * const Tensor& x: Input of shape (N, Cin, H, W)
 * size_t kh, size_t kw: Kernel height and width
 * size_t stride: Step between windows
 * size_t padding: Zero padding added on each side
 *
 * Returns Tensor: Columns of shape (N, Cin * kh * kw, h_out * w_out)
 */
Tensor im2col(const Tensor& x, size_t kh, size_t kw, size_t stride, size_t padding) {
	size_t N = x.shape()[0], Cin = x.shape()[1], H = x.shape()[2], W = x.shape()[3];
	Conv2dOutputSize out_size = conv2d_output_size(H, W, kh, kw, stride, padding);
	size_t h_out = out_size.h_out, w_out = out_size.w_out;
	Tensor col({N, Cin * kh * kw, h_out * w_out});  // zero-filled, so padding cells are never written

	const float* x_data = x.data().data();
	float* col_data = col.data().data();
	size_t sN = x.strides()[0], sC = x.strides()[1], sH = x.strides()[2], sW = x.strides()[3];
	size_t col_sN = col.strides()[0], col_sRow = col.strides()[1];
	size_t per_sample = Cin * kh * kw * h_out * w_out;

	kernels::parallel_range(N, std::max<size_t>(1, kElementwiseChunk / std::max<size_t>(1, per_sample)),
	                        [=](size_t n_lo, size_t n_hi) {
		for (size_t n = n_lo; n < n_hi; n++) {
			for (size_t c = 0; c < Cin; c++) {
				for (size_t ph = 0; ph < kh; ph++) {
					for (size_t pw = 0; pw < kw; pw++) {
						size_t row = (c * kh + ph) * kw + pw;
						float* col_row = col_data + n * col_sN + row * col_sRow;
						const float* x_plane = x_data + n * sN + c * sC;
						WindowRange ow = valid_window(w_out, pw, stride, padding, W);
						for (size_t oh = 0; oh < h_out; oh++) {
							long ih = static_cast<long>(oh * stride + ph) - static_cast<long>(padding);
							if (ih < 0 || ih >= static_cast<long>(H)) continue;
							const float* src = x_plane + static_cast<size_t>(ih) * sH;
							float* dst = col_row + oh * w_out;
							size_t iw = ow.lo * stride + pw - padding;
							for (size_t o = ow.lo; o < ow.hi; o++, iw += stride) dst[o] = src[iw * sW];
						}
					}
				}
			}
		}
	});
	return col;
}

Tensor col2im(const Tensor& col_grad, const std::vector<size_t>& x_shape, size_t kh, size_t kw, size_t stride,
              size_t padding) {
	size_t N = x_shape[0], Cin = x_shape[1], H = x_shape[2], W = x_shape[3];
	Conv2dOutputSize out_size = conv2d_output_size(H, W, kh, kw, stride, padding);
	size_t h_out = out_size.h_out, w_out = out_size.w_out;
	Tensor grad_x(x_shape);

	const float* col_data = col_grad.data().data();
	float* x_data = grad_x.data().data();
	size_t sN = grad_x.strides()[0], sC = grad_x.strides()[1], sH = grad_x.strides()[2], sW = grad_x.strides()[3];
	size_t col_sN = col_grad.strides()[0], col_sRow = col_grad.strides()[1];
	size_t per_sample = Cin * kh * kw * h_out * w_out;

	// Each sample accumulates only into its own slice of grad_x, so samples can run on separate threads
	kernels::parallel_range(N, std::max<size_t>(1, kElementwiseChunk / std::max<size_t>(1, per_sample)),
	                        [=](size_t n_lo, size_t n_hi) {
		for (size_t n = n_lo; n < n_hi; n++) {
			for (size_t c = 0; c < Cin; c++) {
				for (size_t ph = 0; ph < kh; ph++) {
					for (size_t pw = 0; pw < kw; pw++) {
						size_t row = (c * kh + ph) * kw + pw;
						const float* col_row = col_data + n * col_sN + row * col_sRow;
						float* x_plane = x_data + n * sN + c * sC;
						WindowRange ow = valid_window(w_out, pw, stride, padding, W);
						for (size_t oh = 0; oh < h_out; oh++) {
							long ih = static_cast<long>(oh * stride + ph) - static_cast<long>(padding);
							if (ih < 0 || ih >= static_cast<long>(H)) continue;
							float* dst = x_plane + static_cast<size_t>(ih) * sH;
							const float* src = col_row + oh * w_out;
							size_t iw = ow.lo * stride + pw - padding;
							for (size_t o = ow.lo; o < ow.hi; o++, iw += stride) dst[iw * sW] += src[o];
						}
					}
				}
			}
		}
	});
	return grad_x;
}

/**
 * Inserts a size-1 axis, used to re-broadcast a reduced gradient
 *
 * Parameters:
 * const Tensor& t: Tensor to reshape
 * int axis: Position of the new axis
 *
 * Returns Tensor: Reshaped tensor with one more dimension
 */
Tensor insert_axis(const Tensor& t, int axis) {
	std::vector<size_t> new_shape = t.shape();
	new_shape.insert(new_shape.begin() + axis, 1);
	return t.reshape(new_shape);
}

}  // namespace

Value add(const Value& a, const Value& b) {
	Tensor result = add(a.data(), b.data());
	return Value::node(std::move(result), {a, b}, [a, b](const Tensor& grad_out) {
		if (a.requires_grad()) accumulate_grad(a.grad(), grad_out);
		if (b.requires_grad()) accumulate_grad(b.grad(), grad_out);
	});
}

Value sub(const Value& a, const Value& b) {
	Tensor result = sub(a.data(), b.data());
	return Value::node(std::move(result), {a, b}, [a, b](const Tensor& grad_out) {
		if (a.requires_grad()) accumulate_grad(a.grad(), grad_out);
		if (b.requires_grad()) accumulate_grad(b.grad(), grad_out, true);
	});
}

Value mul(const Value& a, const Value& b) {
	Tensor result = mul(a.data(), b.data());
	return Value::node(std::move(result), {a, b}, [a, b](const Tensor& grad_out) {
		if (a.requires_grad()) {
			accumulate_grad(a.grad(), mul(grad_out, b.data()));
		}
		if (b.requires_grad()) {
			accumulate_grad(b.grad(), mul(grad_out, a.data()));
		}
	});
}

Value scale(const Value& a, float scalar) {
	Tensor result = scale(a.data(), scalar);
	return Value::node(std::move(result), {a}, [a, scalar](const Tensor& grad_out) {
		if (a.requires_grad()) a.grad() += scale(grad_out, scalar);
	});
}

Value sum(const Value& x) {
	float total = 0.0f;
	for (float v : x.data().data()) total += v;
	Tensor result({1}, total);
	return Value::node(std::move(result), {x}, [x](const Tensor& grad_out) {
		if (x.requires_grad()) {
			Tensor ones(x.data().shape(), grad_out.data()[0]);
			x.grad() += ones;
		}
	});
}

Value transpose(const Value& x, std::vector<size_t> perm) {
	Tensor result = x.data().transpose(perm);
	std::vector<size_t> inverse(perm.size());
	for (size_t i = 0; i < perm.size(); i++) inverse[perm[i]] = i;
	return Value::node(std::move(result), {x}, [x, inverse](const Tensor& grad_out) {
		if (!x.requires_grad()) return;
		x.grad() += grad_out.transpose(inverse);
	});
}

Value reshape(const Value& x, std::vector<size_t> new_shape) {
	Tensor result = x.data().reshape(new_shape);
	std::vector<size_t> orig_shape = x.data().shape();
	return Value::node(std::move(result), {x}, [x, orig_shape](const Tensor& grad_out) {
		if (!x.requires_grad()) return;
		x.grad() += grad_out.reshape(orig_shape);
	});
}

Value matmul(const Value& a, const Value& b) {
	Tensor result = matmul(a.data(), b.data());
	return Value::node(std::move(result), {a, b}, [a, b](const Tensor& grad_out) {
		if (a.requires_grad()) {
			Tensor grad_a = matmul(grad_out, b.data(), false, true);
			accumulate_grad(a.grad(), grad_a);
		}
		if (b.requires_grad()) {
			Tensor grad_b = matmul(a.data(), grad_out, true, false);
			accumulate_grad(b.grad(), grad_b);
		}
	});
}

Value matmul_bt(const Value& a, const Value& b) {
	Tensor result = matmul(a.data(), b.data(), false, true);
	return Value::node(std::move(result), {a, b}, [a, b](const Tensor& grad_out) {
		if (a.requires_grad()) {
			Tensor grad_a = matmul(grad_out, b.data());
			accumulate_grad(a.grad(), grad_a);
		}
		if (b.requires_grad()) {
			Tensor grad_b = matmul(grad_out, a.data(), true, false);
			accumulate_grad(b.grad(), grad_b);
		}
	});
}

Value relu(const Value& x) {
	Tensor result = x.data();
	float* r = result.data().data();
	kernels::parallel_range(result.size(), kElementwiseChunk, [r](size_t lo, size_t hi) {
		for (size_t i = lo; i < hi; i++) r[i] = r[i] > 0.0f ? r[i] : 0.0f;
	});
	return Value::node(std::move(result), {x}, [x](const Tensor& grad_out) {
		if (!x.requires_grad()) return;
		const float* in = x.data().data().data();
		const float* g = grad_out.data().data();
		float* dst = x.grad().data().data();
		kernels::parallel_range(grad_out.size(), kElementwiseChunk, [=](size_t lo, size_t hi) {
			for (size_t i = lo; i < hi; i++) dst[i] += in[i] > 0.0f ? g[i] : 0.0f;
		});
	});
}

Value sigmoid(const Value& x) {
	Tensor result = x.data();
	for (float& v : result.data()) v = 1.0f / (1.0f + std::exp(-v));
	auto backward_fn = [x, result](const Tensor& grad_out) {
		if (!x.requires_grad()) return;
		Tensor local_grad(result.shape());
		for (size_t i = 0; i < local_grad.size(); i++) {
			float s = result.data()[i];
			local_grad.data()[i] = grad_out.data()[i] * s * (1.0f - s);
		}
		x.grad() += local_grad;
	};
	return Value::node(std::move(result), {x}, std::move(backward_fn));
}

Value tanh(const Value& x) {
	Tensor result = x.data();
	for (float& v : result.data()) v = std::tanh(v);
	auto backward_fn = [x, result](const Tensor& grad_out) {
		if (!x.requires_grad()) return;
		Tensor local_grad(result.shape());
		for (size_t i = 0; i < local_grad.size(); i++) {
			float t = result.data()[i];
			local_grad.data()[i] = grad_out.data()[i] * (1.0f - t * t);
		}
		x.grad() += local_grad;
	};
	return Value::node(std::move(result), {x}, std::move(backward_fn));
}

Value softmax(const Value& x, int axis) {
	Tensor max_val = insert_axis(x.data().max(axis), axis);
	Tensor shifted = sub(x.data(), max_val);
	Tensor exp_val = shifted;
	for (float& v : exp_val.data()) v = std::exp(v);
	Tensor sum_val = insert_axis(exp_val.sum(axis), axis);
	Tensor result = div(exp_val, sum_val);

	auto backward_fn = [x, result, axis](const Tensor& grad_out) {
		if (!x.requires_grad()) return;
		Tensor prod = mul(grad_out, result);
		Tensor sum_prod = insert_axis(prod.sum(axis), axis);
		Tensor local_grad = mul(result, sub(grad_out, sum_prod));
		x.grad() += local_grad;
	};
	return Value::node(std::move(result), {x}, std::move(backward_fn));
}

Value conv2d(const Value& x, const Value& weight, const Value& bias, int stride, int padding) {
	if (x.data().ndim() != 4 || weight.data().ndim() != 4) {
		throw TensorShapeError();
	}
	if (stride <= 0 || padding < 0) {
		throw TensorShapeError();
	}
	size_t Cout = weight.data().shape()[0];
	size_t Cin = weight.data().shape()[1];
	size_t kh = weight.data().shape()[2];
	size_t kw = weight.data().shape()[3];
	size_t N = x.data().shape()[0];
	size_t H = x.data().shape()[2];
	size_t W = x.data().shape()[3];
	if (x.data().shape()[1] != Cin || bias.data().size() != Cout) {
		throw TensorShapeError();
	}
	if (kh > H + 2 * static_cast<size_t>(padding) || kw > W + 2 * static_cast<size_t>(padding)) {
		throw TensorShapeError();
	}
	Conv2dOutputSize out_size =
		conv2d_output_size(H, W, kh, kw, static_cast<size_t>(stride), static_cast<size_t>(padding));
	size_t h_out = out_size.h_out;
	size_t w_out = out_size.w_out;

	Tensor col = im2col(x.data(), kh, kw, static_cast<size_t>(stride), static_cast<size_t>(padding));
	Tensor weight_mat = weight.data().reshape({Cout, Cin * kh * kw});
	Tensor out_biased = matmul(weight_mat, col);
	{
		size_t out_batch = out_biased.shape()[0];
		size_t hw = out_biased.shape()[2];
		float* out_data = out_biased.data().data();
		const float* bias_data = bias.data().data().data();
		for (size_t n = 0; n < out_batch; n++) {
			for (size_t c = 0; c < Cout; c++) {
				float bias_val = bias_data[c];
				float* row = out_data + (n * Cout + c) * hw;
				for (size_t i = 0; i < hw; i++) row[i] += bias_val;
			}
		}
	}
	Tensor result = std::move(out_biased).reshape({N, Cout, h_out, w_out});

	std::vector<size_t> x_shape = x.data().shape();

	return Value::node(
		result, {x, weight, bias},
		[x, weight, bias, col = std::move(col), x_shape, Cout, Cin, kh, kw, stride, padding, h_out, w_out](
			const Tensor& grad_out) {
			Tensor grad_out_mat = grad_out.reshape({grad_out.shape()[0], Cout, h_out * w_out});

			if (bias.requires_grad()) {
				Tensor grad_bias = grad_out_mat.sum(0).sum(1);
				bias.grad() += grad_bias;
			}
			if (weight.requires_grad()) {
				Tensor grad_weight_batched = matmul(grad_out_mat, col, false, true);
				Tensor grad_weight = grad_weight_batched.sum(0).reshape({Cout, Cin, kh, kw});
				weight.grad() += grad_weight;
			}
			if (x.requires_grad()) {
				Tensor weight_mat = weight.data().reshape({Cout, Cin * kh * kw});
				Tensor grad_col = matmul(weight_mat, grad_out_mat, true, false);
				Tensor grad_x =
					col2im(grad_col, x_shape, kh, kw, static_cast<size_t>(stride), static_cast<size_t>(padding));
				x.grad() += grad_x;
			}
		});
}

Value maxpool2d(const Value& x, size_t kernel_size, size_t stride) {
	if (x.data().ndim() != 4) {
		throw TensorShapeError();
	}
	if (kernel_size == 0 || stride == 0) {
		throw TensorShapeError();
	}
	size_t N = x.data().shape()[0];
	size_t C = x.data().shape()[1];
	size_t H = x.data().shape()[2];
	size_t W = x.data().shape()[3];
	if (kernel_size > H || kernel_size > W) {
		throw TensorShapeError();
	}
	Conv2dOutputSize out_size = conv2d_output_size(H, W, kernel_size, kernel_size, stride, 0);
	size_t h_out = out_size.h_out;
	size_t w_out = out_size.w_out;

	Tensor result({N, C, h_out, w_out});
	std::vector<size_t> argmax(N * C * h_out * w_out);

	const float* x_data = x.data().data().data();
	float* result_data = result.data().data();
	size_t* argmax_data = argmax.data();
	size_t out_plane = h_out * w_out;

	// Each (n, c) plane reads and writes only its own slices, so planes can run on separate threads
	size_t chunk = std::max<size_t>(1, kElementwiseChunk / std::max<size_t>(1, out_plane * kernel_size * kernel_size));
	kernels::parallel_range(N * C, chunk, [=](size_t p_lo, size_t p_hi) {
		for (size_t p = p_lo; p < p_hi; p++) {
			const float* plane = x_data + p * H * W;
			for (size_t oh = 0; oh < h_out; oh++) {
				for (size_t ow = 0; ow < w_out; ow++) {
					float best = -std::numeric_limits<float>::infinity();
					size_t best_flat = p * H * W;
					for (size_t ph = 0; ph < kernel_size; ph++) {
						const float* row = plane + (oh * stride + ph) * W + ow * stride;
						for (size_t pw = 0; pw < kernel_size; pw++) {
							if (row[pw] > best) {
								best = row[pw];
								best_flat = p * H * W + (oh * stride + ph) * W + ow * stride + pw;
							}
						}
					}
					result_data[p * out_plane + oh * w_out + ow] = best;
					argmax_data[p * out_plane + oh * w_out + ow] = best_flat;
				}
			}
		}
	});

	return Value::node(std::move(result), {x}, [x, argmax = std::move(argmax), N, C, out_plane](const Tensor& grad_out) {
		if (!x.requires_grad()) return;
		const float* g = grad_out.data().data();
		float* dst = x.grad().data().data();
		const size_t* arg = argmax.data();
		// Every argmax entry of plane p points inside plane p, so planes write disjoint ranges
		size_t chunk = std::max<size_t>(1, kElementwiseChunk / std::max<size_t>(1, out_plane));
		kernels::parallel_range(N * C, chunk, [=](size_t p_lo, size_t p_hi) {
			for (size_t i = p_lo * out_plane; i < p_hi * out_plane; i++) dst[arg[i]] += g[i];
		});
	});
}

Value cross_entropy_loss(const Value& logits, const Tensor& targets) {
	const Tensor& logits_data = logits.data();
	if (logits_data.ndim() != 2) {
		throw TensorShapeError();
	}
	size_t N = logits_data.shape()[0];
	size_t C = logits_data.shape()[1];
	if (targets.ndim() != 1 || targets.size() != N) {
		throw TensorShapeError();
	}

	Tensor max_val = insert_axis(logits_data.max(1), 1);
	Tensor shifted = sub(logits_data, max_val);
	Tensor exp_val = shifted;
	for (float& v : exp_val.data()) v = std::exp(v);
	Tensor sum_exp = exp_val.sum(1);

	Tensor softmax_val(logits_data.shape());
	float total_loss = 0.0f;
	for (size_t n = 0; n < N; n++) {
		float log_sum_exp = std::log(sum_exp.data()[n]);
		size_t target = static_cast<size_t>(std::round(targets.get({n})));
		for (size_t c = 0; c < C; c++) {
			float log_prob = shifted.get({n, c}) - log_sum_exp;
			softmax_val.at({n, c}) = std::exp(log_prob);
		}
		total_loss += -(shifted.get({n, target}) - log_sum_exp);
	}
	total_loss /= static_cast<float>(N);

	Tensor result({1}, total_loss);
	return Value::node(std::move(result), {logits}, [logits, softmax_val, targets, N, C](const Tensor& grad_out) {
		if (!logits.requires_grad()) return;
		Tensor local_grad = softmax_val;
		for (size_t n = 0; n < N; n++) {
			size_t target = static_cast<size_t>(std::round(targets.get({n})));
			local_grad.at({n, target}) -= 1.0f;
		}
		float grad_scale = grad_out.data()[0] / static_cast<float>(N);
		for (float& v : local_grad.data()) v *= grad_scale;
		logits.grad() += local_grad;
	});
}

Value mse_loss(const Value& predictions, const Value& targets) {
	Value diff = sub(predictions, targets);
	Value squared = mul(diff, diff);
	Value total = sum(squared);
	return scale(total, 1.0f / static_cast<float>(predictions.data().size()));
}

Value operator+(const Value& a, const Value& b) { return add(a, b); }
Value operator-(const Value& a, const Value& b) { return sub(a, b); }
Value operator*(const Value& a, const Value& b) { return mul(a, b); }
Value operator*(const Value& a, float scalar) { return scale(a, scalar); }
Value operator*(float scalar, const Value& a) { return scale(a, scalar); }
Value operator-(const Value& a) { return scale(a, -1.0f); }

}  // namespace nn
