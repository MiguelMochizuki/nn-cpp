/**
 * ops.cpp
 * Author: Miguel Mochizuki Silva
 * Description: Differentiable free functions over autograd Values
 */

#include "nn/autograd/ops.hpp"

#include <cmath>
#include <limits>

namespace nn {

namespace {

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
			result = reduced.reshape(reshaped);
		}
	}
	return result;
}

struct Conv2dOutputSize {
	size_t h_out;
	size_t w_out;
};

Conv2dOutputSize conv2d_output_size(size_t h, size_t w, size_t kh, size_t kw, size_t stride, size_t padding) {
	size_t h_out = (h + 2 * padding - kh) / stride + 1;
	size_t w_out = (w + 2 * padding - kw) / stride + 1;
	return {h_out, w_out};
}

Tensor im2col(const Tensor& x, size_t kh, size_t kw, size_t stride, size_t padding) {
	size_t N = x.shape()[0], Cin = x.shape()[1], H = x.shape()[2], W = x.shape()[3];
	auto [h_out, w_out] = conv2d_output_size(H, W, kh, kw, stride, padding);
	Tensor col({N, Cin * kh * kw, h_out * w_out});

	const float* x_data = x.data().data();
	float* col_data = col.data().data();
	size_t sN = x.strides()[0], sC = x.strides()[1], sH = x.strides()[2], sW = x.strides()[3];
	size_t col_sN = col.strides()[0], col_sRow = col.strides()[1];

	for (size_t n = 0; n < N; n++) {
		for (size_t c = 0; c < Cin; c++) {
			for (size_t ph = 0; ph < kh; ph++) {
				for (size_t pw = 0; pw < kw; pw++) {
					size_t row = (c * kh + ph) * kw + pw;
					size_t col_row_base = n * col_sN + row * col_sRow;
					for (size_t oh = 0; oh < h_out; oh++) {
						long ih = static_cast<long>(oh * stride + ph) - static_cast<long>(padding);
						bool ih_valid = ih >= 0 && ih < static_cast<long>(H);
						for (size_t ow = 0; ow < w_out; ow++) {
							long iw = static_cast<long>(ow * stride + pw) - static_cast<long>(padding);
							size_t col_idx = oh * w_out + ow;
							float v = 0.0f;
							if (ih_valid && iw >= 0 && iw < static_cast<long>(W)) {
								v = x_data[n * sN + c * sC + static_cast<size_t>(ih) * sH + static_cast<size_t>(iw) * sW];
							}
							col_data[col_row_base + col_idx] = v;
						}
					}
				}
			}
		}
	}
	return col;
}

Tensor col2im(const Tensor& col_grad, const std::vector<size_t>& x_shape, size_t kh, size_t kw, size_t stride,
              size_t padding) {
	size_t N = x_shape[0], Cin = x_shape[1], H = x_shape[2], W = x_shape[3];
	auto [h_out, w_out] = conv2d_output_size(H, W, kh, kw, stride, padding);
	Tensor grad_x(x_shape);

	const float* col_data = col_grad.data().data();
	float* x_data = grad_x.data().data();
	size_t sN = grad_x.strides()[0], sC = grad_x.strides()[1], sH = grad_x.strides()[2], sW = grad_x.strides()[3];
	size_t col_sN = col_grad.strides()[0], col_sRow = col_grad.strides()[1];

	for (size_t n = 0; n < N; n++) {
		for (size_t c = 0; c < Cin; c++) {
			for (size_t ph = 0; ph < kh; ph++) {
				for (size_t pw = 0; pw < kw; pw++) {
					size_t row = (c * kh + ph) * kw + pw;
					size_t col_row_base = n * col_sN + row * col_sRow;
					for (size_t oh = 0; oh < h_out; oh++) {
						long ih = static_cast<long>(oh * stride + ph) - static_cast<long>(padding);
						bool ih_valid = ih >= 0 && ih < static_cast<long>(H);
						for (size_t ow = 0; ow < w_out; ow++) {
							long iw = static_cast<long>(ow * stride + pw) - static_cast<long>(padding);
							if (ih_valid && iw >= 0 && iw < static_cast<long>(W)) {
								size_t col_idx = oh * w_out + ow;
								x_data[n * sN + c * sC + static_cast<size_t>(ih) * sH + static_cast<size_t>(iw) * sW] +=
									col_data[col_row_base + col_idx];
							}
						}
					}
				}
			}
		}
	}
	return grad_x;
}

Tensor transpose_last_two(const Tensor& t) {
	std::vector<size_t> perm(t.ndim());
	for (size_t i = 0; i < t.ndim(); i++) perm[i] = i;
	std::swap(perm[t.ndim() - 2], perm[t.ndim() - 1]);
	return t.transpose(perm);
}

Tensor insert_axis(const Tensor& t, int axis) {
	std::vector<size_t> new_shape = t.shape();
	new_shape.insert(new_shape.begin() + axis, 1);
	return t.reshape(new_shape);
}

}  // namespace

Value add(const Value& a, const Value& b) {
	Tensor result = add(a.data(), b.data());
	return Value::node(result, {a, b}, [a, b](const Tensor& grad_out) {
		if (a.requires_grad()) a.grad() = add(a.grad(), sum_to_shape(grad_out, a.data().shape()));
		if (b.requires_grad()) b.grad() = add(b.grad(), sum_to_shape(grad_out, b.data().shape()));
	});
}

Value sub(const Value& a, const Value& b) {
	Tensor result = sub(a.data(), b.data());
	return Value::node(result, {a, b}, [a, b](const Tensor& grad_out) {
		if (a.requires_grad()) a.grad() = add(a.grad(), sum_to_shape(grad_out, a.data().shape()));
		if (b.requires_grad()) b.grad() = sub(b.grad(), sum_to_shape(grad_out, b.data().shape()));
	});
}

Value mul(const Value& a, const Value& b) {
	Tensor result = mul(a.data(), b.data());
	return Value::node(result, {a, b}, [a, b](const Tensor& grad_out) {
		if (a.requires_grad()) {
			a.grad() = add(a.grad(), sum_to_shape(mul(grad_out, b.data()), a.data().shape()));
		}
		if (b.requires_grad()) {
			b.grad() = add(b.grad(), sum_to_shape(mul(grad_out, a.data()), b.data().shape()));
		}
	});
}

Value scale(const Value& a, float scalar) {
	Tensor result = scale(a.data(), scalar);
	return Value::node(result, {a}, [a, scalar](const Tensor& grad_out) {
		if (a.requires_grad()) a.grad() = add(a.grad(), scale(grad_out, scalar));
	});
}

Value sum(const Value& x) {
	float total = 0.0f;
	for (float v : x.data().data()) total += v;
	Tensor result({1}, total);
	return Value::node(result, {x}, [x](const Tensor& grad_out) {
		if (x.requires_grad()) {
			Tensor ones(x.data().shape(), grad_out.data()[0]);
			x.grad() = add(x.grad(), ones);
		}
	});
}

Value transpose(const Value& x, std::vector<size_t> perm) {
	Tensor result = x.data().transpose(perm);
	std::vector<size_t> inverse(perm.size());
	for (size_t i = 0; i < perm.size(); i++) inverse[perm[i]] = i;
	return Value::node(result, {x}, [x, inverse](const Tensor& grad_out) {
		if (!x.requires_grad()) return;
		x.grad() = add(x.grad(), grad_out.transpose(inverse));
	});
}

Value reshape(const Value& x, std::vector<size_t> new_shape) {
	Tensor result = x.data().reshape(new_shape);
	std::vector<size_t> orig_shape = x.data().shape();
	return Value::node(result, {x}, [x, orig_shape](const Tensor& grad_out) {
		if (!x.requires_grad()) return;
		x.grad() = add(x.grad(), grad_out.reshape(orig_shape));
	});
}

Value matmul(const Value& a, const Value& b) {
	Tensor result = matmul(a.data(), b.data());
	return Value::node(result, {a, b}, [a, b](const Tensor& grad_out) {
		if (a.requires_grad()) {
			Tensor grad_a = matmul(grad_out, transpose_last_two(b.data()));
			a.grad() = add(a.grad(), sum_to_shape(grad_a, a.data().shape()));
		}
		if (b.requires_grad()) {
			Tensor grad_b = matmul(transpose_last_two(a.data()), grad_out);
			b.grad() = add(b.grad(), sum_to_shape(grad_b, b.data().shape()));
		}
	});
}

Value relu(const Value& x) {
	Tensor result = x.data();
	for (float& v : result.data()) v = v > 0.0f ? v : 0.0f;
	return Value::node(result, {x}, [x](const Tensor& grad_out) {
		if (!x.requires_grad()) return;
		Tensor local_grad(x.data().shape());
		for (size_t i = 0; i < local_grad.size(); i++) {
			local_grad.data()[i] = (x.data().data()[i] > 0.0f) ? grad_out.data()[i] : 0.0f;
		}
		x.grad() = add(x.grad(), local_grad);
	});
}

Value sigmoid(const Value& x) {
	Tensor result = x.data();
	for (float& v : result.data()) v = 1.0f / (1.0f + std::exp(-v));
	Tensor saved = result;
	return Value::node(result, {x}, [x, saved](const Tensor& grad_out) {
		if (!x.requires_grad()) return;
		Tensor local_grad(saved.shape());
		for (size_t i = 0; i < local_grad.size(); i++) {
			float s = saved.data()[i];
			local_grad.data()[i] = grad_out.data()[i] * s * (1.0f - s);
		}
		x.grad() = add(x.grad(), local_grad);
	});
}

Value tanh(const Value& x) {
	Tensor result = x.data();
	for (float& v : result.data()) v = std::tanh(v);
	Tensor saved = result;
	return Value::node(result, {x}, [x, saved](const Tensor& grad_out) {
		if (!x.requires_grad()) return;
		Tensor local_grad(saved.shape());
		for (size_t i = 0; i < local_grad.size(); i++) {
			float t = saved.data()[i];
			local_grad.data()[i] = grad_out.data()[i] * (1.0f - t * t);
		}
		x.grad() = add(x.grad(), local_grad);
	});
}

Value softmax(const Value& x, int axis) {
	Tensor max_val = insert_axis(x.data().max(axis), axis);
	Tensor shifted = sub(x.data(), max_val);
	Tensor exp_val = shifted;
	for (float& v : exp_val.data()) v = std::exp(v);
	Tensor sum_val = insert_axis(exp_val.sum(axis), axis);
	Tensor result = div(exp_val, sum_val);
	Tensor saved = result;

	return Value::node(result, {x}, [x, saved, axis](const Tensor& grad_out) {
		if (!x.requires_grad()) return;
		Tensor prod = mul(grad_out, saved);
		Tensor sum_prod = insert_axis(prod.sum(axis), axis);
		Tensor local_grad = mul(saved, sub(grad_out, sum_prod));
		x.grad() = add(x.grad(), local_grad);
	});
}

Value conv2d(const Value& x, const Value& weight, const Value& bias, int stride, int padding) {
	size_t Cout = weight.data().shape()[0];
	size_t Cin = weight.data().shape()[1];
	size_t kh = weight.data().shape()[2];
	size_t kw = weight.data().shape()[3];
	size_t N = x.data().shape()[0];
	size_t H = x.data().shape()[2];
	size_t W = x.data().shape()[3];
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
	Tensor result = out_biased.reshape({N, Cout, h_out, w_out});

	Tensor saved_col = col;
	std::vector<size_t> x_shape = x.data().shape();

	return Value::node(
		result, {x, weight, bias},
		[x, weight, bias, saved_col, x_shape, Cout, Cin, kh, kw, stride, padding, h_out, w_out](
			const Tensor& grad_out) {
			Tensor grad_out_mat = grad_out.reshape({grad_out.shape()[0], Cout, h_out * w_out});

			if (bias.requires_grad()) {
				Tensor grad_bias = grad_out_mat.sum(0).sum(1);
				bias.grad() = add(bias.grad(), grad_bias);
			}
			if (weight.requires_grad()) {
				Tensor grad_weight_batched = matmul(grad_out_mat, transpose_last_two(saved_col));
				Tensor grad_weight = grad_weight_batched.sum(0).reshape({Cout, Cin, kh, kw});
				weight.grad() = add(weight.grad(), grad_weight);
			}
			if (x.requires_grad()) {
				Tensor weight_mat_t = transpose_last_two(weight.data().reshape({Cout, Cin * kh * kw}));
				Tensor grad_col = matmul(weight_mat_t, grad_out_mat);
				Tensor grad_x =
					col2im(grad_col, x_shape, kh, kw, static_cast<size_t>(stride), static_cast<size_t>(padding));
				x.grad() = add(x.grad(), grad_x);
			}
		});
}

Value maxpool2d(const Value& x, size_t kernel_size, size_t stride) {
	size_t N = x.data().shape()[0];
	size_t C = x.data().shape()[1];
	size_t H = x.data().shape()[2];
	size_t W = x.data().shape()[3];
	Conv2dOutputSize out_size = conv2d_output_size(H, W, kernel_size, kernel_size, stride, 0);
	size_t h_out = out_size.h_out;
	size_t w_out = out_size.w_out;

	Tensor result({N, C, h_out, w_out});
	std::vector<size_t> argmax(N * C * h_out * w_out);

	const float* x_data = x.data().data().data();
	float* result_data = result.data().data();

	for (size_t n = 0; n < N; n++) {
		for (size_t c = 0; c < C; c++) {
			for (size_t oh = 0; oh < h_out; oh++) {
				for (size_t ow = 0; ow < w_out; ow++) {
					float best = -std::numeric_limits<float>::infinity();
					size_t best_flat = 0;
					for (size_t ph = 0; ph < kernel_size; ph++) {
						for (size_t pw = 0; pw < kernel_size; pw++) {
							size_t ih = oh * stride + ph;
							size_t iw = ow * stride + pw;
							size_t flat = ((n * C + c) * H + ih) * W + iw;
							float v = x_data[flat];
							if (v > best) {
								best = v;
								best_flat = flat;
							}
						}
					}
					size_t out_idx = ((n * C + c) * h_out + oh) * w_out + ow;
					result_data[out_idx] = best;
					argmax[out_idx] = best_flat;
				}
			}
		}
	}

	std::vector<size_t> x_shape = x.data().shape();
	return Value::node(result, {x}, [x, argmax, x_shape](const Tensor& grad_out) {
		if (!x.requires_grad()) return;
		Tensor grad_x(x_shape);
		for (size_t i = 0; i < grad_out.size(); i++) {
			grad_x.data()[argmax[i]] += grad_out.data()[i];
		}
		x.grad() = add(x.grad(), grad_x);
	});
}

Value cross_entropy_loss(const Value& logits, const Tensor& targets) {
	const Tensor& logits_data = logits.data();
	size_t N = logits_data.shape()[0];
	size_t C = logits_data.shape()[1];

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
	return Value::node(result, {logits}, [logits, softmax_val, targets, N, C](const Tensor& grad_out) {
		if (!logits.requires_grad()) return;
		Tensor local_grad = softmax_val;
		for (size_t n = 0; n < N; n++) {
			size_t target = static_cast<size_t>(std::round(targets.get({n})));
			local_grad.at({n, target}) -= 1.0f;
		}
		float grad_scale = grad_out.data()[0] / static_cast<float>(N);
		for (float& v : local_grad.data()) v *= grad_scale;
		logits.grad() = add(logits.grad(), local_grad);
	});
}

Value mse_loss(const Value& predictions, const Value& targets) {
	Value diff = sub(predictions, targets);
	Value squared = mul(diff, diff);
	Value total = sum(squared);
	return scale(total, 1.0f / static_cast<float>(predictions.data().size()));
}

}  // namespace nn
