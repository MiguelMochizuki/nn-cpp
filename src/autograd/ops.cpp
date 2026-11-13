/**
 * ops.cpp
 * Author: Miguel Mochizuki Silva
 * Description: Differentiable free functions over autograd Values
 */

#include "nn/autograd/ops.hpp"
#include "nn/kernels/gemm.hpp"
#include "nn/kernels/parallel.hpp"
#include "nn/kernels/vmath.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>

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
	// The first reduction reads grad itself: copying it first would be a full serial pass for nothing
	const Tensor* cur = &grad;
	Tensor result = Tensor::empty({0});
	while (cur->ndim() > target_shape.size()) {
		result = cur->sum(0);
		cur = &result;
	}
	for (size_t i = 0; i < target_shape.size(); i++) {
		if (target_shape[i] == 1 && cur->shape()[i] != 1) {
			Tensor reduced = cur->sum(static_cast<int>(i));
			std::vector<size_t> reshaped = reduced.shape();
			reshaped.insert(reshaped.begin() + i, 1);
			result = std::move(reduced).reshape(reshaped);
			cur = &result;
		}
	}
	return cur == &grad ? grad : result;
}

/**
 * Adds one contribution into a gradient buffer, or stores it when the buffer holds nothing yet
 *
 * Parameters:
 * float* dst: Gradient buffer
 * bool first: Store instead of add (see Value::grad_accum)
 * size_t lo, size_t hi: Element range
 * F term: Callable giving the contribution of element i
 *
 * Returns void: storing writes term + 0, the bits that adding it to a zeroed buffer would leave
 */
template <typename F>
inline void accumulate_range(float* dst, bool first, size_t lo, size_t hi, F term) {
	if (first) {
		for (size_t i = lo; i < hi; i++) dst[i] = term(i) + 0.0f;
	} else {
		for (size_t i = lo; i < hi; i++) dst[i] += term(i);
	}
}

/**
 * Adds a same-sized flat array into the gradient of v, storing it if the gradient holds nothing yet
 *
 * Parameters:
 * const Value& v: Node whose gradient receives the contribution
 * const float* src: v.data().size() floats
 * bool negate: Subtract instead of add
 *
 * Returns void
 */
void add_to_grad(const Value& v, const float* src, bool negate = false) {
	bool first;
	Tensor& dst_t = v.grad_accum(first);
	float* dst = dst_t.data().data();
	kernels::parallel_range(dst_t.size(), kElementwiseChunk, [=](size_t lo, size_t hi) {
		if (negate) {
			// 0 - x, not -x: that is what subtracting from a zeroed buffer gives for x == +0
			if (first) for (size_t i = lo; i < hi; i++) dst[i] = 0.0f - src[i];
			else for (size_t i = lo; i < hi; i++) dst[i] -= src[i];
		} else {
			accumulate_range(dst, first, lo, hi, [=](size_t i) { return src[i]; });
		}
	});
}

/**
 * Adds a gradient into a node's accumulator, reducing broadcast axes first only when the shapes differ
 *
 * Parameters:
 * const Value& v: Node that received the broadcast; its gradient is shaped like its data
 * const Tensor& grad: Gradient with the broadcast (output) shape
 * bool negate: Subtract instead of add
 *
 * Returns void
 */
void accumulate_grad(const Value& v, const Tensor& grad, bool negate = false) {
	if (grad.shape() == v.data().shape()) {
		add_to_grad(v, grad.data().data(), negate);
		return;
	}
	Tensor reduced = sum_to_shape(grad, v.data().shape());
	add_to_grad(v, reduced.data().data(), negate);
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
 * Geometry of one convolution, shared by the per-sample im2col and col2im
 *
 * Attributes:
 * size_t Cin, H, W: Input channels, height and width
 * size_t kh, kw: Kernel height and width
 * size_t stride, padding: Step between windows and zero padding on each side
 * size_t h_out, w_out: Output height and width
 */
struct ConvGeometry {
	size_t Cin, H, W, kh, kw, stride, padding, h_out, w_out;
};

/**
 * Unrolls every convolution window of one sample into a column, so the convolution becomes a matmul
 *
 * Parameters:
 * const float* x: The sample, Cin planes of H x W
 * float* col: Output, (Cin * kh * kw) rows of h_out * w_out, every cell written (taps in the padding as zero)
 * const ConvGeometry& g: The convolution
 *
 * Returns void
 */
void im2col_sample(const float* x, float* col, const ConvGeometry& g) {
	const size_t hw = g.h_out * g.w_out;
	for (size_t c = 0; c < g.Cin; c++) {
		for (size_t ph = 0; ph < g.kh; ph++) {
			for (size_t pw = 0; pw < g.kw; pw++) {
				float* col_row = col + ((c * g.kh + ph) * g.kw + pw) * hw;
				const float* x_plane = x + c * g.H * g.W;
				WindowRange ow = valid_window(g.w_out, pw, g.stride, g.padding, g.W);
				for (size_t oh = 0; oh < g.h_out; oh++) {
					long ih = static_cast<long>(oh * g.stride + ph) - static_cast<long>(g.padding);
					float* dst = col_row + oh * g.w_out;
					if (ih < 0 || ih >= static_cast<long>(g.H)) {
						std::fill(dst, dst + g.w_out, 0.0f);
						continue;
					}
					const float* src = x_plane + static_cast<size_t>(ih) * g.W;
					std::fill(dst, dst + ow.lo, 0.0f);
					std::fill(dst + ow.hi, dst + g.w_out, 0.0f);
					size_t iw = ow.lo * g.stride + pw - g.padding;
					for (size_t o = ow.lo; o < ow.hi; o++, iw += g.stride) dst[o] = src[iw];
				}
			}
		}
	}
}

/**
 * Adds one sample's column gradient back into its input gradient, the transpose of im2col_sample
 *
 * Parameters:
 * const float* col: Column gradient, laid out as im2col_sample writes it
 * float* x: The sample's input gradient, Cin planes of H x W, accumulated into
 * const ConvGeometry& g: The convolution
 *
 * Returns void
 */
void col2im_sample_add(const float* col, float* x, const ConvGeometry& g) {
	const size_t hw = g.h_out * g.w_out;
	for (size_t c = 0; c < g.Cin; c++) {
		for (size_t ph = 0; ph < g.kh; ph++) {
			for (size_t pw = 0; pw < g.kw; pw++) {
				const float* col_row = col + ((c * g.kh + ph) * g.kw + pw) * hw;
				float* x_plane = x + c * g.H * g.W;
				WindowRange ow = valid_window(g.w_out, pw, g.stride, g.padding, g.W);
				for (size_t oh = 0; oh < g.h_out; oh++) {
					long ih = static_cast<long>(oh * g.stride + ph) - static_cast<long>(g.padding);
					if (ih < 0 || ih >= static_cast<long>(g.H)) continue;
					float* dst = x_plane + static_cast<size_t>(ih) * g.W;
					const float* src = col_row + oh * g.w_out;
					size_t iw = ow.lo * g.stride + pw - g.padding;
					for (size_t o = ow.lo; o < ow.hi; o++, iw += g.stride) dst[iw] += src[o];
				}
			}
		}
	}
}

/**
 * Allocates n uninitialised flag bytes for a folded ReLU; every byte is written by the forward pass
 *
 * Parameters:
 * size_t n: Number of flags
 *
 * Returns shared_ptr<unsigned char>: the buffer, freed with delete[]
 */
std::shared_ptr<unsigned char> alloc_alive(size_t n) {
	return std::shared_ptr<unsigned char>(new unsigned char[n], std::default_delete<unsigned char[]>());
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

/**
 * Shared body of the axis sum and mean
 *
 * Parameters:
 * const Value& x: Operand
 * int axis: Dimension to reduce, negative counts from the end
 * bool keepdim: Keep the reduced axis with size 1
 * bool average: Divide by the axis length
 *
 * Returns Value: The reduced Value; each output element is summed serially in axis order,
 * so the result is independent of the thread count
 */
Value reduce_axis(const Value& x, int axis, bool keepdim, bool average) {
	const std::vector<size_t>& shape = x.data().shape();
	int nd = static_cast<int>(shape.size());
	if (axis < -nd || axis >= nd) throw TensorIndexError();
	size_t ax = static_cast<size_t>(axis < 0 ? axis + nd : axis);
	size_t n = shape[ax], outer = 1, inner = 1;
	if (n == 0) throw TensorShapeError();  // would divide by zero in mean and has no defined max
	for (size_t i = 0; i < ax; i++) outer *= shape[i];
	for (size_t i = ax + 1; i < shape.size(); i++) inner *= shape[i];
	std::vector<size_t> out_shape;
	for (size_t i = 0; i < shape.size(); i++) {
		if (i != ax) out_shape.push_back(shape[i]);
		else if (keepdim) out_shape.push_back(1);
	}
	if (out_shape.empty()) out_shape.push_back(1);
	float factor = average ? 1.0f / static_cast<float>(n) : 1.0f;
	Tensor result = Tensor::empty(out_shape);
	const float* src = x.data().data().data();
	float* dst = result.data().data();
	size_t outputs = outer * inner;
	kernels::parallel_range(outputs, std::max<size_t>(1, kElementwiseChunk / std::max<size_t>(1, n)),
	                        [=](size_t lo, size_t hi) {
		for (size_t j = lo; j < hi; j++) {
			size_t base = (j / inner) * n * inner + j % inner;
			float total = 0.0f;
			for (size_t k = 0; k < n; k++) total += src[base + k * inner];
			dst[j] = average ? total * factor : total;
		}
	});
	return Value::node(std::move(result), {x}, [x, n, inner, outputs, factor](const Tensor& grad_out) {
		if (!x.requires_grad()) return;
		const float* g = grad_out.data().data();
		bool first;
		float* dx = x.grad_accum(first).data().data();
		kernels::parallel_range(outputs, std::max<size_t>(1, kElementwiseChunk / std::max<size_t>(1, n)),
		                        [=](size_t lo, size_t hi) {
			for (size_t j = lo; j < hi; j++) {
				size_t base = (j / inner) * n * inner + j % inner;
				if (first) for (size_t k = 0; k < n; k++) dx[base + k * inner] = g[j] * factor + 0.0f;
				else for (size_t k = 0; k < n; k++) dx[base + k * inner] += g[j] * factor;
			}
		});
	});
}

}  // namespace

Value add(const Value& a, const Value& b) {
	Tensor result = add(a.data(), b.data());
	return Value::node(std::move(result), {a, b}, [a, b](const Tensor& grad_out) {
		if (a.requires_grad()) accumulate_grad(a, grad_out);
		if (b.requires_grad()) accumulate_grad(b, grad_out);
	});
}

Value sub(const Value& a, const Value& b) {
	Tensor result = sub(a.data(), b.data());
	return Value::node(std::move(result), {a, b}, [a, b](const Tensor& grad_out) {
		if (a.requires_grad()) accumulate_grad(a, grad_out);
		if (b.requires_grad()) accumulate_grad(b, grad_out, true);
	});
}

Value mul(const Value& a, const Value& b) {
	Tensor result = mul(a.data(), b.data());
	return Value::node(std::move(result), {a, b}, [a, b](const Tensor& grad_out) {
		if (a.requires_grad()) {
			accumulate_grad(a, mul(grad_out, b.data()));
		}
		if (b.requires_grad()) {
			accumulate_grad(b, mul(grad_out, a.data()));
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

Value sum(const Value& x, int axis, bool keepdim) { return reduce_axis(x, axis, keepdim, false); }

Value mean(const Value& x, int axis, bool keepdim) { return reduce_axis(x, axis, keepdim, true); }

Value rsqrt(const Value& x) {
	Tensor result = x.data();
	for (float& v : result.data()) v = 1.0f / std::sqrt(v);
	return Value::node(std::move(result), {x}, [x](const Tensor& grad_out) {
		if (!x.requires_grad()) return;
		const float* in = x.data().data().data();
		const float* g = grad_out.data().data();
		float* dst = x.grad().data().data();
		for (size_t i = 0; i < grad_out.size(); i++) dst[i] += -0.5f * std::pow(in[i], -1.5f) * g[i];
	});
}

namespace {

/**
 * Builds a unary elementwise Value from a forward function and a derivative function
 *
 * Parameters:
 * const Value& x: Operand
 * F fwd: float -> float forward map
 * D deriv: x -> dy/dx, given the input
 *
 * Returns Value: fwd(x) elementwise; backward multiplies grad_out by deriv
 */
template <typename F, typename D>
Value unary_elementwise(const Value& x, F fwd, D deriv) {
	Tensor result = Tensor::empty(x.data().shape());
	const float* in = x.data().data().data();
	float* r = result.data().data();
	kernels::parallel_range(result.size(), kElementwiseChunk, [in, r, fwd](size_t lo, size_t hi) {
		for (size_t i = lo; i < hi; i++) r[i] = fwd(in[i]);
	});
	return Value::node(std::move(result), {x}, [x, deriv](const Tensor& grad_out) {
		if (!x.requires_grad()) return;
		const float* in = x.data().data().data();
		const float* g = grad_out.data().data();
		bool first;
		float* dst = x.grad_accum(first).data().data();
		kernels::parallel_range(grad_out.size(), kElementwiseChunk, [=](size_t lo, size_t hi) {
			accumulate_range(dst, first, lo, hi, [=](size_t i) { return g[i] * deriv(in[i]); });
		});
	});
}

}  // namespace

Value exp(const Value& x) {
	return unary_elementwise(x, [](float v) { return std::exp(v); }, [](float v) { return std::exp(v); });
}

Value log(const Value& x) {
	return unary_elementwise(x, [](float v) { return std::log(v); }, [](float v) { return 1.0f / v; });
}

Value sqrt(const Value& x) {
	return unary_elementwise(x, [](float v) { return std::sqrt(v); }, [](float v) { return 0.5f / std::sqrt(v); });
}

Value div(const Value& a, const Value& b) {
	Tensor result = div(a.data(), b.data());
	return Value::node(std::move(result), {a, b}, [a, b](const Tensor& grad_out) {
		if (a.requires_grad()) accumulate_grad(a, div(grad_out, b.data()));
		if (b.requires_grad()) {
			Tensor q = div(mul(grad_out, a.data()), mul(b.data(), b.data()));
			accumulate_grad(b, q, true);
		}
	});
}

Value gelu(const Value& x) {
	const float k = 0.7978845608f;  // sqrt(2/pi)
	const float c = 0.044715f;
	Tensor result = Tensor::empty(x.data().shape());
	const size_t n = result.size();
	// The tanh of the forward pass is kept for the backward pass, which would otherwise compute it again
	std::shared_ptr<float> saved(new float[n], std::default_delete<float[]>());
	const float* in = x.data().data().data();
	float* r = result.data().data();
	float* th = saved.get();
	kernels::parallel_range(n, kElementwiseChunk, [=](size_t lo, size_t hi) {
		// tanh_inplace gives each element the same bits wherever the chunk boundaries fall
		for (size_t i = lo; i < hi; i++) th[i] = k * (in[i] + c * in[i] * in[i] * in[i]);
		kernels::tanh_inplace(th + lo, hi - lo);
		for (size_t i = lo; i < hi; i++) r[i] = 0.5f * in[i] * (1.0f + th[i]);
	});
	return Value::node(std::move(result), {x}, [x, saved, n, k, c](const Tensor& grad_out) {
		if (!x.requires_grad()) return;
		const float* in = x.data().data().data();
		const float* g = grad_out.data().data();
		const float* th = saved.get();
		bool first;
		float* dst = x.grad_accum(first).data().data();
		kernels::parallel_range(n, kElementwiseChunk, [=](size_t lo, size_t hi) {
			accumulate_range(dst, first, lo, hi, [=](size_t i) {
				const float v = in[i], t = th[i];
				return g[i] * (0.5f * (1.0f + t) + 0.5f * v * (1.0f - t * t) * k * (1.0f + 3.0f * c * v * v));
			});
		});
	});
}

namespace {

/**
 * Resolves a possibly negative axis and the outer/size/inner factorisation around it
 *
 * Parameters:
 * const std::vector<size_t>& shape: Tensor shape
 * int axis: Axis, negative counts from the end
 * size_t& outer: Receives the product of dimensions before axis
 * size_t& inner: Receives the product of dimensions after axis
 *
 * Returns size_t: the normalised axis; throws TensorIndexError if out of range
 */
size_t axis_split(const std::vector<size_t>& shape, int axis, size_t& outer, size_t& inner) {
	int nd = static_cast<int>(shape.size());
	if (axis < -nd || axis >= nd) throw TensorIndexError();
	size_t ax = static_cast<size_t>(axis < 0 ? axis + nd : axis);
	outer = inner = 1;
	for (size_t i = 0; i < ax; i++) outer *= shape[i];
	for (size_t i = ax + 1; i < shape.size(); i++) inner *= shape[i];
	return ax;
}

/**
 * Copies (or accumulates) a band of an axis between a wide and a narrow tensor
 *
 * Parameters:
 * float* wide: Tensor with axis length wn
 * float* narrow: Tensor with axis length nn
 * size_t outer: Product of dimensions before the axis
 * size_t inner: Product of dimensions after the axis
 * size_t wn: Axis length of wide
 * size_t nn: Axis length of narrow
 * size_t offset: Where the band starts inside wide's axis
 * bool to_narrow: true copies wide into narrow; false adds narrow into wide
 *
 * Returns void: each outer row is independent, so the result does not depend on the thread count
 */
void copy_band(float* wide, float* narrow, size_t outer, size_t inner, size_t wn, size_t nn, size_t offset,
               bool to_narrow) {
	size_t row = nn * inner;
	kernels::parallel_range(outer, std::max<size_t>(1, kElementwiseChunk / std::max<size_t>(1, row)),
	                        [=](size_t lo, size_t hi) {
		for (size_t o = lo; o < hi; o++) {
			float* w = wide + (o * wn + offset) * inner;
			float* n = narrow + o * row;
			if (to_narrow) for (size_t i = 0; i < row; i++) n[i] = w[i];
			else for (size_t i = 0; i < row; i++) w[i] += n[i];
		}
	});
}

}  // namespace

Value masked_fill(const Value& x, const Tensor& mask, float value) {
	const std::vector<size_t>& xs = x.data().shape();
	const std::vector<size_t>& ms = mask.shape();
	if (ms.size() > xs.size()) throw TensorShapeError();
	size_t lead = xs.size() - ms.size();
	for (size_t i = 0; i < ms.size(); i++)
		if (ms[i] != xs[lead + i] && ms[i] != 1) throw TensorShapeError();
	// Expand the mask to one flag per element of x via zero strides on broadcast dims.
	size_t total = x.data().size();
	std::vector<size_t> mstride(xs.size(), 0);
	size_t st = 1;
	for (size_t i = ms.size(); i-- > 0;) {
		mstride[lead + i] = ms[i] == 1 ? 0 : st;
		st *= ms[i];
	}
	std::vector<char> hit(total);
	const float* mp = mask.data().data();
	Tensor result = x.data();
	float* r = result.data().data();
	char* hp = hit.data();
	const size_t rank = xs.size();
	// Walk x in order; the mask offset follows an odometer over x's axes (zero stride on broadcast axes)
	kernels::parallel_range(total, kElementwiseChunk, [&](size_t lo, size_t hi) {
		std::vector<size_t> idx(rank);
		size_t rem = lo, mi = 0;
		for (size_t d = rank; d-- > 0;) {
			idx[d] = rem % xs[d];
			rem /= xs[d];
			mi += idx[d] * mstride[d];
		}
		for (size_t f = lo; f < hi; f++) {
			const bool h = mp[mi] != 0.0f;
			hp[f] = h;
			if (h) r[f] = value;
			for (size_t d = rank; d-- > 0;) {
				idx[d]++;
				mi += mstride[d];
				if (idx[d] < xs[d]) break;
				mi -= idx[d] * mstride[d];
				idx[d] = 0;
			}
		}
	});
	return Value::node(std::move(result), {x}, [x, hit = std::move(hit)](const Tensor& grad_out) {
		if (!x.requires_grad()) return;
		const float* g = grad_out.data().data();
		float* dx = x.grad().data().data();
		const char* h = hit.data();
		kernels::parallel_range(hit.size(), kElementwiseChunk, [=](size_t lo, size_t hi) {
			for (size_t f = lo; f < hi; f++)
				if (!h[f]) dx[f] += g[f];
		});
	});
}

Value slice(const Value& x, int axis, size_t start, size_t end) {
	const std::vector<size_t>& shape = x.data().shape();
	size_t outer, inner;
	size_t ax = axis_split(shape, axis, outer, inner);
	size_t n = shape[ax];
	if (start >= end || end > n) throw TensorShapeError();
	std::vector<size_t> out_shape = shape;
	out_shape[ax] = end - start;
	Tensor result = Tensor::empty(out_shape);
	copy_band(const_cast<float*>(x.data().data().data()), result.data().data(), outer, inner, n, end - start, start,
	          true);
	return Value::node(std::move(result), {x}, [x, outer, inner, n, start, end](const Tensor& grad_out) {
		if (!x.requires_grad()) return;
		copy_band(x.grad().data().data(), const_cast<float*>(grad_out.data().data()), outer, inner, n, end - start,
		          start, false);
	});
}

Value concat(const std::vector<Value>& xs, int axis) {
	if (xs.empty()) throw TensorShapeError();
	const std::vector<size_t>& first = xs[0].data().shape();
	size_t outer, inner;
	size_t ax = axis_split(first, axis, outer, inner);
	std::vector<size_t> out_shape = first;
	out_shape[ax] = 0;
	for (const Value& v : xs) {
		const std::vector<size_t>& s = v.data().shape();
		if (s.size() != first.size()) throw TensorShapeError();
		for (size_t i = 0; i < s.size(); i++)
			if (i != ax && s[i] != first[i]) throw TensorShapeError();
		out_shape[ax] += s[ax];
	}
	size_t total_n = out_shape[ax];
	Tensor result(out_shape);  // copy_band accumulates into it, so it must start at zero
	size_t offset = 0;
	for (const Value& v : xs) {
		size_t n = v.data().shape()[ax];
		copy_band(result.data().data(), const_cast<float*>(v.data().data().data()), outer, inner, total_n, n, offset,
		          false);
		offset += n;
	}
	return Value::node(std::move(result), xs, [xs, ax, outer, inner, total_n](const Tensor& grad_out) {
		size_t off = 0;
		for (const Value& v : xs) {
			size_t n = v.data().shape()[ax];
			if (v.requires_grad()) {
				Tensor piece(v.data().shape());
				copy_band(const_cast<float*>(grad_out.data().data()), piece.data().data(), outer, inner, total_n, n,
				          off, true);
				v.grad() += piece;
			}
			off += n;
		}
	});
}

std::vector<Value> split(const Value& x, int axis, size_t parts) {
	const std::vector<size_t>& shape = x.data().shape();
	size_t outer, inner;
	size_t ax = axis_split(shape, axis, outer, inner);
	if (parts == 0 || shape[ax] % parts != 0) throw TensorShapeError();
	size_t w = shape[ax] / parts;
	std::vector<Value> out;
	for (size_t p = 0; p < parts; p++) out.push_back(slice(x, axis, p * w, (p + 1) * w));
	return out;
}

Value transpose(const Value& x, std::vector<size_t> perm) {
	Tensor result = x.data().transpose(perm);
	std::vector<size_t> inverse(perm.size());
	for (size_t i = 0; i < perm.size(); i++) inverse[perm[i]] = i;
	return Value::node(std::move(result), {x}, [x, inverse](const Tensor& grad_out) {
		if (!x.requires_grad()) return;
		// Permuted straight into the gradient: no transposed temporary and no second pass to add it
		bool first;
		float* dx = x.grad_accum(first).data().data();
		grad_out.transpose_into(inverse, dx, first ? Tensor::Combine::kAddToZero : Tensor::Combine::kAdd);
	});
}

Value reshape(const Value& x, std::vector<size_t> new_shape) {
	Tensor result = x.data().reshape(new_shape);
	std::vector<size_t> orig_shape = x.data().shape();
	return Value::node(std::move(result), {x}, [x, orig_shape](const Tensor& grad_out) {
		if (!x.requires_grad()) return;
		// Same elements in the same order: add them straight from grad_out, with no reshaped copy
		if (grad_out.size() != x.data().size()) throw TensorShapeError();
		add_to_grad(x, grad_out.data().data());
	});
}

Value matmul(const Value& a, const Value& b) {
	Tensor result = matmul(a.data(), b.data());
	return Value::node(std::move(result), {a, b}, [a, b](const Tensor& grad_out) {
		// b a plain {K, n} matrix: fold a's leading axes into M rows and accumulate straight into the
		// gradients, with no per-sample temporary and no reduction over the folded axes
		if (b.data().ndim() == 2 && a.data().ndim() >= 2) {
			const size_t K = b.data().shape()[0], n = b.data().shape()[1], M = a.data().size() / K;
			// A gradient that holds nothing yet is overwritten: sgemm then starts each sum from zero,
			// exactly as accumulating into a zeroed buffer does
			bool first;
			if (a.requires_grad()) {
				float* da = a.grad_accum(first).data().data();
				kernels::sgemm(false, true, M, K, n, grad_out.data().data(), n, b.data().data().data(), n, da, K, !first);
			}
			if (b.requires_grad()) {
				float* db = b.grad_accum(first).data().data();
				kernels::sgemm(true, false, K, n, M, a.data().data().data(), K, grad_out.data().data(), n, db, n, !first);
			}
			return;
		}
		if (a.requires_grad()) {
			Tensor grad_a = matmul(grad_out, b.data(), false, true);
			accumulate_grad(a, grad_a);
		}
		if (b.requires_grad()) {
			Tensor grad_b = matmul(a.data(), grad_out, true, false);
			accumulate_grad(b, grad_b);
		}
	});
}

Value linear(const Value& x, const Value& weight, const Value& bias) {
	const std::vector<size_t>& xs = x.data().shape();
	const std::vector<size_t>& ws = weight.data().shape();
	if (xs.size() < 2 || ws.size() != 2 || xs.back() != ws[1] || bias.data().shape() != std::vector<size_t>{ws[0]}) {
		throw TensorShapeError();
	}
	const size_t n = ws[0], K = ws[1], M = x.data().size() / K;
	std::vector<size_t> out_shape(xs.begin(), xs.end() - 1);
	out_shape.push_back(n);
	Tensor result = Tensor::empty(out_shape);  // sgemm overwrites all of it
	float* out = result.data().data();
	kernels::sgemm(false, true, M, n, K, x.data().data().data(), K, weight.data().data().data(), K, out, n, false);
	const float* bd = bias.data().data().data();
	kernels::parallel_range(M, std::max<size_t>(1, kElementwiseChunk / std::max<size_t>(1, n)), [=](size_t lo, size_t hi) {
		for (size_t r = lo; r < hi; r++)
			for (size_t j = 0; j < n; j++) out[r * n + j] += bd[j];
	});
	return Value::node(std::move(result), {x, weight, bias}, [x, weight, bias, M, n, K](const Tensor& grad_out) {
		if (grad_out.size() != M * n) throw TensorShapeError();
		const float* g = grad_out.data().data();
		bool first;
		if (x.requires_grad()) {
			float* dx = x.grad_accum(first).data().data();
			kernels::sgemm(false, false, M, K, n, g, n, weight.data().data().data(), K, dx, K, !first);
		}
		if (weight.requires_grad()) {
			float* dw = weight.grad_accum(first).data().data();
			kernels::sgemm(true, false, n, K, M, g, n, x.data().data().data(), K, dw, K, !first);
		}
		if (bias.requires_grad()) accumulate_grad(bias, grad_out);
	});
}

Value matmul_bt(const Value& a, const Value& b) {
	Tensor result = matmul(a.data(), b.data(), false, true);
	return Value::node(std::move(result), {a, b}, [a, b](const Tensor& grad_out) {
		// Same folding as matmul, with b stored as {n, K}
		if (b.data().ndim() == 2 && a.data().ndim() >= 2) {
			const size_t n = b.data().shape()[0], K = b.data().shape()[1], M = a.data().size() / K;
			bool first;
			if (a.requires_grad()) {
				float* da = a.grad_accum(first).data().data();
				kernels::sgemm(false, false, M, K, n, grad_out.data().data(), n, b.data().data().data(), K, da, K, !first);
			}
			if (b.requires_grad()) {
				float* db = b.grad_accum(first).data().data();
				kernels::sgemm(true, false, n, K, M, grad_out.data().data(), n, a.data().data().data(), K, db, K, !first);
			}
			return;
		}
		if (a.requires_grad()) {
			Tensor grad_a = matmul(grad_out, b.data());
			accumulate_grad(a, grad_a);
		}
		if (b.requires_grad()) {
			Tensor grad_b = matmul(grad_out, a.data(), true, false);
			accumulate_grad(b, grad_b);
		}
	});
}

Value relu(const Value& x) {
	Tensor result = Tensor::empty(x.data().shape());
	const float* in = x.data().data().data();
	float* r = result.data().data();
	kernels::parallel_range(result.size(), kElementwiseChunk, [in, r](size_t lo, size_t hi) {
		for (size_t i = lo; i < hi; i++) r[i] = in[i] > 0.0f ? in[i] : 0.0f;
	});
	return Value::node(std::move(result), {x}, [x](const Tensor& grad_out) {
		if (!x.requires_grad()) return;
		if (x.data().shape() != grad_out.shape()) throw TensorShapeError();
		const float* in = x.data().data().data();
		const float* g = grad_out.data().data();
		bool first;
		float* dst = x.grad_accum(first).data().data();
		kernels::parallel_range(grad_out.size(), kElementwiseChunk, [=](size_t lo, size_t hi) {
			accumulate_range(dst, first, lo, hi, [=](size_t i) { return in[i] > 0.0f ? g[i] : 0.0f; });
		});
	});
}

/**
 * Sum of n floats with 8 independent partial sums, so the loop vectorises; the order is fixed
 *
 * Parameters:
 * const float* p: Elements
 * size_t n: Element count
 *
 * Returns float: The sum
 */
static float lane_sum(const float* p, size_t n) {
	float a[8] = {0, 0, 0, 0, 0, 0, 0, 0};
	size_t i = 0;
	for (; i + 8 <= n; i += 8)
		for (size_t k = 0; k < 8; k++) a[k] += p[i + k];
	float t = ((a[0] + a[1]) + (a[2] + a[3])) + ((a[4] + a[5]) + (a[6] + a[7]));
	for (; i < n; i++) t += p[i];
	return t;
}

/**
 * Sum of (p[i] - mu)^2 with 8 independent partial sums, in a fixed order
 *
 * Parameters:
 * const float* p: Elements
 * size_t n: Element count
 * float mu: Value subtracted from each element
 *
 * Returns float: The sum of squared deviations
 */
static float lane_sq(const float* p, size_t n, float mu) {
	float a[8] = {0, 0, 0, 0, 0, 0, 0, 0};
	size_t i = 0;
	for (; i + 8 <= n; i += 8)
		for (size_t k = 0; k < 8; k++) {
			const float d = p[i + k] - mu;
			a[k] += d * d;
		}
	float t = ((a[0] + a[1]) + (a[2] + a[3])) + ((a[4] + a[5]) + (a[6] + a[7]));
	for (; i < n; i++) t += (p[i] - mu) * (p[i] - mu);
	return t;
}

/**
 * Sum of g[i] * (x[i] - mu) with 8 independent partial sums, in a fixed order
 *
 * Parameters:
 * const float* g: Gradient elements
 * const float* x: Input elements
 * size_t n: Element count
 * float mu: Value subtracted from each input element
 *
 * Returns float: The sum
 */
static float lane_dot(const float* g, const float* x, size_t n, float mu) {
	float a[8] = {0, 0, 0, 0, 0, 0, 0, 0};
	size_t i = 0;
	for (; i + 8 <= n; i += 8)
		for (size_t k = 0; k < 8; k++) a[k] += g[i + k] * (x[i + k] - mu);
	float t = ((a[0] + a[1]) + (a[2] + a[3])) + ((a[4] + a[5]) + (a[6] + a[7]));
	for (; i < n; i++) t += g[i] * (x[i] - mu);
	return t;
}

/**
 * lane_sum of g where alive is nonzero and of 0 elsewhere, the same sum as over a masked copy of g
 *
 * Parameters:
 * const float* g: Elements
 * const unsigned char* alive: One flag per element
 * size_t n: Element count
 *
 * Returns float: The sum
 */
static float lane_sum_masked(const float* g, const unsigned char* alive, size_t n) {
	float a[8] = {0, 0, 0, 0, 0, 0, 0, 0};
	size_t i = 0;
	for (; i + 8 <= n; i += 8)
		for (size_t k = 0; k < 8; k++) a[k] += alive[i + k] ? g[i + k] : 0.0f;
	float t = ((a[0] + a[1]) + (a[2] + a[3])) + ((a[4] + a[5]) + (a[6] + a[7]));
	for (; i < n; i++) t += alive[i] ? g[i] : 0.0f;
	return t;
}

/**
 * lane_dot with g masked as in lane_sum_masked
 *
 * Parameters:
 * const float* g: Gradient elements
 * const unsigned char* alive: One flag per element
 * const float* x: Input elements
 * size_t n: Element count
 * float mu: Value subtracted from each input element
 *
 * Returns float: The sum
 */
static float lane_dot_masked(const float* g, const unsigned char* alive, const float* x, size_t n, float mu) {
	float a[8] = {0, 0, 0, 0, 0, 0, 0, 0};
	size_t i = 0;
	for (; i + 8 <= n; i += 8)
		for (size_t k = 0; k < 8; k++) a[k] += (alive[i + k] ? g[i + k] : 0.0f) * (x[i + k] - mu);
	float t = ((a[0] + a[1]) + (a[2] + a[3])) + ((a[4] + a[5]) + (a[6] + a[7]));
	for (; i < n; i++) t += (alive[i] ? g[i] : 0.0f) * (x[i] - mu);
	return t;
}

Value batchnorm(const Value& x, const Value& weight, const Value& bias, float eps, Tensor* mean_out, Tensor* var_out,
                bool fuse_relu) {
	const std::vector<size_t>& shape = x.data().shape();
	if (shape.size() < 2) throw TensorShapeError();
	const size_t N = shape[0], C = shape[1];
	size_t S = 1;
	for (size_t i = 2; i < shape.size(); i++) S *= shape[i];
	if (weight.data().size() != C || bias.data().size() != C) throw TensorShapeError();
	const size_t M = N * S;

	Tensor result = Tensor::empty(shape);
	std::vector<float> mean(C), var(C), inv_std(C);
	const float* xd = x.data().data().data();
	const float* w = weight.data().data().data();
	const float* b = bias.data().data().data();
	float* yd = result.data().data();
	float* mp = mean.data();
	float* vp = var.data();
	float* ip = inv_std.data();
	std::shared_ptr<unsigned char> alive;
	if (fuse_relu) alive = alloc_alive(result.size());
	unsigned char* flags = alive.get();
	// One thread owns a whole channel, so each sum runs in a fixed order for any thread count
	kernels::parallel_range(C, 1, [=](size_t lo, size_t hi) {
		for (size_t c = lo; c < hi; c++) {
			double sum = 0.0;
			for (size_t n = 0; n < N; n++) sum += lane_sum(xd + (n * C + c) * S, S);
			const float mu = static_cast<float>(sum / static_cast<double>(M));
			double sq = 0.0;
			for (size_t n = 0; n < N; n++) sq += lane_sq(xd + (n * C + c) * S, S, mu);
			const float v = static_cast<float>(sq / static_cast<double>(M));
			const float inv = 1.0f / std::sqrt(v + eps);
			mp[c] = mu;
			vp[c] = v;
			ip[c] = inv;
			const float scale = w[c] * inv;
			for (size_t n = 0; n < N; n++) {
				const float* p = xd + (n * C + c) * S;
				float* q = yd + (n * C + c) * S;
				if (flags) {
					unsigned char* f = flags + (n * C + c) * S;
					for (size_t s = 0; s < S; s++) {
						const float v = (p[s] - mu) * scale + b[c];
						f[s] = v > 0.0f;
						q[s] = v > 0.0f ? v : 0.0f;
					}
				} else {
					for (size_t s = 0; s < S; s++) q[s] = (p[s] - mu) * scale + b[c];
				}
			}
		}
	});
	if (mean_out) *mean_out = Tensor({C}, mean);
	if (var_out) *var_out = Tensor({C}, var);

	return Value::node(std::move(result), {x, weight, bias},
		[x, weight, bias, mean, inv_std, alive, N, C, S, M](const Tensor& grad_out) {
		if (x.data().shape() != grad_out.shape()) throw TensorShapeError();
		const float* xd = x.data().data().data();
		// A folded ReLU zeroes the gradient where the output was clipped. The flags are applied
		// inside the passes below rather than by building a masked copy of grad_out first.
		const float* g = grad_out.data().data();
		const unsigned char* fl = alive.get();
		const float* w = weight.data().data().data();
		const float* mp = mean.data();
		const float* ip = inv_std.data();
		bool first = false;
		float* dx = x.requires_grad() ? x.grad_accum(first).data().data() : nullptr;
		float* dw = weight.requires_grad() ? weight.grad().data().data() : nullptr;
		float* db = bias.requires_grad() ? bias.grad().data().data() : nullptr;
		kernels::parallel_range(C, 1, [=](size_t lo, size_t hi) {
			for (size_t c = lo; c < hi; c++) {
				const float mu = mp[c], inv = ip[c];
				double sdy = 0.0, sdg = 0.0;
				for (size_t n = 0; n < N; n++) {
					const size_t base = (n * C + c) * S;
					sdy += fl ? lane_sum_masked(g + base, fl + base, S) : lane_sum(g + base, S);
					sdg += fl ? lane_dot_masked(g + base, fl + base, xd + base, S, mu) : lane_dot(g + base, xd + base, S, mu);
				}
				const double sdyx = sdg * inv;
				if (db) db[c] += static_cast<float>(sdy);
				if (dw) dw[c] += static_cast<float>(sdyx);
				if (dx) {
					const float k = w[c] * inv / static_cast<float>(M);
					const float fm = static_cast<float>(M), a = static_cast<float>(sdy), bb = static_cast<float>(sdyx);
					for (size_t n = 0; n < N; n++) {
						const size_t base = (n * C + c) * S;
						if (fl) {
							accumulate_range(dx, first, base, base + S, [=](size_t i) {
								return k * (fm * (fl[i] ? g[i] : 0.0f) - a - ((xd[i] - mu) * inv) * bb);
							});
						} else {
							accumulate_range(dx, first, base, base + S, [=](size_t i) {
								return k * (fm * g[i] - a - ((xd[i] - mu) * inv) * bb);
							});
						}
					}
				}
			}
		});
	});
}

Value layernorm(const Value& x, const Value& gamma, const Value& beta, float eps) {
	const std::vector<size_t>& shape = x.data().shape();
	if (shape.empty() || gamma.data().size() != shape.back() || beta.data().size() != shape.back()) throw TensorShapeError();
	const size_t D = shape.back(), R = x.data().size() / D;
	Tensor result = Tensor::empty(shape);
	std::vector<float> mean(R), inv_std(R);
	const float* in = x.data().data().data();
	const float* gm = gamma.data().data().data();
	const float* bt = beta.data().data().data();
	float* out = result.data().data();
	float* mp = mean.data();
	float* ip = inv_std.data();
	const size_t grain = std::max<size_t>(1, kElementwiseChunk / D);
	// One thread owns a whole row, so every sum runs in a fixed order for any thread count
	kernels::parallel_range(R, grain, [=](size_t lo, size_t hi) {
		for (size_t r = lo; r < hi; r++) {
			const float* row = in + r * D;
			float sum = 0.0f;
			for (size_t j = 0; j < D; j++) sum += row[j];
			const float mu = sum / static_cast<float>(D);
			float sq = 0.0f;
			for (size_t j = 0; j < D; j++) sq += (row[j] - mu) * (row[j] - mu);
			const float inv = 1.0f / std::sqrt(sq / static_cast<float>(D) + eps);
			mp[r] = mu;
			ip[r] = inv;
			float* o = out + r * D;
			for (size_t j = 0; j < D; j++) o[j] = (row[j] - mu) * inv * gm[j] + bt[j];
		}
	});
	return Value::node(std::move(result), {x, gamma, beta},
		[x, gamma, beta, mean, inv_std, D, R, grain](const Tensor& grad_out) {
		if (x.data().shape() != grad_out.shape()) throw TensorShapeError();
		const float* in = x.data().data().data();
		const float* g = grad_out.data().data();
		const float* gm = gamma.data().data().data();
		const float* mp = mean.data();
		const float* ip = inv_std.data();
		if (x.requires_grad()) {
			bool first;
			float* dx = x.grad_accum(first).data().data();
			kernels::parallel_range(R, grain, [=](size_t lo, size_t hi) {
				for (size_t r = lo; r < hi; r++) {
					const float mu = mp[r], inv = ip[r];
					float a = 0.0f, b = 0.0f;
					for (size_t j = 0; j < D; j++) {
						const float dxhat = g[r * D + j] * gm[j];
						a += dxhat;
						b += dxhat * ((in[r * D + j] - mu) * inv);
					}
					const float k = inv / static_cast<float>(D), fd = static_cast<float>(D);
					accumulate_range(dx + r * D, first, 0, D, [=](size_t j) {
						const float dxhat = g[r * D + j] * gm[j];
						return k * (fd * dxhat - a - ((in[r * D + j] - mu) * inv) * b);
					});
				}
			});
		}
		if (gamma.requires_grad() || beta.requires_grad()) {
			float* dg = gamma.requires_grad() ? gamma.grad().data().data() : nullptr;
			float* db = beta.requires_grad() ? beta.grad().data().data() : nullptr;
			// Parallel over columns: each column sums its rows in order, so the result does not depend on threads
			kernels::parallel_range(D, std::max<size_t>(1, kElementwiseChunk / R), [=](size_t lo, size_t hi) {
				for (size_t j = lo; j < hi; j++) {
					float sg = 0.0f, sb = 0.0f;
					for (size_t r = 0; r < R; r++) {
						const float gv = g[r * D + j];
						sg += gv * ((in[r * D + j] - mp[r]) * ip[r]);
						sb += gv;
					}
					if (dg) dg[j] += sg;
					if (db) db[j] += sb;
				}
			});
		}
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
	const std::vector<size_t>& shape = x.data().shape();
	if (axis >= 0 && static_cast<size_t>(axis) + 1 == shape.size() && shape.back() > 0) {
		// Last axis: one pass per row, no temporaries; rows are independent so threads never share a write
		const size_t D = shape.back(), R = x.data().size() / D;
		Tensor result = Tensor::empty(shape);
		const float* in = x.data().data().data();
		float* out = result.data().data();
		const size_t grain = std::max<size_t>(1, kElementwiseChunk / D);
		kernels::parallel_range(R, grain, [=](size_t lo, size_t hi) {
			for (size_t r = lo; r < hi; r++) {
				const float* row = in + r * D;
				float* o = out + r * D;
				float m = row[0];
				for (size_t j = 1; j < D; j++) m = std::max(m, row[j]);
				float sum = 0.0f;
				for (size_t j = 0; j < D; j++) {
					o[j] = std::exp(row[j] - m);
					sum += o[j];
				}
				for (size_t j = 0; j < D; j++) o[j] /= sum;
			}
		});
		Tensor saved = result;
		return Value::node(std::move(result), {x}, [x, saved = std::move(saved), D, R, grain](const Tensor& grad_out) {
			if (!x.requires_grad()) return;
			if (x.data().shape() != grad_out.shape()) throw TensorShapeError();
			const float* y = saved.data().data();
			const float* g = grad_out.data().data();
			bool first;
			float* dx = x.grad_accum(first).data().data();
			kernels::parallel_range(R, grain, [=](size_t lo, size_t hi) {
				for (size_t r = lo; r < hi; r++) {
					const float* yr = y + r * D;
					const float* gr = g + r * D;
					float dot = 0.0f;
					for (size_t j = 0; j < D; j++) dot += gr[j] * yr[j];
					accumulate_range(dx + r * D, first, 0, D, [=](size_t j) { return yr[j] * (gr[j] - dot); });
				}
			});
		});
	}
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

/**
 * One row of the scaled, masked softmax: scale, padding at -1e9, causal at -2e9, then softmax
 *
 * Parameters:
 * const float* row: Tk raw scores; may be the same array as o
 * float* o: Tk probabilities out
 * size_t Tk: Row length
 * size_t i: Query position of the row, for the causal mask
 * const float* ph: The sample's Tk padding flags (0 marks a padded key), or nullptr
 * bool causal: Hide keys after position i
 * float scale_by: Factor applied to every score first
 *
 * Returns void
 */
static inline void masked_softmax_row(const float* row, float* o, size_t Tk, size_t i, const float* ph, bool causal,
	float scale_by) {
	for (size_t j = 0; j < Tk; j++) {
		float v = row[j] * scale_by;
		if (ph && ph[j] == 0.0f) v = -1e9f;
		if (causal && j > i) v = -2e9f;
		o[j] = v;
	}
	float m = o[0];
	for (size_t j = 1; j < Tk; j++) m = std::max(m, o[j]);
	// Row i always sees key 0 at -1e9 or above, so its maximum is at least that and a causally
	// hidden key at -2e9 has exp(-2e9 - m) == 0 exactly: write the zero instead of taking
	// expf through its underflow path
	const size_t live = causal ? std::min(Tk, i + 1) : Tk;
	float sum = 0.0f;
	for (size_t j = 0; j < live; j++) {
		o[j] = std::exp(o[j] - m);
		sum += o[j];
	}
	for (size_t j = live; j < Tk; j++) o[j] = 0.0f;
	for (size_t j = 0; j < Tk; j++) o[j] /= sum;
}

/**
 * Backward of masked_softmax_row: turns dL/d(probabilities) into dL/d(raw scores), in place
 *
 * Parameters:
 * const float* y: The row's Tk probabilities
 * float* g: In, dL/dy; out, dL/d(raw score), 0 at masked positions
 * size_t Tk, size_t i, const float* ph, bool causal, float scale_by: as masked_softmax_row
 *
 * Returns void
 */
static inline void masked_softmax_row_backward(const float* y, float* g, size_t Tk, size_t i, const float* ph,
	bool causal, float scale_by) {
	float dot = 0.0f;
	for (size_t j = 0; j < Tk; j++) dot += g[j] * y[j];
	for (size_t j = 0; j < Tk; j++) {
		const bool masked = (ph && ph[j] == 0.0f) || (causal && j > i);
		g[j] = masked ? 0.0f : scale_by * (y[j] * (g[j] - dot)) + 0.0f;
	}
}

Value attention(const Value& q, const Value& k, const Value& v, size_t heads, const Tensor& key_padding_mask,
	bool causal) {
	const std::vector<size_t>& qs = q.data().shape();
	const std::vector<size_t>& ks = k.data().shape();
	if (qs.size() != 3 || ks.size() != 3 || v.data().shape() != ks || qs[0] != ks[0] || qs[2] != ks[2]) throw TensorShapeError();
	const size_t N = qs[0], Tq = qs[1], Tk = ks[1], D = qs[2];
	if (heads == 0 || D % heads != 0 || Tq == 0 || Tk == 0 || (causal && Tq != Tk)) throw TensorShapeError();
	const size_t H = heads, Dh = D / H;
	const bool padded = key_padding_mask.size() != 0;
	if (padded && key_padding_mask.shape() != std::vector<size_t>{N, Tk}) throw TensorShapeError();
	const float scale_by = 1.0f / std::sqrt(static_cast<float>(Dh));

	Tensor result = Tensor::empty({N, Tq, D});
	Tensor probs = Tensor::empty({N, H, Tq, Tk});  // kept for the backward pass
	// One unit of work per sample when there is a sample per thread (a head's columns share cache
	// lines with its neighbours'), otherwise per sample-head pair
	const bool by_sample = N >= kernels::num_threads();
	const size_t units = by_sample ? N : N * H, heads_per_unit = by_sample ? H : 1;
	{
		const float* qd = q.data().data().data();
		const float* kd = k.data().data().data();
		const float* vd = v.data().data().data();
		const float* pad = padded ? key_padding_mask.data().data() : nullptr;
		float* out = result.data().data();
		float* pr = probs.data().data();
		kernels::parallel_range(units, 1, [=](size_t lo, size_t hi) {
			for (size_t nh = lo * heads_per_unit; nh < hi * heads_per_unit; nh++) {
				const size_t n = nh / H, h = nh % H;
				float* p = pr + nh * Tq * Tk;
				// Head h of a {T, D} sample is the matrix starting at column h * Dh with row stride D
				kernels::sgemm(false, true, Tq, Tk, Dh, qd + n * Tq * D + h * Dh, D, kd + n * Tk * D + h * Dh, D, p, Tk, false);
				for (size_t i = 0; i < Tq; i++)
					masked_softmax_row(p + i * Tk, p + i * Tk, Tk, i, pad ? pad + n * Tk : nullptr, causal, scale_by);
				kernels::sgemm(false, false, Tq, Dh, Tk, p, Tk, vd + n * Tk * D + h * Dh, D, out + n * Tq * D + h * Dh, D, false);
			}
		});
	}
	return Value::node(std::move(result), {q, k, v},
		[q, k, v, probs = std::move(probs), key_padding_mask, causal, scale_by, N, H, Tq, Tk, D, Dh, units, heads_per_unit](
			const Tensor& grad_out) {
		if (grad_out.shape() != std::vector<size_t>{N, Tq, D}) throw TensorShapeError();
		const float* g = grad_out.data().data();
		const float* qd = q.data().data().data();
		const float* kd = k.data().data().data();
		const float* vd = v.data().data().data();
		const float* pr = probs.data().data();
		const float* pad = key_padding_mask.size() != 0 ? key_padding_mask.data().data() : nullptr;
		// Fetched in the order the blocks below write them, so one node passed as two inputs is
		// stored into by its first use and added to by the next
		bool first_v = false, first_q = false, first_k = false;
		float* dv = v.requires_grad() ? v.grad_accum(first_v).data().data() : nullptr;
		float* dq = q.requires_grad() ? q.grad_accum(first_q).data().data() : nullptr;
		float* dk = k.requires_grad() ? k.grad_accum(first_k).data().data() : nullptr;
		kernels::parallel_range(units, 1, [=](size_t lo, size_t hi) {
			std::vector<float> ds(Tq * Tk);
			for (size_t nh = lo * heads_per_unit; nh < hi * heads_per_unit; nh++) {
				const size_t n = nh / H, h = nh % H;
				const float* p = pr + nh * Tq * Tk;
				const float* gh = g + n * Tq * D + h * Dh;
				// dL/dP = G V^T, then through the softmax to dL/dS; dV = P^T G, dQ = dS K, dK = dS^T Q
				kernels::sgemm(false, true, Tq, Tk, Dh, gh, D, vd + n * Tk * D + h * Dh, D, ds.data(), Tk, false);
				if (dv) kernels::sgemm(true, false, Tk, Dh, Tq, p, Tk, gh, D, dv + n * Tk * D + h * Dh, D, !first_v);
				if (!dq && !dk) continue;
				for (size_t i = 0; i < Tq; i++)
					masked_softmax_row_backward(p + i * Tk, ds.data() + i * Tk, Tk, i, pad ? pad + n * Tk : nullptr, causal, scale_by);
				if (dq) kernels::sgemm(false, false, Tq, Dh, Tk, ds.data(), Tk, kd + n * Tk * D + h * Dh, D, dq + n * Tq * D + h * Dh, D, !first_q);
				if (dk) kernels::sgemm(true, false, Tk, Dh, Tq, ds.data(), Tk, qd + n * Tq * D + h * Dh, D, dk + n * Tk * D + h * Dh, D, !first_k);
			}
		});
	});
}

Value scaled_masked_softmax(const Value& x, float scale_by, const Tensor& key_padding_mask, bool causal) {
	const std::vector<size_t>& shape = x.data().shape();
	if (shape.size() != 4 || shape[3] == 0) throw TensorShapeError();
	const size_t N = shape[0], H = shape[1], Tq = shape[2], Tk = shape[3];
	const bool padded = key_padding_mask.size() != 0;
	if (padded && key_padding_mask.shape() != std::vector<size_t>{N, Tk}) throw TensorShapeError();
	const size_t R = N * H * Tq;
	const size_t grain = std::max<size_t>(1, kElementwiseChunk / Tk);
	Tensor result = Tensor::empty(shape);
	const float* in = x.data().data().data();
	const float* pad = padded ? key_padding_mask.data().data() : nullptr;
	float* out = result.data().data();
	// Rows are independent and each is computed in the same order as scale, masked_fill and softmax were
	kernels::parallel_range(R, grain, [=](size_t lo, size_t hi) {
		for (size_t r = lo; r < hi; r++) {
			const size_t i = r % Tq, n = r / (H * Tq);
			masked_softmax_row(in + r * Tk, out + r * Tk, Tk, i, pad ? pad + n * Tk : nullptr, causal, scale_by);
		}
	});
	Tensor saved = result;
	return Value::node(std::move(result), {x},
		[x, saved = std::move(saved), key_padding_mask, causal, scale_by, H, Tq, Tk, R, grain](const Tensor& grad_out) {
		if (!x.requires_grad()) return;
		if (x.data().shape() != grad_out.shape()) throw TensorShapeError();
		const float* y = saved.data().data();
		const float* g = grad_out.data().data();
		const float* pad = key_padding_mask.size() != 0 ? key_padding_mask.data().data() : nullptr;
		bool first;
		float* dx = x.grad_accum(first).data().data();
		kernels::parallel_range(R, grain, [=](size_t lo, size_t hi) {
			for (size_t r = lo; r < hi; r++) {
				const size_t i = r % Tq, n = r / (H * Tq);
				const float* ph = pad ? pad + n * Tk : nullptr;
				float dot = 0.0f;
				for (size_t j = 0; j < Tk; j++) dot += g[r * Tk + j] * y[r * Tk + j];
				// A masked position was overwritten by a constant, so no gradient reaches the scores there
				for (size_t j = 0; j < Tk; j++) {
					const bool masked = (ph && ph[j] == 0.0f) || (causal && j > i);
					const float term = scale_by * (y[r * Tk + j] * (g[r * Tk + j] - dot));
					if (first) dx[r * Tk + j] = masked ? 0.0f : term + 0.0f;
					else if (!masked) dx[r * Tk + j] += term;
				}
			}
		});
	});
}

static Value conv2d_impl(const Value& x, const Value& weight, const Value& bias, int stride, int padding,
                         bool fuse_relu) {
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

	// Each sample is unrolled into a column buffer of its own thread, multiplied and finished
	// (bias, ReLU) while that buffer and the output are still in cache. One big column tensor for
	// the whole batch would be written, read back by the matmul and kept for the backward pass,
	// all of it through main memory: 100 MB per step on a 64 x 3 x 64 x 64 batch.
	const ConvGeometry geo{Cin, H, W, kh, kw, static_cast<size_t>(stride), static_cast<size_t>(padding), h_out, w_out};
	const size_t K = Cin * kh * kw, hw = h_out * w_out;
	Tensor result = Tensor::empty({N, Cout, h_out, w_out});
	std::shared_ptr<unsigned char> alive;
	if (fuse_relu) alive = alloc_alive(result.size());
	// With a sample per thread the samples run in parallel and each matmul is serial; a smaller
	// batch runs its samples in turn and lets each matmul use the pool. Either way a sample's
	// output comes from the same serial tile sequence.
	const bool by_sample = N >= kernels::num_threads();
	{
		const float* xd = x.data().data().data();
		const float* wd = weight.data().data().data();
		const float* bias_data = bias.data().data().data();
		float* yd = result.data().data();
		unsigned char* flags = alive.get();
		auto run = [=](size_t n_lo, size_t n_hi) {
			Storage col(K * hw);
			for (size_t n = n_lo; n < n_hi; n++) {
				im2col_sample(xd + n * Cin * H * W, col.data(), geo);
				float* y = yd + n * Cout * hw;
				kernels::sgemm(false, false, Cout, hw, K, wd, K, col.data(), hw, y, hw, false);
				for (size_t c = 0; c < Cout; c++) {
					const float bias_val = bias_data[c];
					float* row = y + c * hw;
					if (flags) {
						unsigned char* frow = flags + (n * Cout + c) * hw;
						for (size_t i = 0; i < hw; i++) {
							const float v = row[i] + bias_val;
							frow[i] = v > 0.0f;
							row[i] = v > 0.0f ? v : 0.0f;
						}
					} else {
						for (size_t i = 0; i < hw; i++) row[i] += bias_val;
					}
				}
			}
		};
		if (by_sample) kernels::parallel_range(N, 1, run);
		else run(0, N);
	}

	return Value::node(
		std::move(result), {x, weight, bias},
		[x, weight, bias, alive, geo, N, Cout, Cin, kh, kw, K, hw, by_sample](const Tensor& grad_out) {
			if (grad_out.size() != N * Cout * hw) throw TensorShapeError();
			const bool need_b = bias.requires_grad(), need_w = weight.requires_grad(), need_x = x.requires_grad();
			const float* go = grad_out.data().data();
			const float* xd = x.data().data().data();
			const float* wd = weight.data().data().data();
			const unsigned char* flags = alive.get();
			// Per-sample weight and bias gradients, reduced in sample order below so the result
			// does not depend on which thread ran which sample
			Tensor gw = Tensor::empty({N, need_w ? Cout * K : 0});
			Tensor gb = Tensor::empty({N, need_b ? Cout : 0});
			float* gwd = gw.data().data();
			float* gbd = gb.data().data();
			bool first = false;
			float* dx = need_x ? x.grad_accum(first).data().data() : nullptr;
			const size_t x_sample = Cin * geo.H * geo.W;
			auto run = [=](size_t n_lo, size_t n_hi) {
				Storage col(K * hw), masked(flags ? Cout * hw : 0);
				for (size_t n = n_lo; n < n_hi; n++) {
					// A folded ReLU zeroes the gradient where the output was clipped
					const float* g = go + n * Cout * hw;
					if (flags) {
						const unsigned char* f = flags + n * Cout * hw;
						for (size_t i = 0; i < Cout * hw; i++) masked[i] = f[i] ? g[i] : 0.0f;
						g = masked.data();
					}
					if (need_b)
						for (size_t c = 0; c < Cout; c++) gbd[n * Cout + c] = lane_sum(g + c * hw, hw);
					if (need_w) {
						// The columns are rebuilt rather than kept from the forward pass
						im2col_sample(xd + n * x_sample, col.data(), geo);
						kernels::sgemm(false, true, Cout, K, hw, g, hw, col.data(), hw, gwd + n * Cout * K, K, false);
					}
					if (need_x) {
						kernels::sgemm(true, false, K, hw, Cout, wd, K, g, hw, col.data(), hw, false);
						float* dxn = dx + n * x_sample;
						if (first) std::fill(dxn, dxn + x_sample, 0.0f);
						col2im_sample_add(col.data(), dxn, geo);
					}
				}
			};
			if (by_sample) kernels::parallel_range(N, 1, run);
			else run(0, N);
			if (need_b) bias.grad() += gb.sum(0);
			if (need_w) weight.grad() += gw.sum(0).reshape({Cout, Cin, kh, kw});
		});
}

Value conv2d(const Value& x, const Value& weight, const Value& bias, int stride, int padding) {
	return conv2d_impl(x, weight, bias, stride, padding, false);
}

Value conv2d_relu(const Value& x, const Value& weight, const Value& bias, int stride, int padding) {
	return conv2d_impl(x, weight, bias, stride, padding, true);
}

namespace {
/**
 * Fills the im2col matrix of one group of one sample, honouring dilation
 *
 * Parameters:
 * const float* x: Start of the group's Cg input channels, each H x W
 * float* col: Output, (Cg * kh * kw) x (h_out * w_out), row-major
 * size_t Cg, H, W, kh, kw, h_out, w_out: Geometry
 * size_t stride, padding, dilation: Convolution parameters
 *
 * Returns void
 */
void group_im2col(const float* x, float* col, size_t Cg, size_t H, size_t W, size_t kh, size_t kw, size_t h_out,
	size_t w_out, size_t stride, size_t padding, size_t dilation) {
	for (size_t c = 0; c < Cg; c++)
		for (size_t i = 0; i < kh; i++)
			for (size_t j = 0; j < kw; j++) {
				float* row = col + ((c * kh + i) * kw + j) * h_out * w_out;
				for (size_t oh = 0; oh < h_out; oh++)
					for (size_t ow = 0; ow < w_out; ow++) {
						// Unsigned wrap-around turns negative (padding) positions into huge values the bounds test rejects
						size_t ih = oh * stride + i * dilation - padding;
						size_t iw = ow * stride + j * dilation - padding;
						row[oh * w_out + ow] = (ih < H && iw < W) ? x[(c * H + ih) * W + iw] : 0.0f;
					}
			}
}

/**
 * Adds the gradient of one group's im2col matrix back into the input, the transpose of group_im2col
 *
 * Parameters: as group_im2col, with const float* col read and float* x accumulated into
 *
 * Returns void
 */
void group_col2im(const float* col, float* x, size_t Cg, size_t H, size_t W, size_t kh, size_t kw, size_t h_out,
	size_t w_out, size_t stride, size_t padding, size_t dilation) {
	for (size_t c = 0; c < Cg; c++)
		for (size_t i = 0; i < kh; i++)
			for (size_t j = 0; j < kw; j++) {
				const float* row = col + ((c * kh + i) * kw + j) * h_out * w_out;
				for (size_t oh = 0; oh < h_out; oh++)
					for (size_t ow = 0; ow < w_out; ow++) {
						size_t ih = oh * stride + i * dilation - padding;
						size_t iw = ow * stride + j * dilation - padding;
						if (ih < H && iw < W) x[(c * H + ih) * W + iw] += row[oh * w_out + ow];
					}
			}
}
}  // namespace

Value conv2d(const Value& x, const Value& weight, const Value& bias, int stride, int padding, int dilation,
	int groups) {
	if (dilation == 1 && groups == 1) return conv2d(x, weight, bias, stride, padding);
	if (x.data().ndim() != 4 || weight.data().ndim() != 4) throw TensorShapeError();
	if (stride <= 0 || padding < 0 || dilation <= 0 || groups <= 0) throw TensorShapeError();
	size_t G = static_cast<size_t>(groups), D = static_cast<size_t>(dilation);
	size_t S = static_cast<size_t>(stride), P = static_cast<size_t>(padding);
	size_t Cout = weight.data().shape()[0], Cg = weight.data().shape()[1];
	size_t kh = weight.data().shape()[2], kw = weight.data().shape()[3];
	size_t N = x.data().shape()[0], Cin = x.data().shape()[1];
	size_t H = x.data().shape()[2], W = x.data().shape()[3];
	if (Cin % G != 0 || Cout % G != 0 || Cg != Cin / G || bias.data().size() != Cout || kh == 0 || kw == 0) {
		throw TensorShapeError();
	}
	size_t ekh = D * (kh - 1) + 1, ekw = D * (kw - 1) + 1;
	if (ekh > H + 2 * P || ekw > W + 2 * P) throw TensorShapeError();
	size_t h_out = (H + 2 * P - ekh) / S + 1, w_out = (W + 2 * P - ekw) / S + 1;
	size_t hw = h_out * w_out, K = Cg * kh * kw, Cog = Cout / G;
	size_t x_sample = Cin * H * W, y_sample = Cout * hw, w_size = Cout * K;

	Tensor result({N, Cout, h_out, w_out}, 0.0f);
	{
		const float* xd = x.data().data().data();
		const float* wd = weight.data().data().data();
		const float* bd = bias.data().data().data();
		float* yd = result.data().data();
		kernels::parallel_range(N, 1, [=](size_t lo, size_t hi) {
			std::vector<float> col(K * hw);
			for (size_t n = lo; n < hi; n++)
				for (size_t g = 0; g < G; g++) {
					group_im2col(xd + n * x_sample + g * Cg * H * W, col.data(), Cg, H, W, kh, kw, h_out, w_out, S, P, D);
					float* y = yd + n * y_sample + g * Cog * hw;
					kernels::sgemm(false, false, Cog, hw, K, wd + g * Cog * K, K, col.data(), hw, y, hw, false);
					for (size_t c = 0; c < Cog; c++)
						for (size_t i = 0; i < hw; i++) y[c * hw + i] += bd[g * Cog + c];
				}
		});
	}

	std::vector<size_t> x_shape = x.data().shape();
	return Value::node(std::move(result), {x, weight, bias},
		[=](const Tensor& grad_out) {
			const float* go = grad_out.data().data();
			const float* xd = x.data().data().data();
			const float* wd = weight.data().data().data();
			bool need_x = x.requires_grad(), need_w = weight.requires_grad();
			Tensor grad_x(x_shape, 0.0f);
			// Per-sample weight gradients, summed in sample order below so the result is thread-count independent
			std::vector<float> gw(need_w ? N * w_size : 0);
			float* gxd = grad_x.data().data();
			float* gwd = gw.data();
			kernels::parallel_range(N, 1, [=](size_t lo, size_t hi) {
				std::vector<float> col(K * hw);
				for (size_t n = lo; n < hi; n++)
					for (size_t g = 0; g < G; g++) {
						const float* gy = go + n * y_sample + g * Cog * hw;
						if (need_w) {
							group_im2col(xd + n * x_sample + g * Cg * H * W, col.data(), Cg, H, W, kh, kw, h_out, w_out, S, P, D);
							kernels::sgemm(false, true, Cog, K, hw, gy, hw, col.data(), hw,
								gwd + n * w_size + g * Cog * K, K, false);
						}
						if (need_x) {
							kernels::sgemm(true, false, K, hw, Cog, wd + g * Cog * K, K, gy, hw, col.data(), hw, false);
							group_col2im(col.data(), gxd + n * x_sample + g * Cg * H * W, Cg, H, W, kh, kw, h_out, w_out, S, P, D);
						}
					}
			});
			if (bias.requires_grad()) {
				Tensor grad_bias({Cout}, 0.0f);
				for (size_t n = 0; n < N; n++)
					for (size_t c = 0; c < Cout; c++)
						for (size_t i = 0; i < hw; i++) grad_bias.data()[c] += go[n * y_sample + c * hw + i];
				bias.grad() += grad_bias;
			}
			if (need_w) {
				Tensor grad_weight(weight.data().shape(), 0.0f);
				for (size_t n = 0; n < N; n++)
					for (size_t i = 0; i < w_size; i++) grad_weight.data()[i] += gw[n * w_size + i];
				weight.grad() += grad_weight;
			}
			if (need_x) x.grad() += grad_x;
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

	// The argmax is kept as an offset inside its own plane, which fits 32 bits
	if (H * W > std::numeric_limits<uint32_t>::max()) throw TensorShapeError();
	Tensor result = Tensor::empty({N, C, h_out, w_out});  // every output cell is written below
	std::shared_ptr<uint32_t> argmax(new uint32_t[N * C * h_out * w_out], std::default_delete<uint32_t[]>());

	const float* x_data = x.data().data().data();
	float* result_data = result.data().data();
	uint32_t* argmax_data = argmax.get();
	size_t out_plane = h_out * w_out;
	const size_t plane_size = H * W;

	// Each (n, c) plane reads and writes only its own slices, so planes can run on separate threads
	size_t chunk = std::max<size_t>(1, kElementwiseChunk / std::max<size_t>(1, out_plane * kernel_size * kernel_size));
	kernels::parallel_range(N * C, chunk, [=](size_t p_lo, size_t p_hi) {
		const float ninf = -std::numeric_limits<float>::infinity();
		for (size_t p = p_lo; p < p_hi; p++) {
			const float* plane = x_data + p * plane_size;
			float* out = result_data + p * out_plane;
			uint32_t* arg = argmax_data + p * out_plane;
			if (kernel_size == 2 && stride == 2) {
				// The common window, unrolled: the same four comparisons in the same order as the
				// general loop (first maximum wins, offset 0 when nothing beats -inf), without its loops
				for (size_t oh = 0; oh < h_out; oh++) {
					const float* r0 = plane + 2 * oh * W;
					const float* r1 = r0 + W;
					const uint32_t base = static_cast<uint32_t>(2 * oh * W);
					for (size_t ow = 0; ow < w_out; ow++) {
						const uint32_t o0 = base + static_cast<uint32_t>(2 * ow);
						float best = ninf;
						uint32_t at = 0;
						if (r0[2 * ow] > best) { best = r0[2 * ow]; at = o0; }
						if (r0[2 * ow + 1] > best) { best = r0[2 * ow + 1]; at = o0 + 1; }
						if (r1[2 * ow] > best) { best = r1[2 * ow]; at = o0 + static_cast<uint32_t>(W); }
						if (r1[2 * ow + 1] > best) { best = r1[2 * ow + 1]; at = o0 + static_cast<uint32_t>(W) + 1; }
						out[oh * w_out + ow] = best;
						arg[oh * w_out + ow] = at;
					}
				}
				continue;
			}
			for (size_t oh = 0; oh < h_out; oh++) {
				for (size_t ow = 0; ow < w_out; ow++) {
					float best = ninf;
					uint32_t at = 0;
					for (size_t ph = 0; ph < kernel_size; ph++) {
						const float* row = plane + (oh * stride + ph) * W + ow * stride;
						for (size_t pw = 0; pw < kernel_size; pw++) {
							if (row[pw] > best) {
								best = row[pw];
								at = static_cast<uint32_t>((oh * stride + ph) * W + ow * stride + pw);
							}
						}
					}
					out[oh * w_out + ow] = best;
					arg[oh * w_out + ow] = at;
				}
			}
		}
	});

	return Value::node(std::move(result), {x}, [x, argmax = std::move(argmax), N, C, out_plane, plane_size](const Tensor& grad_out) {
		if (!x.requires_grad()) return;
		if (!x.has_grad()) throw TensorShapeError();
		const float* g = grad_out.data().data();
		bool first;
		float* dst = x.grad_accum(first).data().data();
		const uint32_t* arg = argmax.get();
		// Every argmax entry of plane p points inside plane p, so planes write disjoint ranges
		size_t chunk = std::max<size_t>(1, kElementwiseChunk / std::max<size_t>(1, out_plane));
		kernels::parallel_range(N * C, chunk, [=](size_t p_lo, size_t p_hi) {
			for (size_t p = p_lo; p < p_hi; p++) {
				float* plane = dst + p * plane_size;
				// A gradient holding nothing yet is zeroed plane by plane, right before the scatter into it
				if (first) std::fill(plane, plane + plane_size, 0.0f);
				for (size_t i = p * out_plane; i < (p + 1) * out_plane; i++) plane[arg[i]] += g[i];
			}
		});
	});
}

Value avgpool2d(const Value& x, size_t kernel_size, size_t stride) {
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
	const float* x_data = x.data().data().data();
	float* result_data = result.data().data();
	size_t out_plane = h_out * w_out;
	float inv_area = 1.0f / static_cast<float>(kernel_size * kernel_size);

	// Each (n, c) plane reads and writes only its own slices, so planes can run on separate threads
	size_t chunk = std::max<size_t>(1, kElementwiseChunk / std::max<size_t>(1, out_plane * kernel_size * kernel_size));
	kernels::parallel_range(N * C, chunk, [=](size_t p_lo, size_t p_hi) {
		for (size_t p = p_lo; p < p_hi; p++) {
			const float* plane = x_data + p * H * W;
			for (size_t oh = 0; oh < h_out; oh++) {
				for (size_t ow = 0; ow < w_out; ow++) {
					float acc = 0.0f;
					for (size_t ph = 0; ph < kernel_size; ph++) {
						const float* row = plane + (oh * stride + ph) * W + ow * stride;
						for (size_t pw = 0; pw < kernel_size; pw++) acc += row[pw];
					}
					result_data[p * out_plane + oh * w_out + ow] = acc * inv_area;
				}
			}
		}
	});

	return Value::node(std::move(result), {x}, [x, N, C, H, W, h_out, w_out, kernel_size, stride, inv_area](const Tensor& grad_out) {
		if (!x.requires_grad()) return;
		if (x.grad().shape() != x.data().shape()) throw TensorShapeError();
		const float* g = grad_out.data().data();
		float* dst = x.grad().data().data();
		size_t out_plane = h_out * w_out;
		// Plane p only touches its own input and output slices, so planes write disjoint ranges
		size_t chunk = std::max<size_t>(1, kElementwiseChunk / std::max<size_t>(1, out_plane * kernel_size * kernel_size));
		kernels::parallel_range(N * C, chunk, [=](size_t p_lo, size_t p_hi) {
			for (size_t p = p_lo; p < p_hi; p++) {
				float* plane = dst + p * H * W;
				for (size_t oh = 0; oh < h_out; oh++) {
					for (size_t ow = 0; ow < w_out; ow++) {
						float share = g[p * out_plane + oh * w_out + ow] * inv_area;
						for (size_t ph = 0; ph < kernel_size; ph++) {
							float* row = plane + (oh * stride + ph) * W + ow * stride;
							for (size_t pw = 0; pw < kernel_size; pw++) row[pw] += share;
						}
					}
				}
			}
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

	// Validated up front: the rows below run on pool threads, where a throw would terminate
	std::vector<size_t> target_idx(N);
	for (size_t n = 0; n < N; n++) {
		const float t = std::round(targets.data()[n]);
		if (!(t >= 0.0f && t < static_cast<float>(C))) throw TensorIndexError();
		target_idx[n] = static_cast<size_t>(t);
	}

	Tensor softmax_val = Tensor::empty(logits_data.shape());
	std::vector<float> row_loss(N);
	{
		const float* in = logits_data.data().data();
		float* sm_data = softmax_val.data().data();
		const size_t* tg = target_idx.data();
		float* rl = row_loss.data();
		// Rows are independent; each computes max, shift, exp-sum and log in the order the
		// tensor ops it replaces used, with the shifted row held in the softmax buffer
		kernels::parallel_range(N, std::max<size_t>(1, kElementwiseChunk / std::max<size_t>(1, C)), [=](size_t lo, size_t hi) {
			for (size_t n = lo; n < hi; n++) {
				const float* row = in + n * C;
				float* sm = sm_data + n * C;
				float max_val = -std::numeric_limits<float>::infinity();
				for (size_t c = 0; c < C; c++) max_val = std::max(max_val, row[c]);
				float sum_exp = 0.0f;
				for (size_t c = 0; c < C; c++) {
					sm[c] = row[c] - max_val;
					sum_exp += std::exp(sm[c]);
				}
				const float log_sum_exp = std::log(sum_exp);
				rl[n] = -(sm[tg[n]] - log_sum_exp);
				for (size_t c = 0; c < C; c++) sm[c] = std::exp(sm[c] - log_sum_exp);
			}
		});
	}
	float total_loss = 0.0f;  // serial, in row order: independent of thread count
	for (size_t n = 0; n < N; n++) total_loss += row_loss[n];
	total_loss /= static_cast<float>(N);

	Tensor result({1}, total_loss);
	return Value::node(std::move(result), {logits},
		[logits, softmax_val = std::move(softmax_val), target_idx = std::move(target_idx), N, C](const Tensor& grad_out) {
		if (!logits.requires_grad()) return;
		const float grad_scale = grad_out.data()[0] / static_cast<float>(N);
		const float* sm_data = softmax_val.data().data();
		const size_t* tg = target_idx.data();
		bool first;
		float* dst = logits.grad_accum(first).data().data();
		kernels::parallel_range(N, std::max<size_t>(1, kElementwiseChunk / std::max<size_t>(1, C)), [=](size_t lo, size_t hi) {
			for (size_t n = lo; n < hi; n++) {
				const float* sm = sm_data + n * C;
				const size_t target = tg[n];
				accumulate_range(dst + n * C, first, 0, C, [=](size_t c) { return (c == target ? sm[c] - 1.0f : sm[c]) * grad_scale; });
			}
		});
	});
}

Value cross_entropy_loss(const Value& logits, const Tensor& targets, const Tensor& class_weights,
	float label_smoothing) {
	const Tensor& logits_data = logits.data();
	if (logits_data.ndim() != 2) throw TensorShapeError();
	size_t N = logits_data.shape()[0];
	size_t C = logits_data.shape()[1];
	if (targets.ndim() != 1 || targets.size() != N) throw TensorShapeError();
	if (class_weights.ndim() != 1 || class_weights.size() != C) throw TensorShapeError();
	if (!(label_smoothing >= 0.0f && label_smoothing < 1.0f)) {
		throw std::invalid_argument("cross_entropy_loss: label_smoothing must be in [0, 1)");
	}
	for (float w : class_weights.data()) {
		if (!(std::isfinite(w) && w >= 0.0f)) {
			throw std::invalid_argument("cross_entropy_loss: class weights must be finite and >= 0");
		}
	}
	std::vector<size_t> target_idx(N);
	double total_weight = 0.0;
	for (size_t n = 0; n < N; n++) {
		float t = std::round(targets.data()[n]);
		if (!(t >= 0.0f && t < static_cast<float>(C))) {
			throw std::invalid_argument("cross_entropy_loss: target out of range");
		}
		target_idx[n] = static_cast<size_t>(t);
		total_weight += class_weights.data()[target_idx[n]];
	}
	if (!(total_weight > 0.0)) throw std::invalid_argument("cross_entropy_loss: total sample weight is zero");

	Tensor softmax_val(logits_data.shape());
	double total_loss = 0.0;
	const float inv_c = 1.0f / static_cast<float>(C);
	for (size_t n = 0; n < N; n++) {
		const float* row = logits_data.data().data() + n * C;
		float max_val = row[0];
		for (size_t c = 1; c < C; c++) max_val = std::max(max_val, row[c]);
		float sum_exp = 0.0f;
		for (size_t c = 0; c < C; c++) sum_exp += std::exp(row[c] - max_val);
		float log_sum_exp = std::log(sum_exp);
		float mean_nll = 0.0f;
		for (size_t c = 0; c < C; c++) {
			float log_prob = row[c] - max_val - log_sum_exp;
			softmax_val.data()[n * C + c] = std::exp(log_prob);
			mean_nll -= log_prob * inv_c;
		}
		float nll = -(row[target_idx[n]] - max_val - log_sum_exp);
		float l_n = (1.0f - label_smoothing) * nll + label_smoothing * mean_nll;
		total_loss += static_cast<double>(class_weights.data()[target_idx[n]]) * l_n;
	}
	Tensor result({1}, static_cast<float>(total_loss / total_weight));
	float inv_total = static_cast<float>(1.0 / total_weight);
	return Value::node(std::move(result), {logits},
		[logits, softmax_val, target_idx, class_weights, N, C, label_smoothing, inv_total](const Tensor& grad_out) {
			if (!logits.requires_grad()) return;
			Tensor local_grad = softmax_val;
			float g = grad_out.data()[0] * inv_total;
			for (size_t n = 0; n < N; n++) {
				float scale = g * class_weights.data()[target_idx[n]];
				float* row = local_grad.data().data() + n * C;
				for (size_t c = 0; c < C; c++) row[c] -= label_smoothing / static_cast<float>(C);
				row[target_idx[n]] -= 1.0f - label_smoothing;
				for (size_t c = 0; c < C; c++) row[c] *= scale;
			}
			logits.grad() += local_grad;
		});
}

Value sequence_cross_entropy(const Value& logits, const Tensor& targets, int ignore_index,
	float label_smoothing) {
	const Tensor& logits_data = logits.data();
	if (logits_data.ndim() != 3) throw TensorShapeError();
	size_t N = logits_data.shape()[0], T = logits_data.shape()[1], V = logits_data.shape()[2];
	if (targets.ndim() != 2 || targets.shape()[0] != N || targets.shape()[1] != T) throw TensorShapeError();
	if (!(label_smoothing >= 0.0f && label_smoothing < 1.0f)) {
		throw std::invalid_argument("sequence_cross_entropy: label_smoothing must be in [0, 1)");
	}
	const size_t P = N * T;
	// target_idx[p] == V marks an ignored position
	std::vector<size_t> target_idx(P);
	size_t count = 0;
	for (size_t p = 0; p < P; p++) {
		float t = std::round(targets.data()[p]);
		if (t == static_cast<float>(ignore_index)) {
			target_idx[p] = V;
			continue;
		}
		if (!(t >= 0.0f && t < static_cast<float>(V))) {
			throw std::invalid_argument("sequence_cross_entropy: target out of range");
		}
		target_idx[p] = static_cast<size_t>(t);
		count++;
	}
	if (count == 0) throw std::invalid_argument("sequence_cross_entropy: every position is ignored");

	Tensor softmax_val(logits_data.shape(), 0.0f);
	std::vector<float> pos_loss(P, 0.0f);
	const float inv_v = 1.0f / static_cast<float>(V);
	kernels::parallel_range(P, std::max<size_t>(1, kElementwiseChunk / std::max<size_t>(1, V)),
		[&](size_t lo, size_t hi) {
			for (size_t p = lo; p < hi; p++) {
				if (target_idx[p] == V) continue;
				const float* row = logits_data.data().data() + p * V;
				float* sm = softmax_val.data().data() + p * V;
				float max_val = row[0];
				for (size_t c = 1; c < V; c++) max_val = std::max(max_val, row[c]);
				float sum_exp = 0.0f;
				for (size_t c = 0; c < V; c++) sum_exp += std::exp(row[c] - max_val);
				float log_sum_exp = std::log(sum_exp);
				float mean_nll = 0.0f;
				for (size_t c = 0; c < V; c++) {
					float log_prob = row[c] - max_val - log_sum_exp;
					sm[c] = std::exp(log_prob);
					mean_nll -= log_prob * inv_v;
				}
				float nll = -(row[target_idx[p]] - max_val - log_sum_exp);
				pos_loss[p] = (1.0f - label_smoothing) * nll + label_smoothing * mean_nll;
			}
		});
	double total_loss = 0.0;  // serial, in position order: independent of thread count
	for (size_t p = 0; p < P; p++) total_loss += pos_loss[p];

	Tensor result({1}, static_cast<float>(total_loss / static_cast<double>(count)));
	float inv_count = 1.0f / static_cast<float>(count);
	return Value::node(std::move(result), {logits},
		[logits, softmax_val, target_idx, P, V, label_smoothing, inv_count](const Tensor& grad_out) {
			if (!logits.requires_grad()) return;
			Tensor local_grad(softmax_val.shape(), 0.0f);
			float g = grad_out.data()[0] * inv_count;
			for (size_t p = 0; p < P; p++) {
				if (target_idx[p] == V) continue;
				const float* sm = softmax_val.data().data() + p * V;
				float* row = local_grad.data().data() + p * V;
				for (size_t c = 0; c < V; c++) row[c] = sm[c] - label_smoothing / static_cast<float>(V);
				row[target_idx[p]] -= 1.0f - label_smoothing;
				for (size_t c = 0; c < V; c++) row[c] *= g;
			}
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
Value embedding(const Value& weight, const Tensor& ids) {
	const std::vector<size_t>& ws = weight.data().shape();
	if (ws.size() != 2) throw TensorShapeError();
	size_t V = ws[0], D = ws[1];
	const Storage& idv = ids.data();
	std::vector<size_t> rows(idv.size());
	for (size_t i = 0; i < idv.size(); i++) {
		float f = idv[i];
		if (!(f >= 0.0f) || f != std::floor(f) || f >= static_cast<float>(V)) throw TensorShapeError();
		rows[i] = static_cast<size_t>(f);
	}
	std::vector<size_t> out_shape = ids.shape();
	out_shape.push_back(D);
	Tensor result = Tensor::empty(out_shape);
	const float* w = weight.data().data().data();
	float* out = result.data().data();
	for (size_t i = 0; i < rows.size(); i++) std::copy(w + rows[i] * D, w + (rows[i] + 1) * D, out + i * D);
	return Value::node(std::move(result), {weight}, [weight, rows, D](const Tensor& grad_out) {
		if (!weight.requires_grad()) return;
		float* g = weight.grad().data().data();
		const float* go = grad_out.data().data();
		for (size_t i = 0; i < rows.size(); i++)
			for (size_t d = 0; d < D; d++) g[rows[i] * D + d] += go[i * D + d];
	});
}

Value operator*(const Value& a, float scalar) { return scale(a, scalar); }
Value operator*(float scalar, const Value& a) { return scale(a, scalar); }
Value operator-(const Value& a) { return scale(a, -1.0f); }

}  // namespace nn
