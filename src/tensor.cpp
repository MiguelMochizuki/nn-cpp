/**
 * tensor.cpp
 * Author: Miguel Mochizuki Silva
 * Description: General N-dimensional float array with row-major contiguous storage
 */

#include "nn/tensor.hpp"
#include "nn/kernels/gemm.hpp"
#include "nn/kernels/parallel.hpp"
#include "nn/random.hpp"

#include <numeric>
#include <functional>
#include <random>
#include <algorithm>
#include <limits>

#ifdef __GLIBC__
#include <malloc.h>
#endif

namespace nn {

namespace {

#ifdef __GLIBC__
// A training step allocates and frees many multi-megabyte tensors. By default glibc serves
// blocks above ~128 KB with mmap and returns them with munmap, so every step page-faults and
// zeroes fresh memory in the kernel, serially: about 40% of a cnn_fer step. Keeping freed
// blocks in the heap makes later allocations reuse warm pages. The mmap threshold cannot be set
// above 32 MiB, and a 64 x 32 x 64 x 64 activation is exactly that, so mmap is switched off.
// Memory stays reserved at the high-water mark of the process.
// ponytail: process-wide glibc setting; a pooling allocator for Tensor storage would scope it to the library
[[maybe_unused]] const bool kMallocTuned = [] {
	mallopt(M_MMAP_THRESHOLD, 32 << 20);
	mallopt(M_MMAP_MAX, 0);
	mallopt(M_TRIM_THRESHOLD, 1 << 30);
	mallopt(M_TOP_PAD, 64 << 20);
	return true;
}();
#endif

// Elements per thread below which an elementwise loop stays serial
constexpr size_t kElementwiseChunk = size_t(1) << 13;

/**
 * Computes row-major strides for a shape
 *
 * Parameters:
 * const std::vector<size_t>& shape: Dimensions
 *
 * Returns std::vector<size_t>: Elements to skip per step along each axis
 */
std::vector<size_t> compute_strides(const std::vector<size_t>& shape) {
	std::vector<size_t> strides(shape.size());
	size_t acc = 1;
	for (size_t i = shape.size(); i-- > 0;) {
		strides[i] = acc;
		acc *= shape[i];
	}
	return strides;
}

/**
 * Multiplies all dimensions of a shape
 *
 * Parameters:
 * const std::vector<size_t>& shape: Dimensions
 *
 * Returns size_t: Element count (1 for an empty shape)
 */
size_t product(const std::vector<size_t>& shape) {
	return std::accumulate(shape.begin(), shape.end(), size_t{1}, std::multiplies<size_t>());
}

/**
 * Removes one axis from a shape
 *
 * Parameters:
 * const std::vector<size_t>& shape: Dimensions
 * size_t axis: Axis to remove
 *
 * Returns std::vector<size_t>: Shape with one fewer dimension
 */
std::vector<size_t> drop_axis(const std::vector<size_t>& shape, size_t axis) {
	std::vector<size_t> out;
	out.reserve(shape.size() - 1);
	for (size_t i = 0; i < shape.size(); i++) {
		if (i != axis) out.push_back(shape[i]);
	}
	return out;
}

/**
 * Converts a flat element offset into a multi-dimensional index
 *
 * Parameters:
 * size_t flat: Offset into the data array
 * const std::vector<size_t>& strides: Row-major strides of the tensor
 * const std::vector<size_t>& shape: Dimensions of the tensor
 *
 * Returns std::vector<size_t>: Index along each axis
 */
std::vector<size_t> unravel_index(size_t flat, const std::vector<size_t>& strides, const std::vector<size_t>& shape) {
	std::vector<size_t> idx(shape.size());
	for (size_t i = 0; i < shape.size(); i++) {
		idx[i] = (flat / strides[i]) % shape[i];
	}
	return idx;
}

/**
 * Computes the numpy-style broadcast shape of two shapes
 *
 * Parameters:
 * const std::vector<size_t>& a: First shape
 * const std::vector<size_t>& b: Second shape
 *
 * Returns std::vector<size_t>: Broadcast shape; throws TensorShapeError if the shapes are incompatible
 */
std::vector<size_t> broadcast_shape(const std::vector<size_t>& a, const std::vector<size_t>& b) {
	size_t rank = std::max(a.size(), b.size());
	std::vector<size_t> out(rank);
	for (size_t i = 0; i < rank; i++) {
		size_t a_dim = (i < rank - a.size()) ? 1 : a[i - (rank - a.size())];
		size_t b_dim = (i < rank - b.size()) ? 1 : b[i - (rank - b.size())];
		if (a_dim == b_dim || b_dim == 1) {
			out[i] = a_dim;
		} else if (a_dim == 1) {
			out[i] = b_dim;
		} else {
			throw TensorShapeError();
		}
	}
	return out;
}

size_t broadcast_offset(const std::vector<size_t>& out_idx, const std::vector<size_t>& src_shape,
                         const std::vector<size_t>& src_strides) {
	size_t rank_diff = out_idx.size() - src_shape.size();
	size_t offset = 0;
	for (size_t d = 0; d < src_shape.size(); d++) {
		if (src_shape[d] != 1) {
			offset += out_idx[d + rank_diff] * src_strides[d];
		}
	}
	return offset;
}

/**
 * Applies a binary operation elementwise, broadcasting when the shapes differ.
 *
 * The result shape and a stride per operand (0 on broadcast or missing axes) are computed once,
 * adjacent axes that both operands walk contiguously are merged, and the innermost merged axis
 * runs as a tight loop. The output is split into disjoint flat ranges over the thread pool, so
 * the result does not depend on the thread count; each element is computed as op(a, b) exactly
 * as in a per-element index loop.
 *
 * Parameters:
 * const Tensor& a: Left operand
 * const Tensor& b: Right operand
 * Op op: Callable taking two floats and returning a float
 *
 * Returns Tensor: Result with the broadcast shape
 */
template <typename Op>
Tensor elementwise(const Tensor& a, const Tensor& b, Op op) {
	std::vector<size_t> out_shape = a.shape() == b.shape() ? a.shape() : broadcast_shape(a.shape(), b.shape());
	Tensor out = Tensor::empty(out_shape);  // every element is computed below
	const size_t n = out.size();
	if (n == 0) return out;

	struct Axis {
		size_t dim, sa, sb;
	};
	std::vector<Axis> axes;
	const size_t rank = out_shape.size();
	for (size_t i = 0; i < rank; i++) {
		if (out_shape[i] == 1) continue;
		size_t sa = 0, sb = 0;
		if (i + a.ndim() >= rank && a.shape()[i + a.ndim() - rank] != 1) sa = a.strides()[i + a.ndim() - rank];
		if (i + b.ndim() >= rank && b.shape()[i + b.ndim() - rank] != 1) sb = b.strides()[i + b.ndim() - rank];
		if (!axes.empty() && axes.back().sa == sa * out_shape[i] && axes.back().sb == sb * out_shape[i]) {
			axes.back().dim *= out_shape[i];
			axes.back().sa = sa;
			axes.back().sb = sb;
		} else {
			axes.push_back({out_shape[i], sa, sb});
		}
	}
	if (axes.empty()) axes.push_back({1, 0, 0});

	const size_t inner = axes.back().dim, isa = axes.back().sa, isb = axes.back().sb;
	const size_t outer = axes.size() - 1;
	const float* ad = a.data().data();
	const float* bd = b.data().data();
	float* od = out.data().data();

	kernels::parallel_range(n, kElementwiseChunk, [&](size_t lo, size_t hi) {
		std::vector<size_t> idx(outer);
		size_t a_off = 0, b_off = 0;
		size_t rows = lo / inner, j = lo % inner;
		for (size_t d = outer; d-- > 0;) {
			idx[d] = rows % axes[d].dim;
			rows /= axes[d].dim;
			a_off += idx[d] * axes[d].sa;
			b_off += idx[d] * axes[d].sb;
		}
		for (size_t pos = lo; pos < hi;) {
			size_t len = std::min(inner - j, hi - pos);
			const float* pa = ad + a_off + j * isa;
			const float* pb = bd + b_off + j * isb;
			float* po = od + pos;
			if (isa == 1 && isb == 1) {
				for (size_t k = 0; k < len; k++) po[k] = op(pa[k], pb[k]);
			} else if (isa == 1 && isb == 0) {
				const float v = *pb;
				for (size_t k = 0; k < len; k++) po[k] = op(pa[k], v);
			} else if (isa == 0 && isb == 1) {
				const float v = *pa;
				for (size_t k = 0; k < len; k++) po[k] = op(v, pb[k]);
			} else {
				for (size_t k = 0; k < len; k++) po[k] = op(pa[k * isa], pb[k * isb]);
			}
			pos += len;
			j = 0;
			// Advance the odometer over the outer axes
			for (size_t d = outer; d-- > 0;) {
				a_off += axes[d].sa;
				b_off += axes[d].sb;
				if (++idx[d] < axes[d].dim) break;
				a_off -= axes[d].sa * axes[d].dim;
				b_off -= axes[d].sb * axes[d].dim;
				idx[d] = 0;
			}
		}
	});
	return out;
}

}  // namespace

namespace {
/**
 * Writes value to every element of storage, in parallel for large buffers, so that the first touch
 * of fresh memory is spread over the threads instead of one core zeroing it alone
 *
 * Parameters:
 * Storage& data: Buffer to fill
 * float value: Value written to each element
 *
 * Returns void
 */
void fill_storage(Storage& data, float value) {
	float* p = data.data();
	kernels::parallel_range(data.size(), size_t(1) << 16, [=](size_t lo, size_t hi) { std::fill(p + lo, p + hi, value); });
}
}  // namespace

Tensor::Tensor(std::vector<size_t> shape, Uninit)
	: shape_(std::move(shape)), strides_(compute_strides(shape_)), data_(product(shape_)) {}

Tensor Tensor::empty(std::vector<size_t> shape) { return Tensor(std::move(shape), Uninit{}); }

Tensor::Tensor(std::vector<size_t> shape) : Tensor(std::move(shape), Uninit{}) { fill_storage(data_, 0.0f); }

Tensor::Tensor(std::vector<size_t> shape, std::vector<float> data)
	: shape_(std::move(shape)), strides_(compute_strides(shape_)), data_(data.begin(), data.end()) {
	if (data_.size() != product(shape_)) {
		throw TensorShapeError();
	}
}

Tensor::Tensor(std::vector<size_t> shape, Storage data)
	: shape_(std::move(shape)), strides_(compute_strides(shape_)), data_(std::move(data)) {
	if (data_.size() != product(shape_)) {
		throw TensorShapeError();
	}
}

Tensor::Tensor(std::vector<size_t> shape, float fill) : Tensor(std::move(shape), Uninit{}) { fill_storage(data_, fill); }

const std::vector<size_t>& Tensor::shape() const { return shape_; }
const std::vector<size_t>& Tensor::strides() const { return strides_; }
size_t Tensor::ndim() const { return shape_.size(); }

size_t Tensor::flat_index(const std::vector<size_t>& idx) const {
	if (idx.size() != shape_.size()) {
		throw TensorIndexError();
	}
	size_t flat = 0;
	for (size_t i = 0; i < idx.size(); i++) {
		if (idx[i] >= shape_[i]) {
			throw TensorIndexError();
		}
		flat += idx[i] * strides_[i];
	}
	return flat;
}

float Tensor::get(const std::vector<size_t>& idx) const { return data_[flat_index(idx)]; }
float& Tensor::at(const std::vector<size_t>& idx) { return data_[flat_index(idx)]; }
const Storage& Tensor::data() const { return data_; }
Storage& Tensor::data() { return data_; }

Tensor& Tensor::operator+=(const Tensor& other) {
	if (shape_ != other.shape_) throw TensorShapeError();
	float* d = data_.data();
	const float* o = other.data_.data();
	kernels::parallel_range(data_.size(), kElementwiseChunk, [=](size_t lo, size_t hi) {
		for (size_t i = lo; i < hi; i++) d[i] += o[i];
	});
	return *this;
}

Tensor& Tensor::operator-=(const Tensor& other) {
	if (shape_ != other.shape_) throw TensorShapeError();
	float* d = data_.data();
	const float* o = other.data_.data();
	kernels::parallel_range(data_.size(), kElementwiseChunk, [=](size_t lo, size_t hi) {
		for (size_t i = lo; i < hi; i++) d[i] -= o[i];
	});
	return *this;
}

Tensor& Tensor::operator*=(const Tensor& other) {
	if (shape_ != other.shape_) throw TensorShapeError();
	for (size_t i = 0; i < data_.size(); i++) data_[i] *= other.data_[i];
	return *this;
}

Tensor& Tensor::operator/=(const Tensor& other) {
	if (shape_ != other.shape_) throw TensorShapeError();
	for (size_t i = 0; i < data_.size(); i++) data_[i] /= other.data_[i];
	return *this;
}

Tensor& Tensor::operator*=(float scalar) {
	for (float& v : data_) v *= scalar;
	return *this;
}

void Tensor::zero_() { fill_storage(data_, 0.0f); }

Tensor Tensor::reshape(std::vector<size_t> new_shape) const& {
	if (product(new_shape) != data_.size()) {
		throw TensorShapeError();
	}
	// A parallel copy into uninitialised storage; the vector copy constructor would do it on one thread
	Tensor out(std::move(new_shape), Uninit{});
	const float* src = data_.data();
	float* dst = out.data_.data();
	kernels::parallel_range(data_.size(), size_t(1) << 15, [=](size_t lo, size_t hi) { std::copy(src + lo, src + hi, dst + lo); });
	return out;
}

Tensor Tensor::reshape(std::vector<size_t> new_shape) && {
	if (product(new_shape) != data_.size()) {
		throw TensorShapeError();
	}
	return Tensor(std::move(new_shape), std::move(data_));
}

Tensor Tensor::transpose(std::vector<size_t> perm) const {
	if (perm.size() != shape_.size()) {
		throw TensorShapeError();
	}
	std::vector<size_t> out_shape(shape_.size());
	for (size_t i = 0; i < perm.size(); i++) {
		out_shape[i] = shape_[perm[i]];
	}
	Tensor out = Tensor::empty(out_shape);  // transpose_into writes every element
	transpose_into(perm, out.data_.data(), Combine::kCopy);
	return out;
}

void Tensor::transpose_into(const std::vector<size_t>& perm, float* dst, Combine mode) const {
	if (perm.size() != shape_.size()) {
		throw TensorShapeError();
	}
	const size_t rank = perm.size();
	if (rank == 0 || data_.empty()) return;
	// Input stride of each output axis; walking the output in order, the input offset follows an
	// odometer over the outer axes while the last axis runs as a tight strided loop
	std::vector<size_t> out_shape(rank), in_stride(rank);
	for (size_t i = 0; i < rank; i++) {
		out_shape[i] = shape_[perm[i]];
		in_stride[i] = strides_[perm[i]];
	}
	const size_t inner = out_shape[rank - 1], istep = in_stride[rank - 1];
	const float* src = data_.data();
	// Whole rows of the last axis per chunk: every output element is still written by one thread
	kernels::parallel_range(data_.size() / inner, std::max<size_t>(1, kElementwiseChunk / inner), [&](size_t lo, size_t hi) {
		std::vector<size_t> idx(rank - 1);
		size_t rem = lo, in = 0;
		for (size_t d = rank - 1; d-- > 0;) {
			idx[d] = rem % out_shape[d];
			rem /= out_shape[d];
			in += idx[d] * in_stride[d];
		}
		for (size_t row = lo; row < hi; row++) {
			float* o = dst + row * inner;
			const float* s = src + in;
			if (mode == Combine::kCopy) {
				for (size_t j = 0; j < inner; j++) o[j] = s[j * istep];
			} else if (mode == Combine::kAdd) {
				for (size_t j = 0; j < inner; j++) o[j] += s[j * istep];
			} else {
				for (size_t j = 0; j < inner; j++) o[j] = s[j * istep] + 0.0f;
			}
			for (size_t d = rank - 1; d-- > 0;) {
				idx[d]++;
				in += in_stride[d];
				if (idx[d] < out_shape[d]) break;
				in -= idx[d] * in_stride[d];
				idx[d] = 0;
			}
		}
	});
}

Tensor Tensor::sum(int axis) const {
	if (axis < 0 || static_cast<size_t>(axis) >= shape_.size()) {
		throw TensorIndexError();
	}
	std::vector<size_t> out_shape = drop_axis(shape_, static_cast<size_t>(axis));
	Tensor out(out_shape);
	if (data_.empty()) {
		return out;
	}

	size_t inner_size = strides_[axis];
	size_t axis_size = shape_[axis];
	size_t outer_size = data_.size() / (axis_size * inner_size);
	const float* src = data_.data();
	float* dst = out.data_.data();

	// Every output element is written by one thread and summed in the same order as a serial loop
	if (outer_size > 1) {
		kernels::parallel_range(outer_size, std::max<size_t>(1, kElementwiseChunk / (axis_size * inner_size)),
			[=](size_t lo, size_t hi) {
			for (size_t outer = lo; outer < hi; outer++)
				for (size_t a = 0; a < axis_size; a++) {
					const float* row = src + outer * axis_size * inner_size + a * inner_size;
					float* acc = dst + outer * inner_size;
					for (size_t inner = 0; inner < inner_size; inner++) acc[inner] += row[inner];
				}
		});
	} else {
		kernels::parallel_range(inner_size, std::max<size_t>(1, kElementwiseChunk / axis_size), [=](size_t lo, size_t hi) {
			for (size_t a = 0; a < axis_size; a++)
				for (size_t inner = lo; inner < hi; inner++) dst[inner] += src[a * inner_size + inner];
		});
	}
	return out;
}

Tensor Tensor::mean(int axis) const {
	Tensor s = sum(axis);
	float n = static_cast<float>(shape_[axis]);
	for (float& v : s.data_) v /= n;
	return s;
}

Tensor Tensor::max(int axis) const {
	if (axis < 0 || static_cast<size_t>(axis) >= shape_.size()) {
		throw TensorIndexError();
	}
	std::vector<size_t> out_shape = drop_axis(shape_, static_cast<size_t>(axis));
	Tensor out(out_shape, -std::numeric_limits<float>::infinity());
	if (data_.empty()) {
		return out;
	}

	size_t inner_size = strides_[axis];
	size_t axis_size = shape_[axis];
	size_t outer_size = data_.size() / (axis_size * inner_size);
	const float* src = data_.data();
	float* dst = out.data_.data();

	if (outer_size > 1) {
		kernels::parallel_range(outer_size, std::max<size_t>(1, kElementwiseChunk / (axis_size * inner_size)),
			[=](size_t lo, size_t hi) {
			for (size_t outer = lo; outer < hi; outer++)
				for (size_t a = 0; a < axis_size; a++) {
					const float* row = src + outer * axis_size * inner_size + a * inner_size;
					float* acc = dst + outer * inner_size;
					for (size_t inner = 0; inner < inner_size; inner++) acc[inner] = std::max(acc[inner], row[inner]);
				}
		});
	} else {
		kernels::parallel_range(inner_size, std::max<size_t>(1, kElementwiseChunk / axis_size), [=](size_t lo, size_t hi) {
			for (size_t a = 0; a < axis_size; a++)
				for (size_t inner = lo; inner < hi; inner++) dst[inner] = std::max(dst[inner], src[a * inner_size + inner]);
		});
	}
	return out;
}

Tensor Tensor::zeros(std::vector<size_t> shape) { return Tensor(std::move(shape)); }
Tensor Tensor::ones(std::vector<size_t> shape) { return Tensor(std::move(shape), 1.0f); }
Tensor Tensor::full(std::vector<size_t> shape, float v) { return Tensor(std::move(shape), v); }

Tensor Tensor::random(std::vector<size_t> shape, float lo, float hi) {
	// One stream per seed epoch, so repeated calls continue it and manual_seed restarts it
	// Not synchronised for concurrent use (same as v1.0.0)
	static std::mt19937 gen;
	static std::uint64_t epoch = 0;
	static bool init = false;
	if (!init || epoch != seed_epoch()) {
		epoch = seed_epoch();
		gen = derive_generator("tensor.random");
		init = true;
	}
	std::uniform_real_distribution<float> dist(lo, hi);
	Tensor out(std::move(shape));
	for (float& v : out.data_) {
		v = dist(gen);
	}
	return out;
}

Tensor Tensor::stack(const std::vector<Tensor>& items) {
	if (items.empty()) {
		throw TensorShapeError();
	}
	const std::vector<size_t>& item_shape = items[0].shape_;
	std::vector<size_t> out_shape;
	out_shape.push_back(items.size());
	out_shape.insert(out_shape.end(), item_shape.begin(), item_shape.end());

	std::vector<float> out_data;
	out_data.reserve(items.size() * items[0].size());
	for (const Tensor& t : items) {
		if (t.shape_ != item_shape) {
			throw TensorShapeError();
		}
		out_data.insert(out_data.end(), t.data_.begin(), t.data_.end());
	}
	return Tensor(std::move(out_shape), std::move(out_data));
}

Tensor add(const Tensor& a, const Tensor& b) {
	return elementwise(a, b, [](float x, float y) { return x + y; });
}

Tensor sub(const Tensor& a, const Tensor& b) {
	return elementwise(a, b, [](float x, float y) { return x - y; });
}

Tensor mul(const Tensor& a, const Tensor& b) {
	return elementwise(a, b, [](float x, float y) { return x * y; });
}

Tensor div(const Tensor& a, const Tensor& b) {
	return elementwise(a, b, [](float x, float y) { return x / y; });
}

Tensor scale(const Tensor& a, float scalar) {
	Tensor out = Tensor::empty(a.shape());
	const float* src = a.data().data();
	float* dst = out.data().data();
	kernels::parallel_range(a.size(), kElementwiseChunk, [=](size_t lo, size_t hi) {
		for (size_t i = lo; i < hi; i++) dst[i] = src[i] * scalar;
	});
	return out;
}

Tensor matmul(const Tensor& a, const Tensor& b) { return matmul(a, b, false, false); }

Tensor matmul(const Tensor& a, const Tensor& b, bool trans_a, bool trans_b) {
	if (a.ndim() < 2 || b.ndim() < 2) {
		throw TensorShapeError();
	}
	size_t a_rows = a.shape()[a.ndim() - 2];
	size_t a_cols = a.shape()[a.ndim() - 1];
	size_t b_rows = b.shape()[b.ndim() - 2];
	size_t b_cols = b.shape()[b.ndim() - 1];
	size_t m = trans_a ? a_cols : a_rows;
	size_t k = trans_a ? a_rows : a_cols;
	size_t k2 = trans_b ? b_cols : b_rows;
	size_t n = trans_b ? b_rows : b_cols;
	if (k != k2) {
		throw TensorShapeError();
	}

	// A batched left operand times a plain matrix: a is contiguous, so fold its batch axes into M rows
	// and run one GEMM instead of one per sample
	if (b.ndim() == 2 && !trans_a && a.ndim() > 2) {
		std::vector<size_t> folded_shape(a.shape().begin(), a.shape().end() - 1);
		folded_shape.push_back(n);
		Tensor folded = Tensor::empty(folded_shape);  // sgemm without accumulate overwrites all of it
		kernels::sgemm(false, trans_b, a.size() / a_cols, n, k, a.data().data(), a_cols, b.data().data(), b_cols,
			folded.data().data(), n, false);
		return folded;
	}

	std::vector<size_t> a_batch(a.shape().begin(), a.shape().end() - 2);
	std::vector<size_t> b_batch(b.shape().begin(), b.shape().end() - 2);
	std::vector<size_t> batch_shape = broadcast_shape(a_batch, b_batch);

	std::vector<size_t> out_shape = batch_shape;
	out_shape.push_back(m);
	out_shape.push_back(n);
	Tensor out = Tensor::empty(out_shape);  // each batch's sgemm overwrites its whole block

	size_t batch_count = 1;
	for (size_t d : batch_shape) batch_count *= d;

	// Tensors are dense row-major, so a matrix's leading dimension is its stored column count
	std::vector<size_t> a_batch_strides(a.strides().begin(), a.strides().end() - 2);
	std::vector<size_t> b_batch_strides(b.strides().begin(), b.strides().end() - 2);
	std::vector<size_t> batch_strides = compute_strides(batch_shape);

	auto one_matrix = [&](size_t bi) {
		size_t a_off = 0, b_off = 0;
		if (batch_count > 1) {
			std::vector<size_t> batch_idx = unravel_index(bi, batch_strides, batch_shape);
			a_off = broadcast_offset(batch_idx, a_batch, a_batch_strides);
			b_off = broadcast_offset(batch_idx, b_batch, b_batch_strides);
		}
		kernels::sgemm(trans_a, trans_b, m, n, k, a.data().data() + a_off, a_cols, b.data().data() + b_off, b_cols,
			out.data().data() + bi * m * n, n, false);
	};
	// sgemm only goes parallel above this many multiply-adds (see gemm.cpp). Below it each matrix is
	// serial, so many small matrices (attention heads) are spread over threads instead. A batch with
	// at least one matrix per thread is spread the same way whatever the matrix size: one fork/join
	// for the whole batch instead of two per matrix, and a wide flat matrix (a convolution's
	// Cout x K x hw) splits into too few chunks to use every thread. Every output matrix is still
	// computed by one thread with the same tile order, so results do not change.
	constexpr size_t kSmallGemm = size_t(1) << 20;
	if (batch_count > 1 && (m * n * k < kSmallGemm || batch_count >= kernels::num_threads())) {
		kernels::parallel_range(batch_count, std::max<size_t>(1, (size_t(1) << 16) / std::max<size_t>(1, m * n * k)),
			[&](size_t lo, size_t hi) {
			for (size_t bi = lo; bi < hi; bi++) one_matrix(bi);
		});
	} else {
		for (size_t bi = 0; bi < batch_count; bi++) one_matrix(bi);
	}
	return out;
}

Tensor operator+(const Tensor& a, const Tensor& b) { return add(a, b); }
Tensor operator-(const Tensor& a, const Tensor& b) { return sub(a, b); }
Tensor operator*(const Tensor& a, const Tensor& b) { return mul(a, b); }
Tensor operator/(const Tensor& a, const Tensor& b) { return div(a, b); }
Tensor operator*(const Tensor& a, float scalar) { return scale(a, scalar); }
Tensor operator*(float scalar, const Tensor& a) { return scale(a, scalar); }
Tensor operator-(const Tensor& a) { return scale(a, -1.0f); }

}  // namespace nn
