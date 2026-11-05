/**
 * tensor.cpp
 * Author: Miguel Mochizuki Silva
 * Description: General N-dimensional float array with row-major contiguous storage
 */

#include "nn/tensor.hpp"
#include "nn/kernels/gemm.hpp"

#include <numeric>
#include <functional>
#include <random>
#include <algorithm>
#include <limits>

namespace nn {

namespace {

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
		if (a_dim != b_dim && a_dim != 1 && b_dim != 1) {
			throw TensorShapeError();
		}
		out[i] = std::max(a_dim, b_dim);
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
 * Applies a binary operation elementwise, broadcasting when the shapes differ
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
	if (a.shape() == b.shape()) {
		Tensor out(a.shape());
		const float* a_data = a.data().data();
		const float* b_data = b.data().data();
		float* out_data = out.data().data();
		size_t n = a.size();
		for (size_t i = 0; i < n; i++) {
			out_data[i] = op(a_data[i], b_data[i]);
		}
		return out;
	}

	std::vector<size_t> out_shape = broadcast_shape(a.shape(), b.shape());
	Tensor out(out_shape);
	std::vector<size_t> out_strides = compute_strides(out_shape);

	for (size_t flat = 0; flat < out.size(); flat++) {
		std::vector<size_t> out_idx = unravel_index(flat, out_strides, out_shape);
		size_t a_off = broadcast_offset(out_idx, a.shape(), a.strides());
		size_t b_off = broadcast_offset(out_idx, b.shape(), b.strides());
		out.data()[flat] = op(a.data()[a_off], b.data()[b_off]);
	}
	return out;
}

}  // namespace

Tensor::Tensor(std::vector<size_t> shape)
	: shape_(std::move(shape)), strides_(compute_strides(shape_)), data_(product(shape_), 0.0f) {}

Tensor::Tensor(std::vector<size_t> shape, std::vector<float> data)
	: shape_(std::move(shape)), strides_(compute_strides(shape_)), data_(std::move(data)) {
	if (data_.size() != product(shape_)) {
		throw TensorShapeError();
	}
}

Tensor::Tensor(std::vector<size_t> shape, float fill)
	: shape_(std::move(shape)), strides_(compute_strides(shape_)), data_(product(shape_), fill) {}

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
const std::vector<float>& Tensor::data() const { return data_; }
std::vector<float>& Tensor::data() { return data_; }

Tensor& Tensor::operator+=(const Tensor& other) {
	if (shape_ != other.shape_) throw TensorShapeError();
	for (size_t i = 0; i < data_.size(); i++) data_[i] += other.data_[i];
	return *this;
}

Tensor& Tensor::operator-=(const Tensor& other) {
	if (shape_ != other.shape_) throw TensorShapeError();
	for (size_t i = 0; i < data_.size(); i++) data_[i] -= other.data_[i];
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

void Tensor::zero_() {
	std::fill(data_.begin(), data_.end(), 0.0f);
}

Tensor Tensor::reshape(std::vector<size_t> new_shape) const& {
	if (product(new_shape) != data_.size()) {
		throw TensorShapeError();
	}
	return Tensor(std::move(new_shape), data_);
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
	Tensor out(out_shape);
	std::vector<size_t> out_strides = compute_strides(out_shape);

	for (size_t flat = 0; flat < data_.size(); flat++) {
		std::vector<size_t> in_idx(shape_.size());
		size_t remaining = flat;
		for (size_t i = 0; i < shape_.size(); i++) {
			in_idx[i] = remaining / strides_[i];
			remaining %= strides_[i];
		}
		size_t out_flat = 0;
		for (size_t i = 0; i < perm.size(); i++) {
			out_flat += in_idx[perm[i]] * out_strides[i];
		}
		out.data_[out_flat] = data_[flat];
	}
	return out;
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

	for (size_t outer = 0; outer < outer_size; outer++) {
		for (size_t a = 0; a < axis_size; a++) {
			size_t src_base = outer * axis_size * inner_size + a * inner_size;
			size_t dst_base = outer * inner_size;
			for (size_t inner = 0; inner < inner_size; inner++) {
				dst[dst_base + inner] += src[src_base + inner];
			}
		}
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

	for (size_t outer = 0; outer < outer_size; outer++) {
		for (size_t a = 0; a < axis_size; a++) {
			size_t src_base = outer * axis_size * inner_size + a * inner_size;
			size_t dst_base = outer * inner_size;
			for (size_t inner = 0; inner < inner_size; inner++) {
				dst[dst_base + inner] = std::max(dst[dst_base + inner], src[src_base + inner]);
			}
		}
	}
	return out;
}

Tensor Tensor::zeros(std::vector<size_t> shape) { return Tensor(std::move(shape)); }
Tensor Tensor::ones(std::vector<size_t> shape) { return Tensor(std::move(shape), 1.0f); }
Tensor Tensor::full(std::vector<size_t> shape, float v) { return Tensor(std::move(shape), v); }

Tensor Tensor::random(std::vector<size_t> shape, float lo, float hi) {
	static std::mt19937 gen(std::random_device{}());
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
	Tensor out = a;
	for (float& v : out.data()) v *= scalar;
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

	std::vector<size_t> a_batch(a.shape().begin(), a.shape().end() - 2);
	std::vector<size_t> b_batch(b.shape().begin(), b.shape().end() - 2);
	std::vector<size_t> batch_shape = broadcast_shape(a_batch, b_batch);

	std::vector<size_t> out_shape = batch_shape;
	out_shape.push_back(m);
	out_shape.push_back(n);
	Tensor out(out_shape);

	size_t batch_count = 1;
	for (size_t d : batch_shape) batch_count *= d;

	// Tensors are dense row-major, so a matrix's leading dimension is its stored column count
	std::vector<size_t> a_batch_strides(a.strides().begin(), a.strides().end() - 2);
	std::vector<size_t> b_batch_strides(b.strides().begin(), b.strides().end() - 2);
	std::vector<size_t> batch_strides = compute_strides(batch_shape);

	for (size_t bi = 0; bi < batch_count; bi++) {
		size_t a_off = 0, b_off = 0;
		if (batch_count > 1) {
			std::vector<size_t> batch_idx = unravel_index(bi, batch_strides, batch_shape);
			a_off = broadcast_offset(batch_idx, a_batch, a_batch_strides);
			b_off = broadcast_offset(batch_idx, b_batch, b_batch_strides);
		}
		kernels::sgemm(trans_a, trans_b, m, n, k, a.data().data() + a_off, a_cols, b.data().data() + b_off, b_cols,
			out.data().data() + bi * m * n, n, false);
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
