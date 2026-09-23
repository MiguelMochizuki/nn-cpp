/**
 * tensor.cpp
 * Author: Miguel Mochizuki Silva
 * Description: General N-dimensional float array with row-major contiguous storage
 */

#include "nn/tensor.hpp"

#include <numeric>
#include <functional>
#include <random>
#include <algorithm>
#include <limits>

namespace nn {

namespace {

std::vector<size_t> compute_strides(const std::vector<size_t>& shape) {
	std::vector<size_t> strides(shape.size());
	size_t acc = 1;
	for (size_t i = shape.size(); i-- > 0;) {
		strides[i] = acc;
		acc *= shape[i];
	}
	return strides;
}

size_t product(const std::vector<size_t>& shape) {
	return std::accumulate(shape.begin(), shape.end(), size_t{1}, std::multiplies<size_t>());
}

std::vector<size_t> drop_axis(const std::vector<size_t>& shape, size_t axis) {
	std::vector<size_t> out;
	out.reserve(shape.size() - 1);
	for (size_t i = 0; i < shape.size(); i++) {
		if (i != axis) out.push_back(shape[i]);
	}
	return out;
}

std::vector<size_t> unravel_index(size_t flat, const std::vector<size_t>& strides, const std::vector<size_t>& shape) {
	std::vector<size_t> idx(shape.size());
	for (size_t i = 0; i < shape.size(); i++) {
		idx[i] = (flat / strides[i]) % shape[i];
	}
	return idx;
}

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

template <typename Op>
Tensor elementwise(const Tensor& a, const Tensor& b, Op op) {
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
size_t Tensor::size() const { return data_.size(); }

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

Tensor Tensor::reshape(std::vector<size_t> new_shape) const {
	if (product(new_shape) != data_.size()) {
		throw TensorShapeError();
	}
	return Tensor(std::move(new_shape), data_);
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
	std::vector<size_t> out_shape = drop_axis(shape_, static_cast<size_t>(axis));
	Tensor out(out_shape);
	std::vector<size_t> out_strides = compute_strides(out_shape);

	for (size_t flat = 0; flat < data_.size(); flat++) {
		std::vector<size_t> idx = unravel_index(flat, strides_, shape_);
		idx.erase(idx.begin() + axis);
		size_t out_flat = 0;
		for (size_t i = 0; i < idx.size(); i++) out_flat += idx[i] * out_strides[i];
		out.data_[out_flat] += data_[flat];
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
	std::vector<size_t> out_shape = drop_axis(shape_, static_cast<size_t>(axis));
	Tensor out(out_shape, -std::numeric_limits<float>::infinity());
	std::vector<size_t> out_strides = compute_strides(out_shape);

	for (size_t flat = 0; flat < data_.size(); flat++) {
		std::vector<size_t> idx = unravel_index(flat, strides_, shape_);
		idx.erase(idx.begin() + axis);
		size_t out_flat = 0;
		for (size_t i = 0; i < idx.size(); i++) out_flat += idx[i] * out_strides[i];
		out.data_[out_flat] = std::max(out.data_[out_flat], data_[flat]);
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

Tensor matmul(const Tensor& a, const Tensor& b) {
	if (a.ndim() < 2 || b.ndim() < 2) {
		throw TensorShapeError();
	}
	size_t m = a.shape()[a.ndim() - 2];
	size_t k = a.shape()[a.ndim() - 1];
	size_t k2 = b.shape()[b.ndim() - 2];
	size_t n = b.shape()[b.ndim() - 1];
	if (k != k2) {
		throw TensorShapeError();
	}

	std::vector<size_t> a_batch(a.shape().begin(), a.shape().end() - 2);
	std::vector<size_t> b_batch(b.shape().begin(), b.shape().end() - 2);
	std::vector<size_t> a_batch_strides(a.strides().begin(), a.strides().end() - 2);
	std::vector<size_t> b_batch_strides(b.strides().begin(), b.strides().end() - 2);
	std::vector<size_t> batch_shape = broadcast_shape(a_batch, b_batch);

	std::vector<size_t> out_shape = batch_shape;
	out_shape.push_back(m);
	out_shape.push_back(n);
	Tensor out(out_shape);

	size_t batch_count = 1;
	for (size_t d : batch_shape) batch_count *= d;

	size_t a_row_stride = a.strides()[a.ndim() - 2];
	size_t a_col_stride = a.strides()[a.ndim() - 1];
	size_t b_row_stride = b.strides()[b.ndim() - 2];
	size_t b_col_stride = b.strides()[b.ndim() - 1];
	size_t out_row_stride = out.strides()[out.ndim() - 2];
	size_t out_col_stride = out.strides()[out.ndim() - 1];
	std::vector<size_t> batch_strides = compute_strides(batch_shape);

	for (size_t bi = 0; bi < batch_count; bi++) {
		std::vector<size_t> batch_idx = unravel_index(bi, batch_strides, batch_shape);
		size_t a_base = broadcast_offset(batch_idx, a_batch, a_batch_strides);
		size_t b_base = broadcast_offset(batch_idx, b_batch, b_batch_strides);
		size_t out_base = bi * m * n;

		for (size_t i = 0; i < m; i++) {
			for (size_t kk = 0; kk < k; kk++) {
				float a_val = a.data()[a_base + i * a_row_stride + kk * a_col_stride];
				for (size_t j = 0; j < n; j++) {
					out.data()[out_base + i * out_row_stride + j * out_col_stride] +=
						a_val * b.data()[b_base + kk * b_row_stride + j * b_col_stride];
				}
			}
		}
	}
	return out;
}

}  // namespace nn
