/**
 * tensor.hpp
 * Author: Miguel Mochizuki Silva
 * Description: General N-dimensional float array with row-major contiguous storage
 */

#pragma once

#include <vector>
#include <exception>

namespace nn {

/*
 * Thrown when tensor shapes are incompatible for an operation: mismatched
 * elementwise shapes, non-broadcastable dimensions, or non-conformable matmul.
 *
 * Attributes: none
 */
class TensorShapeError : public std::exception {
public:
	const char* what() const noexcept override {
		return "Tensor shapes are incompatible for this operation.";
	}
};

/*
 * Thrown when an index is out of bounds for a tensor's shape, or has the
 * wrong number of dimensions.
 *
 * Attributes: none
 */
class TensorIndexError : public std::exception {
public:
	const char* what() const noexcept override {
		return "Tensor index out of bounds.";
	}
};

/*
 * General N-dimensional float array, row-major contiguous storage.
 *
 * Attributes:
 * vector<size_t> shape_: size of each dimension
 * vector<size_t> strides_: row-major stride of each dimension, computed from shape_
 * vector<float> data_: flattened element storage, size equal to the product of shape_
 */
class Tensor {
public:
	/* Public, construct a zero-initialized tensor
	 *
	 * Parameters:
	 * vector<size_t> shape: size of each dimension
	 *
	 * Returns Tensor: none (constructor)
	 */
	explicit Tensor(std::vector<size_t> shape);

	/* Public, construct a tensor from explicit values
	 *
	 * Parameters:
	 * vector<size_t> shape: size of each dimension
	 * vector<float> data: values in row-major order, size must equal the product of shape
	 *
	 * Returns Tensor: none (constructor); throws TensorShapeError if data.size() does not match shape
	 */
	Tensor(std::vector<size_t> shape, std::vector<float> data);

	/* Public, construct a tensor filled with one value
	 *
	 * Parameters:
	 * vector<size_t> shape: size of each dimension
	 * float fill: value assigned to every element
	 *
	 * Returns Tensor: none (constructor)
	 */
	Tensor(std::vector<size_t> shape, float fill);

	/* Public, dimension sizes
	 *
	 * Parameters: none
	 *
	 * Returns const vector<size_t>&: size of each dimension
	 */
	const std::vector<size_t>& shape() const;

	/* Public, row-major strides
	 *
	 * Parameters: none
	 *
	 * Returns const vector<size_t>&: stride of each dimension
	 */
	const std::vector<size_t>& strides() const;

	/* Public, number of dimensions
	 *
	 * Parameters: none
	 *
	 * Returns size_t: rank of the tensor
	 */
	size_t ndim() const;

	/* Public, total element count
	 *
	 * Parameters: none
	 *
	 * Returns size_t: product of all dimension sizes
	 */
	size_t size() const;

	/* Public, read one element
	 *
	 * Parameters:
	 * vector<size_t> idx: one index per dimension
	 *
	 * Returns float: element value at idx; throws TensorIndexError if idx is malformed or out of range
	 */
	float get(const std::vector<size_t>& idx) const;

	/* Public, read-write access to one element
	 *
	 * Parameters:
	 * vector<size_t> idx: one index per dimension
	 *
	 * Returns float&: reference to the element at idx; throws TensorIndexError if idx is malformed or out of range
	 */
	float& at(const std::vector<size_t>& idx);

	/* Public, read-only access to the underlying flat storage
	 *
	 * Parameters: none
	 *
	 * Returns const vector<float>&: flattened row-major element storage
	 */
	const std::vector<float>& data() const;

	/* Public, read-write access to the underlying flat storage
	 *
	 * Parameters: none
	 *
	 * Returns vector<float>&: flattened row-major element storage
	 */
	std::vector<float>& data();

	/* Public, in-place elementwise addition, same shape only
	 *
	 * Parameters:
	 * Tensor other: operand; must have the exact same shape as this tensor
	 *
	 * Returns Tensor&: *this, each element incremented by the matching element of other;
	 * throws TensorShapeError if shapes differ
	 */
	Tensor& operator+=(const Tensor& other);

	/* Public, in-place elementwise subtraction, same shape only
	 *
	 * Parameters:
	 * Tensor other: operand; must have the exact same shape as this tensor
	 *
	 * Returns Tensor&: *this, each element decremented by the matching element of other;
	 * throws TensorShapeError if shapes differ
	 */
	Tensor& operator-=(const Tensor& other);

	/* Public, in-place elementwise multiplication, same shape only
	 *
	 * Parameters:
	 * Tensor other: operand; must have the exact same shape as this tensor
	 *
	 * Returns Tensor&: *this, each element multiplied by the matching element of other;
	 * throws TensorShapeError if shapes differ
	 */
	Tensor& operator*=(const Tensor& other);

	/* Public, in-place elementwise division, same shape only
	 *
	 * Parameters:
	 * Tensor other: operand; must have the exact same shape as this tensor
	 *
	 * Returns Tensor&: *this, each element divided by the matching element of other;
	 * throws TensorShapeError if shapes differ
	 */
	Tensor& operator/=(const Tensor& other);

	/* Public, in-place scalar multiplication
	 *
	 * Parameters:
	 * float scalar: multiplier
	 *
	 * Returns Tensor&: *this, every element multiplied by scalar
	 */
	Tensor& operator*=(float scalar);

	/* Public, set every element to zero in place
	 *
	 * Parameters: none
	 *
	 * Returns void: none
	 */
	void zero_();

	/* Public, reshape to new dimensions without changing row-major element order
	 *
	 * Parameters:
	 * vector<size_t> new_shape: new dimension sizes; product must equal size()
	 *
	 * Returns Tensor: new tensor with new_shape; throws TensorShapeError if sizes don't match
	 */
	Tensor reshape(std::vector<size_t> new_shape) const;

	/* Public, permute dimensions
	 *
	 * Parameters:
	 * vector<size_t> perm: perm[i] is the source dimension that becomes output dimension i;
	 * must be a permutation of [0, ndim())
	 *
	 * Returns Tensor: new tensor with permuted shape and correspondingly permuted data
	 */
	Tensor transpose(std::vector<size_t> perm) const;

	/* Public, sum along one axis
	 *
	 * Parameters:
	 * int axis: dimension to reduce, 0-based
	 *
	 * Returns Tensor: new tensor with that axis removed, each remaining element the sum
	 * over the reduced axis
	 */
	Tensor sum(int axis) const;

	/* Public, mean along one axis
	 *
	 * Parameters:
	 * int axis: dimension to reduce, 0-based
	 *
	 * Returns Tensor: new tensor with that axis removed, each remaining element the mean
	 * over the reduced axis
	 */
	Tensor mean(int axis) const;

	/* Public, maximum along one axis
	 *
	 * Parameters:
	 * int axis: dimension to reduce, 0-based
	 *
	 * Returns Tensor: new tensor with that axis removed, each remaining element the max
	 * over the reduced axis
	 */
	Tensor max(int axis) const;

	/* Public, zero-filled tensor factory
	 *
	 * Parameters:
	 * vector<size_t> shape: size of each dimension
	 *
	 * Returns Tensor: new zero-filled tensor
	 */
	static Tensor zeros(std::vector<size_t> shape);

	/* Public, one-filled tensor factory
	 *
	 * Parameters:
	 * vector<size_t> shape: size of each dimension
	 *
	 * Returns Tensor: new one-filled tensor
	 */
	static Tensor ones(std::vector<size_t> shape);

	/* Public, constant-filled tensor factory
	 *
	 * Parameters:
	 * vector<size_t> shape: size of each dimension
	 * float v: fill value
	 *
	 * Returns Tensor: new tensor filled with v
	 */
	static Tensor full(std::vector<size_t> shape, float v);

	/* Public, uniform-random tensor factory
	 *
	 * Parameters:
	 * vector<size_t> shape: size of each dimension
	 * float lo: inclusive lower bound
	 * float hi: inclusive upper bound
	 *
	 * Returns Tensor: new tensor with each element drawn uniformly from [lo, hi]
	 */
	static Tensor random(std::vector<size_t> shape, float lo, float hi);

	/* Public, stack same-shaped tensors along a new leading axis
	 *
	 * Parameters:
	 * vector<Tensor> items: tensors to stack; all must share the same shape
	 *
	 * Returns Tensor: new tensor of shape {items.size(), ...items[0].shape()}; throws
	 * TensorShapeError if items is empty or shapes differ
	 */
	static Tensor stack(const std::vector<Tensor>& items);

private:
	std::vector<size_t> shape_;
	std::vector<size_t> strides_;
	std::vector<float> data_;

	size_t flat_index(const std::vector<size_t>& idx) const;
};

/* Public, elementwise addition with numpy-style broadcasting
 *
 * Parameters:
 * Tensor a: left operand
 * Tensor b: right operand
 *
 * Returns Tensor: elementwise sum; throws TensorShapeError if shapes don't broadcast
 */
Tensor add(const Tensor& a, const Tensor& b);

/* Public, elementwise subtraction with numpy-style broadcasting
 *
 * Parameters:
 * Tensor a: left operand
 * Tensor b: right operand
 *
 * Returns Tensor: elementwise difference; throws TensorShapeError if shapes don't broadcast
 */
Tensor sub(const Tensor& a, const Tensor& b);

/* Public, elementwise multiplication with numpy-style broadcasting
 *
 * Parameters:
 * Tensor a: left operand
 * Tensor b: right operand
 *
 * Returns Tensor: elementwise product; throws TensorShapeError if shapes don't broadcast
 */
Tensor mul(const Tensor& a, const Tensor& b);

/* Public, elementwise division with numpy-style broadcasting
 *
 * Parameters:
 * Tensor a: left operand
 * Tensor b: right operand
 *
 * Returns Tensor: elementwise quotient; throws TensorShapeError if shapes don't broadcast
 */
Tensor div(const Tensor& a, const Tensor& b);

/* Public, scalar multiplication
 *
 * Parameters:
 * Tensor a: operand
 * float scalar: multiplier
 *
 * Returns Tensor: a with every element multiplied by scalar
 */
Tensor scale(const Tensor& a, float scalar);

/* Public, batched matrix multiplication
 *
 * Parameters:
 * Tensor a: left operand, shape {..., M, K}
 * Tensor b: right operand, shape {..., K, N}
 *
 * Returns Tensor: shape {..., M, N}, where leading "..." dimensions broadcast between
 * a and b; throws TensorShapeError if either operand has fewer than 2 dimensions, if
 * the K dimensions differ, or if the leading dimensions don't broadcast
 */
Tensor matmul(const Tensor& a, const Tensor& b);

/* Public, elementwise addition operator, alias for add(a, b) */
Tensor operator+(const Tensor& a, const Tensor& b);

/* Public, elementwise subtraction operator, alias for sub(a, b) */
Tensor operator-(const Tensor& a, const Tensor& b);

/* Public, elementwise multiplication operator, alias for mul(a, b) */
Tensor operator*(const Tensor& a, const Tensor& b);

/* Public, elementwise division operator, alias for div(a, b) */
Tensor operator/(const Tensor& a, const Tensor& b);

/* Public, scalar multiplication operator, alias for scale(a, scalar) */
Tensor operator*(const Tensor& a, float scalar);
Tensor operator*(float scalar, const Tensor& a);

/* Public, negation operator, alias for scale(a, -1.0f) */
Tensor operator-(const Tensor& a);

}  // namespace nn
