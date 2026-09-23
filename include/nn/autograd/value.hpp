/**
 * value.hpp
 * Author: Miguel Mochizuki Silva
 * Description: Reference-counted autograd graph node handle over Tensor
 */

#pragma once

#include "nn/tensor.hpp"

#include <memory>
#include <vector>
#include <functional>
#include <exception>

namespace nn {

/*
 * Thrown when backward() is called on a non-scalar Value.
 *
 * Attributes: none
 */
class AutogradShapeError : public std::exception {
public:
	const char* what() const noexcept override {
		return "backward() requires a scalar Value (size 1).";
	}
};

class Value;

/*
 * Internal graph node state, jointly owned by every Value handle referencing it.
 * Edges point one direction only, child to parent, so the graph structure itself
 * cannot form a cycle; see nn::Value::node for the one rule that must hold to
 * keep it that way.
 *
 * Attributes:
 * Tensor data: forward value
 * Tensor grad: accumulated dL/d(data), same shape as data
 * bool requires_grad: whether this node participates in backward
 * vector<Value> parents: inputs this node was computed from
 * function<void(const Tensor&)> backward_fn: given dL/d(this.data), accumulate into each parent's grad
 */
struct NodeImpl {
	Tensor data;
	Tensor grad;
	bool requires_grad;
	std::vector<Value> parents;
	std::function<void(const Tensor&)> backward_fn;

	NodeImpl(Tensor d, bool rg);
	~NodeImpl();
};

/*
 * Reference-counted handle to a graph node. Copyable, cheap (shared_ptr copy).
 * Two Values wrapping the same NodeImpl refer to the same tensor/grad/graph position.
 *
 * Attributes:
 * shared_ptr<NodeImpl> impl_: shared graph node state
 */
class Value {
public:
	/* Public, wrap raw data as a leaf node with no parents
	 *
	 * Parameters:
	 * Tensor data: initial tensor value
	 * bool requires_grad: track gradient if true
	 *
	 * Returns Value: new leaf handle
	 */
	static Value leaf(Tensor data, bool requires_grad = false);

	/* Public, construct a non-leaf node from parents and a backward rule; used only by
	 * autograd op implementations, never directly by layer/application code. backward_fn
	 * must capture only parents and saved Tensors, never the Value this call returns —
	 * capturing it would create a shared_ptr reference cycle
	 *
	 * Parameters:
	 * Tensor data: forward result
	 * vector<Value> parents: inputs this node was computed from
	 * function<void(const Tensor&)> backward_fn: local backward rule
	 *
	 * Returns Value: new non-leaf handle, requires_grad() true
	 */
	static Value node(Tensor data, std::vector<Value> parents, std::function<void(const Tensor&)> backward_fn);

	/* Public, forward value, read-write
	 *
	 * Parameters: none
	 *
	 * Returns Tensor&: reference to the node's data
	 */
	Tensor& data() const;

	/* Public, accumulated gradient, read-write
	 *
	 * Parameters: none
	 *
	 * Returns Tensor&: reference to the node's grad
	 */
	Tensor& grad() const;

	/* Public, whether this node participates in backward
	 *
	 * Parameters: none
	 *
	 * Returns bool: requires_grad flag
	 */
	bool requires_grad() const;

	/* Public, identity of the underlying graph node, for use as a visited-set key
	 *
	 * Parameters: none
	 *
	 * Returns NodeImpl*: raw pointer to the shared node state
	 */
	NodeImpl* impl() const;

	/* Public, run backward pass from this node
	 *
	 * Parameters: none
	 *
	 * Returns void: seeds this node's grad to 1, then runs every reachable node's
	 * backward_fn in reverse topological order; throws AutogradShapeError if this
	 * Value is not scalar (size() != 1)
	 */
	void backward() const;

	/* Public, reset this node's own grad to zero
	 *
	 * Parameters: none
	 *
	 * Returns void: none
	 */
	void zero_grad() const;

	/* Public, count of currently-live graph nodes, for leak testing
	 *
	 * Parameters: none
	 *
	 * Returns size_t: number of NodeImpl instances currently allocated
	 */
	static size_t live_node_count();

private:
	explicit Value(std::shared_ptr<NodeImpl> impl);
	std::shared_ptr<NodeImpl> impl_;
};

}  // namespace nn
